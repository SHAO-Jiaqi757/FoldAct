# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import threading

from verl import DataProto
from verl.tools.schemas import TrajectoryComponent, TrajectoryFeedback
from verl.utils.trajectory import get_components
from verl.utils.reward_score import default_compute_score
from verl.utils.event_ledger import EventLedger, EventType

# Basic logger (console)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File logging: use absolute path with process unique file to avoid multi-process confusion
import os
from datetime import datetime

def _ensure_file_logger() -> tuple[logging.Logger, str]:
    """Create/return a per-process file logger and its file path.
    - Absolute log dir: env REWARD_LOG_DIR or ./logs (abspath)
    - File name includes date and PID to disambiguate Ray workers
    """
    file_logger = logging.getLogger(f'file_logger.{os.getpid()}')
    if getattr(file_logger, '_configured', False):
        return file_logger, getattr(file_logger, '_log_path', '')

    log_dir = os.environ.get('REWARD_LOG_DIR', 'logs')
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'hallucination_penalty_reward_{timestamp}_{os.getpid()}.log')

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Clean old handlers to avoid duplicate additions
    for h in list(file_logger.handlers):
        file_logger.removeHandler(h)
    file_logger.addHandler(file_handler)
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    file_logger._configured = True  # type: ignore[attr-defined]
    file_logger._log_path = log_filename  # type: ignore[attr-defined]

    logger.info(f"Hallucination penalty reward manager logging to file: {log_filename}")
    return file_logger, log_filename

# Initialize module-level file logger (for use without instantiation)
file_logger, log_filename = _ensure_file_logger()

class HallucinationPenaltyRewardManager:
    """Reward manager that penalizes <information> (hallucination) and rewards <information_summary> when context has <information>."""
    
    def __init__(self, tokenizer, num_examine=100, compute_score=None, reward_fn_key="data_source", 
                 log_dir: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # Initialize file logger (if custom directory is provided, rebind to that directory)
        if log_dir is not None:
            os.environ['REWARD_LOG_DIR'] = log_dir
            self.file_logger, self.log_filename = _ensure_file_logger()
        else:
            self.file_logger, self.log_filename = _ensure_file_logger()
        
        # Configuration for the new reward system
        self.config = self._get_hallucination_penalty_config()
        # Override config with kwargs if provided
        if kwargs:
            self.config.update(kwargs)

        # In-memory caches
        self._cache_lock = threading.Lock()
        
        # Set up logging
        self._setup_logging()
        
        logger.info(f"HallucinationPenaltyRewardManager initialized.")
        logger.info(f"Reward config: per_step_distribution={self.config.get('per_step_distribution', 'last_token')}")
        self.file_logger.info(f"Reward file path: {self.log_filename}")
        self.file_logger.info(f"Reward config: per_step_distribution={self.config.get('per_step_distribution', 'last_token')}")
    
    def _get_hallucination_penalty_config(self):
        """Get configuration for hallucination penalty reward system"""
        return {
            "information_penalty": 0.0,              # Penalty for <information> components (hallucination)
            "information_summary_bonus": 0.0,           # Bonus for <information_summary> components with context
            "information_summary_ground_truth_bonus": 0.2,  # Bonus when summaries correctly echo ground-truth evidence
            "format_bonus": 0.1,                     # Bonus for proper format (sequence ends with answer)
            "no_answer_penalty": -0.2,               # Penalty when the agent never produces an answer component
            "enable_debug_logs": False,
            "reward_allocation_strategy": "component_based",
            "min_reward_value": -1.0,
            "max_reward_value": 2.0,
            "smooth_reward_transition": False,
            "step_level_allocation": True,
            "per_step_distribution": "even",  # Changed from "even" to "last_token" for stronger gradient signals
        }
    
    def _setup_logging(self):
        """Set up logging level and format"""
        if self.config.get("enable_debug_logs", True):
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger()
        if logger.handlers:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    
    
    
    

    
    
    def analyze_trajectory(self, components: List[TrajectoryComponent], 
                          ground_truth, question: str = None,
                          event_ledger: Optional[EventLedger] = None,
                          step_ids: Optional[torch.Tensor] = None) -> TrajectoryFeedback:
        """
        Analyze entire trajectory and calculate feedback.
        
        NEW: Accepts event_ledger for context-invariant reward computation.
        NEW: Accepts step_ids to correctly map components to turns.
        """
        logger.info(f"Analyzing trajectory with {len(components)} components")
        
        # Log to file
        self.file_logger.info(f"=== TRAJECTORY ANALYSIS ===")
        self.file_logger.info(f"Ground truth: {ground_truth}")
        self.file_logger.info(f"Question: {question}")
        self.file_logger.info(f"Components count: {len(components)}")
        if event_ledger:
            self.file_logger.info(f"Event ledger: {len(event_ledger)} events")
        
        # Extract key information
        search_components = [c for c in components if c.component_type == "search"]
        information_components = [c for c in components if c.component_type == "information"]
        information_summary_components = [c for c in components if c.component_type == "information_summary"]
        answer_components = [c for c in components if c.component_type == "answer"]
        
        logger.debug(f"Component breakdown: search={len(search_components)}, "
                    f"information={len(information_components)}, "
                    f"information_summary={len(information_summary_components)}, "
                    f"answer={len(answer_components)}")
        
        # Log component distribution to file
        self.file_logger.info(f"Component breakdown:")
        self.file_logger.info(f"  Search components: {len(search_components)}")
        self.file_logger.info(f"  Information components: {len(information_components)}")
        self.file_logger.info(f"  Information summary components: {len(information_summary_components)}")
        self.file_logger.info(f"  Answer components: {len(answer_components)}")
        
        # Analyze hallucination penalty and information summary rewards (with ledger and step_ids)
        hallucination_analysis = self._analyze_hallucination_penalty(
            components,
            question,
            event_ledger,
            step_ids,
            ground_truth=ground_truth
        )
        
        # Compute component scores
        answer_quality_score = self._score_answer_quality_simplified(answer_components, ground_truth)
        
        logger.info(f"Component scores - Answer: {answer_quality_score:.3f}")
        
        # Log penalty information to file
        self.file_logger.info(f"Penalty analysis:")
        self.file_logger.info(f"  Information penalty: {hallucination_analysis['information_penalty']:.3f}")
        self.file_logger.info(f"  Information summary bonus: {hallucination_analysis['information_summary_bonus']:.3f}")
        
        # Create trajectory feedback
        feedback = TrajectoryFeedback(
            trajectory_id="trajectory_1",
            components=components,
            final_answer=answer_components[-1].content if answer_components else "",
            ground_truth=ground_truth,
            answer_quality_score=answer_quality_score,
        )
        
        # Store hallucination analysis results as class attribute
        feedback._hallucination_analysis = hallucination_analysis
        
        # Log hallucination analysis results to file
        self.file_logger.info(f"Hallucination analysis results:")
        for key, value in hallucination_analysis.items():
            self.file_logger.info(f"  {key}: {value}")
        
        self.file_logger.info("=" * 50)
        
        return feedback
    
    def _contains_information_tag(self, text: Optional[str]) -> bool:
        """Return True if text contains an <information>...</information> block."""
        if not text:
            return False
        # Optimize: use simple string search instead of regex for better performance
        return "<information>" in text and "</information>" in text

    def _is_model_generated_information(self, component: TrajectoryComponent) -> bool:
        """Identify information blocks produced by the model (not environment observations)."""
        if component.component_type == "information":
            # Optimize: avoid lower() conversion and use faster string check
            content = component.content or ""
            return "[turn" not in content  # Observations always include turn headers
        return self._contains_information_tag(component.content)

    def _get_turn_id_from_component(self, component: TrajectoryComponent, 
                                    step_ids: Optional[torch.Tensor]) -> int:
        """
        Get turn ID for a component from step_ids tensor.
        
        Args:
            component: The component to get turn ID for
            step_ids: Tensor mapping token indices to turn IDs
        
        Returns:
            turn_id: The turn this component belongs to
        """
        if step_ids is not None:
            # Get turn_id from step_ids using component's start token
            start_idx = component.start_token_idx
            if start_idx < len(step_ids):
                return int(step_ids[start_idx].item() if hasattr(step_ids[start_idx], 'item') else step_ids[start_idx])
        
        # Fallback: use metadata or assume sequential
        return component.metadata.get('turn_id', 0)
    
    def _analyze_hallucination_penalty(self, components: List[TrajectoryComponent], 
                                      question: str = None,
                                      event_ledger: Optional[EventLedger] = None,
                                      step_ids: Optional[torch.Tensor] = None,
                                      ground_truth: Optional[Any] = None) -> Dict:
        """
        Analyze hallucination penalty and information summary rewards.
        
        NEW: Uses event ledger (full context) instead of visible components for evidence checking.
        NEW: Uses step_ids to correctly map components to turns.
        This ensures that rewards are based on true state, not on what context compression shows.
        """

        observation_information_indices = []
        model_information_indices = []
        information_summary_indices = []
        information_summary_ground_truth_component_ids: List[int] = []

        # Optimize: single pass through components, cache results
        for idx, component in enumerate(components):
            if component.component_type == "information":
                # Cache the result to avoid repeated calls
                is_model_generated = self._is_model_generated_information(component)
                if is_model_generated:
                    model_information_indices.append(idx)
                else:
                    observation_information_indices.append(idx)
            elif component.component_type == "information_summary":
                information_summary_indices.append(idx)
            else:
                # Only check for model-generated information if not already classified
                if self._is_model_generated_information(component):
                    model_information_indices.append(idx)

        summaries_with_evidence = 0
        summaries_without_evidence = 0
        
        # Prepare normalized ground-truth strings for bonus computation
        normalized_gt_parts: List[str] = []
        if ground_truth is not None:
            normalized_gt_parts = [
                gt.lower()
                for gt in self._extract_ground_truth_parts(ground_truth)
                if isinstance(gt, str) and gt.strip()
            ]

        # NEW: Use event ledger for ground truth evidence checking
        if event_ledger is not None:
            self.file_logger.info(f"[Event Ledger] Using ledger with {len(event_ledger)} events for evidence checking")
            
            for summary_idx in information_summary_indices:
                summary_component = components[summary_idx]
                # Get the turn_id using step_ids (CORRECT WAY)
                turn_id = self._get_turn_id_from_component(summary_component, step_ids)
                
                # Check if there is environment evidence BEFORE this turn in the ledger
                has_evidence = event_ledger.has_evidence_before_turn(turn_id)
                
                if has_evidence:
                    summaries_with_evidence += 1
                    self.file_logger.info(f"[Event Ledger] Summary at turn {turn_id} HAS evidence in ledger")
                else:
                    summaries_without_evidence += 1
                    self.file_logger.info(f"[Event Ledger] Summary at turn {turn_id} has NO evidence in ledger")

            # Ground-truth consistency reward: summary must echo evidence containing ground truth
            if normalized_gt_parts and information_summary_indices:
                ledger_evidence = [
                    evidence.lower()
                    for evidence in event_ledger.get_all_evidence()
                    if isinstance(evidence, str)
                ]
                gt_present_in_evidence = {
                    gt for gt in normalized_gt_parts
                    if len(gt) >= 3 and any(gt in evidence for evidence in ledger_evidence)
                }
                if gt_present_in_evidence:
                    for summary_idx in information_summary_indices:
                        summary_component = components[summary_idx]
                        summary_text = (summary_component.content or "").lower()
                        if any(gt in summary_text for gt in gt_present_in_evidence):
                            information_summary_ground_truth_component_ids.append(id(summary_component))
                            self.file_logger.info(
                                "[Reward] Information summary echoes ground truth present in evidence "
                                f"(component_idx={summary_idx})"
                            )
        else:
            # FALLBACK: Use old method based on visible components (less accurate with compression)
            self.file_logger.warning("[Event Ledger] No ledger provided, falling back to component-based checking (may be inaccurate with compression)")
            
            for summary_idx in information_summary_indices:
                has_prior_observation_info = any(info_idx < summary_idx for info_idx in observation_information_indices)
                if has_prior_observation_info:
                    summaries_with_evidence += 1
                else:
                    summaries_without_evidence += 1

        information_penalty = self.config["information_penalty"] if model_information_indices else 0.0
        
        # NEW: Reward/penalize based on whether evidence exists in ledger (ground truth)
        if len(information_summary_indices) > 0:
            if summaries_with_evidence > 0:
                # Has evidence: give bonus
                information_summary_bonus = self.config["information_summary_bonus"]
                self.file_logger.info(f"[Reward] Information summary bonus: {information_summary_bonus} (evidence found in ledger)")
            else:
                information_summary_bonus = 0.0
        else:
            information_summary_bonus = 0.0

        information_summary_ground_truth_bonus = 0.0
        if information_summary_ground_truth_component_ids:
            total_bonus = float(self.config.get("information_summary_ground_truth_bonus", 0.0) or 0.0)
            if total_bonus:
                information_summary_ground_truth_bonus = total_bonus / len(information_summary_ground_truth_component_ids)
                self.file_logger.info(
                    "[Reward] Information summary ground-truth bonus per component: "
                    f"{information_summary_ground_truth_bonus:.3f} "
                    f"(matches={len(information_summary_ground_truth_component_ids)}, total={total_bonus:.3f})"
                )
        
        return {
            "information_penalty": information_penalty,
            "information_summary_bonus": information_summary_bonus,
            "information_summary_with_evidence": summaries_with_evidence,
            "information_summary_without_evidence": summaries_without_evidence,
            "model_information_indices": model_information_indices,
            "observation_information_indices": observation_information_indices,
            "information_summary_ground_truth_bonus": information_summary_ground_truth_bonus,
            "information_summary_ground_truth_component_ids": information_summary_ground_truth_component_ids,
        }
    
    
    
    def _extract_ground_truth_parts(self, ground_truth) -> List[str]:
        """Normalize ground truth into a list of candidate strings for matching."""
        parts: List[str] = []
        if isinstance(ground_truth, str):
            # Process separator format
            if "<|answer_split|>" in ground_truth:
                parts = [gt.strip() for gt in ground_truth.split("<|answer_split|>")]
            else:
                parts = [ground_truth.strip()]
        elif isinstance(ground_truth, dict):
            # Extract answer list from common fields
            candidate_keys = ["target", "targets", "answers", "answer", "labels"]
            gt_values = []
            for k in candidate_keys:
                if k in ground_truth:
                    v = ground_truth[k]
                    if isinstance(v, (list, np.ndarray, set, tuple)):
                        gt_values.extend(list(v))
                    else:
                        gt_values.append(v)
            if not gt_values:
                gt_values = [ground_truth]
            parts = [str(gt).strip() for gt in gt_values]
        elif isinstance(ground_truth, (list, np.ndarray, set, tuple)):
            parts = [str(gt).strip() for gt in ground_truth]
        else:
            parts = [str(ground_truth).strip()]
        return [p for p in parts if isinstance(p, str) and p.strip()]

    def _score_answer_quality_simplified(self, answer_components: List[TrajectoryComponent], 
                                       ground_truth) -> float:
        """Simplified answer quality score: only consider exact matches"""
        if not answer_components:
            return 0.0
        
        final_answer = answer_components[-1].content.strip()
        # If the decoded answer block still contains tags, extract inner text
        try:
            m = re.search(r"<answer>(.*?)</answer>", final_answer, re.DOTALL)
            if m:
                final_answer = m.group(1).strip()
        except Exception:
            pass
        self.file_logger.debug(f"Scoring final answer: {final_answer[:100]}...")
        
        gt_parts = self._extract_ground_truth_parts(ground_truth)
        
        logger.debug(f"Ground truth parts: {gt_parts}")
        
        # Only check exact matches (case-insensitive)
        fa_lower = final_answer.lower()
        for gt_part in gt_parts:
            if not gt_part:
                continue
            if fa_lower == gt_part.lower():
                logger.debug(f"Exact match found: {final_answer} == {gt_part}")
                return 1.0
        
        # No exact match, return 0
        logger.debug(f"No exact match found for: {final_answer}")
        return 0.0
    
    def create_dense_reward_tensor(self, feedback: TrajectoryFeedback, 
                                  response_length: int) -> torch.Tensor:
        """Create dense reward tensor with hallucination penalty allocation strategy."""
        try:
            logger.info(f"Creating dense reward tensor for response length: {response_length}")
            
            # Log to file
            self.file_logger.info(f"=== CREATING DENSE REWARD TENSOR ===")
            self.file_logger.info(f"Response length: {response_length}")
            self.file_logger.info(f"Components count: {len(feedback.components)}")
            self.file_logger.info(f"Reward allocation strategy: {self.config['reward_allocation_strategy']}")
            
            # Parameter validation
            if response_length <= 0:
                logger.warning(f"Invalid response_length: {response_length}, using default 500")
                response_length = 500
            
            # Create reward tensor
            reward_tensor = torch.zeros(response_length, dtype=torch.float32)
            
            if not feedback.components:
                self.file_logger.info("No components found, applying uniform score: 0.0")
                self.file_logger.info("=" * 50)
                return reward_tensor
            
            # Use hallucination penalty reward allocation strategy
            reward_tensor = self._allocate_rewards_hallucination_penalty(feedback, response_length)
            distribution_mode = self.config.get("per_step_distribution", "last_token")
            self.file_logger.info(f"Used hallucination penalty reward allocation (per_step_distribution={distribution_mode})")
            logger.info(f"Reward allocation: per_step_distribution={distribution_mode}")
            
            # Apply smooth transition
            if self.config["smooth_reward_transition"]:
                reward_tensor = self._smooth_reward_transitions(reward_tensor)
                self.file_logger.info("Applied smooth reward transitions")
            
            # Limit reward range
            reward_tensor = torch.clamp(reward_tensor, 
                                        self.config["min_reward_value"], 
                                        self.config["max_reward_value"])
            
            # Log reward tensor details to file
            self._log_reward_tensor_details(reward_tensor, feedback)
            
            return reward_tensor
            
        except Exception as e:
            logger.error(f"Error creating dense reward tensor: {e}")
            # Return default reward tensor
            default_reward = torch.zeros(response_length, dtype=torch.float32)
            default_reward[:] = 0.5
            return default_reward
    
    def _allocate_rewards_hallucination_penalty(self, feedback: TrajectoryFeedback, response_length: int) -> torch.Tensor:
        """Allocate rewards based on hallucination penalty strategy"""
        reward_tensor = torch.zeros(response_length, dtype=torch.float32)
        
        # Get hallucination analysis results
        hallucination_meta = getattr(feedback, '_hallucination_analysis', {})
        # Sort components by token positions
        sorted_components = sorted(feedback.components, key=lambda x: x.start_token_idx)

        def _clamp_span(start: int, end: int) -> Optional[tuple[int, int]]:
            """Clamp a token span to [0, response_length). Return None if empty after clamp."""
            try:
                if end <= 0:
                    return None
                s = max(0, min(int(start), max(0, response_length - 1)))
                e = max(s + 1, min(int(end), response_length))
                if s >= response_length or e <= 0 or e <= s:
                    return None
                return s, e
            except Exception:
                return None

        def _distribute_within_span(s: int, e: int, score: float):
            """Distribute a step-level score within a token span without length bias"""
            length = max(0, e - s)
            if length <= 0:
                return
            mode = str(self.config.get("per_step_distribution", "even")).lower()
            if mode == "last_token":
                reward_tensor[e - 1] += float(score)
            else:  # even
                per = float(score) / float(length)
                reward_tensor[s:e] += per
        
        for i, component in enumerate(sorted_components):
            # Compute clamped token span within [0, response_length]
            span = _clamp_span(component.start_token_idx, component.end_token_idx)
            if span is None:
                continue
            start_idx, end_idx = span
            
            # Apply hallucination penalty reward strategy
            final_score = self._apply_hallucination_penalty_rewards(
                component, feedback, hallucination_meta, i
            )
            
            # Allocate reward
            if bool(self.config.get("step_level_allocation", True)):
                _distribute_within_span(start_idx, end_idx, final_score)
            else:
                reward_tensor[start_idx:end_idx] = final_score
            
            # Log to file
            distribution_mode = self.config.get("per_step_distribution", "last_token")
            reward_pos = f"pos {end_idx-1}" if distribution_mode == "last_token" else f"span {start_idx}-{end_idx}"
            self.file_logger.info(f"Component {i+1} ({component.component_type}): "
                                f"final_score={final_score:.3f}, reward at {reward_pos}")
        
        # Apply no-answer penalty if trajectory never produced an answer component
        if response_length > 0 and not any(c.component_type == "answer" for c in feedback.components):
            no_answer_penalty = float(self.config.get("no_answer_penalty", 0.0) or 0.0)
            if no_answer_penalty != 0.0:
                penalty_position = response_length - 1
                reward_tensor[penalty_position] += no_answer_penalty
                logger.info(f"Applied no_answer_penalty={no_answer_penalty:.3f} at token {penalty_position}")
                self.file_logger.info(
                    f"Applied no_answer_penalty={no_answer_penalty:.3f} at token {penalty_position} "
                    f"(missing answer component)"
                )
        
        return reward_tensor
    
    def _apply_hallucination_penalty_rewards(self, component: TrajectoryComponent, 
                                            feedback: TrajectoryFeedback, 
                                            hallucination_meta: Dict, 
                                            component_idx: int) -> float:
        """Apply hallucination penalty rewards based on component type"""
        score = 0.0
        debug_parts = []
        gt_component_id_set = set(hallucination_meta.get("information_summary_ground_truth_component_ids", []))
        
        try:
            debug_parts.append(f"base=0.0")
        except Exception:
            pass
        
        # 1. Information summary reward/penalty based on context availability
        if component.component_type == "information_summary":
            bonus = hallucination_meta.get("information_summary_bonus", 0.0)
            if bonus:
                score += bonus
                try:
                    debug_parts.append(f"information_summary_bonus=+{bonus:.3f}")
                except Exception:
                    pass

            if gt_component_id_set and id(component) in gt_component_id_set:
                gt_bonus = hallucination_meta.get("information_summary_ground_truth_bonus", 0.0)
                if gt_bonus:
                    score += gt_bonus
                    try:
                        debug_parts.append(f"ground_truth_bonus=+{gt_bonus:.3f}")
                    except Exception:
                        pass
                    logger.debug(
                        f"Rewarding information summary for correctly extracting ground truth: {score:.3f}"
                    )
          
        # 2. Information components receive hallucination penalty when generated by the model
        elif self._is_model_generated_information(component):
            penalty = hallucination_meta.get("information_penalty", 0.0)
            if penalty:
                score += penalty
                try:
                    debug_parts.append(f"information_penalty={penalty:.3f}")
                except Exception:
                    pass
                logger.debug(f"Penalizing model-generated information component: {score:.3f}")
        
        # 3. Answer quality bonus
        elif component.component_type == "answer":
            answer_score = feedback.answer_quality_score 
            score += answer_score
            try:
                debug_parts.append(f"answer_quality=+{answer_score:.3f}")
            except Exception:
                pass
            logger.debug(f"Rewarding answer quality: {score:.3f}")
            
            # Add format bonus for final answer (if this is the last answer component)
            answer_components = [c for c in feedback.components if c.component_type == "answer"]
            is_last_answer = False
            if answer_components:
                sorted_answer_components = sorted(answer_components, key=lambda x: x.start_token_idx)
                is_last_answer = (component.start_token_idx == sorted_answer_components[-1].start_token_idx)
            
            if is_last_answer:
                format_bonus = self.config.get("format_bonus")
                if format_bonus:
                    score += format_bonus
                    try:
                        debug_parts.append(f"format_bonus=+{format_bonus:.3f}")
                    except Exception:
                        pass
                    logger.debug(f"Rewarding proper format (sequence ends with answer): {score:.3f}")
                    self.file_logger.info(f"Applied format bonus {format_bonus:.3f} for final answer component")
        # Emit a compact score breakdown line for this component
        try:
            logger.debug(
                f"Score breakdown for component {component_idx+1} ({component.component_type}): "
                + "; ".join(debug_parts)
                + f"; score={score:.3f}"
            )
        except Exception:
            pass
        
        return score
    
    def _smooth_reward_transitions(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        """Smooth reward transitions to avoid sudden changes"""
        if len(reward_tensor) < 3:
            return reward_tensor
        
        # Use simple moving average to smooth
        smoothed = reward_tensor.clone()
        kernel_size = 3
        
        for i in range(1, len(reward_tensor) - 1):
            start_idx = max(0, i - kernel_size // 2)
            end_idx = min(len(reward_tensor), i + kernel_size // 2 + 1)
            smoothed[i] = reward_tensor[start_idx:end_idx].mean()
        
        return smoothed
    
    def _log_reward_tensor_details(self, reward_tensor: torch.Tensor, feedback: TrajectoryFeedback):
        """Log reward tensor details to file"""
        self.file_logger.info(f"Reward tensor details:")
        self.file_logger.info(f"  Shape: {reward_tensor.shape}")
        self.file_logger.info(f"  Min value: {reward_tensor.min().item():.3f}")
        self.file_logger.info(f"  Max value: {reward_tensor.max().item():.3f}")
        self.file_logger.info(f"  Mean value: {reward_tensor.mean().item():.3f}")
        
        # Count non-zero rewards and their distribution
        non_zero_rewards = reward_tensor[reward_tensor != 0]
        self.file_logger.info(f"  Non-zero rewards: {len(non_zero_rewards)}")
        if len(non_zero_rewards) > 0:
            self.file_logger.info(f"  Non-zero min: {non_zero_rewards.min().item():.3f}")
            self.file_logger.info(f"  Non-zero max: {non_zero_rewards.max().item():.3f}")
            self.file_logger.info(f"  Non-zero mean: {non_zero_rewards.mean().item():.3f}")
            self.file_logger.info(f"  Non-zero std: {non_zero_rewards.std().item():.3f}")
        
        # Log reward allocation for each component
        if feedback.components:
            self.file_logger.info(f"  Component reward allocation:")
            for i, component in enumerate(feedback.components):
                start_idx = component.start_token_idx
                end_idx = min(component.end_token_idx, len(reward_tensor))
                if start_idx < len(reward_tensor):
                    component_rewards = reward_tensor[start_idx:end_idx]
                    if len(component_rewards) > 0:
                        self.file_logger.info(f"    {component.component_type} (step {component.step_number}): "
                                       f"tokens {start_idx}-{end_idx}, "
                                       f"reward range [{component_rewards.min().item():.3f}, {component_rewards.max().item():.3f}], "
                                       f"mean {component_rewards.mean().item():.3f}")
        
        self.file_logger.info("=" * 50)

    def __call__(self, data: DataProto, return_dict=False):
        """Process batch data and return dense reward tensors (Synchronous version)"""
        logger.info(f"Processing batch with {len(data)} items")

        # Log to file
        self.file_logger.info(f"=== BATCH PROCESSING START ===")
        self.file_logger.info(f"Batch size: {len(data)}")
        self.file_logger.info(f"Return dict: {return_dict}")

        # Pre-compute target output shape
        try:
            batch_responses = data.batch.get("responses", None)
            if isinstance(batch_responses, torch.Tensor) and batch_responses.dim() >= 2:
                batch_size, resp_len = int(batch_responses.shape[0]), int(batch_responses.shape[1])
            else:
                batch_size, resp_len = int(len(data) or 0), 0
        except Exception:
            batch_size, resp_len = int(len(data) or 0), 0

        # Safe check: empty batch
        if not data or len(data) == 0:
            logger.warning("Empty data batch received")
            file_logger.warning("Empty data batch received")
            empty = torch.zeros((batch_size, resp_len), dtype=torch.float32)
            return {"reward_tensor": empty} if return_dict else empty

        dense_reward_tensors = []
        try:
            for i in range(len(data)):
                try:
                    data_item = data[i]
                    self.file_logger.info(f"--- Processing item {i+1}/{len(data)} ---")
                    
                    # Safe check data item
                    if not hasattr(data_item, 'batch') or not hasattr(data_item, 'non_tensor_batch'):
                        logger.warning(f"Data item {i} missing required attributes, skipping")
                        self.file_logger.warning(f"Data item {i} missing required attributes, skipping")
                        continue
                    
                    # Check necessary fields. If missing, fall back to zeros for this item.
                    required_fields = ["prompts", "responses", "attention_mask"]
                    missing_required = False
                    for field in required_fields:
                        if field not in data_item.batch:
                            missing_required = True
                            logger.warning(f"Data item {i} missing required field: {field}, using zero rewards for this item")
                            self.file_logger.warning(f"Data item {i} missing required field: {field}, using zero rewards for this item")
                            break
                    if missing_required:
                        dense_reward_tensors.append(torch.zeros(resp_len, dtype=torch.float32))
                        continue

                    # reward_model metadata may be absent in some configs; do not hard fail.
                    if "reward_model" not in data_item.non_tensor_batch:
                        logger.warning(f"Data item {i} missing reward_model field, using zero rewards for this item")
                        self.file_logger.warning(f"Data item {i} missing reward_model field, using zero rewards for this item")
                        dense_reward_tensors.append(torch.zeros(resp_len, dtype=torch.float32))
                        continue
                    
                    # Process single data item (Synchronous call)
                    result = self._process_single_item_sync(data_item, i)
                    if result:
                        dense_reward_tensors.append(result["reward_tensor"])
                        self.file_logger.info(f"Item {i+1} processed successfully")
                    else:
                        self.file_logger.warning(f"Item {i+1} processing returned None")
                    
                except Exception as e:
                    logger.error(f"Error processing data item {i}: {e}")
                    self.file_logger.error(f"Error processing data item {i}: {e}")
                    # Create default reward tensor as fallback with correct response length
                    dense_reward_tensors.append(torch.zeros(resp_len, dtype=torch.float32))
                    continue
            
            # Ensure we have an output even if no items produced a reward
            if not dense_reward_tensors:
                logger.warning("No valid reward tensors generated; returning zeros of expected shape")
                stacked_rewards = torch.zeros((batch_size, resp_len), dtype=torch.float32)
            else:
                # Align each sample's reward length to its response length (no fixed cap)
                if not isinstance(data.batch.get("responses", None), torch.Tensor):
                    raise ValueError("Batch is missing 'responses' tensor for shaping reward output")

                # Build full batch-sized tensor and fill from available per-item rewards
                stacked_rewards = torch.zeros((batch_size, resp_len), dtype=torch.float32)
                fill_count = min(batch_size, len(dense_reward_tensors))
                for idx in range(fill_count):
                    rt = dense_reward_tensors[idx]
                    if rt.numel() >= resp_len:
                        stacked_rewards[idx, :] = rt[:resp_len]
                    else:
                        stacked_rewards[idx, : rt.numel()] = rt
            logger.info(f"Final stacked rewards shape: {stacked_rewards.shape}")
            
            # Return result
            if return_dict:
                result = {"reward_tensor": stacked_rewards}
                logger.info(f"Returning dict with keys: {list(result.keys())}")
                return result
            else:
                logger.info(f"Returning tensor directly: {stacked_rewards.shape}")
                return stacked_rewards
                
        except Exception as e:
            logger.error(f"Critical error in reward manager: {e}")
            # Return default values with best-effort shape
            fallback = torch.zeros((batch_size, resp_len), dtype=torch.float32)
            return {"reward_tensor": fallback} if return_dict else fallback
    
    def _process_single_item_sync(self, data_item, item_index):
        """Process single data item, return feedback and reward tensor (Synchronous version)"""
        try:
            # Log to file (only in debug mode to reduce I/O overhead)
            if self.config.get("enable_debug_logs", False):
                self.file_logger.info(f"=== PROCESSING SINGLE ITEM {item_index+1} ===")
            
            # Get response data
            response_ids = data_item.batch["responses"]
            response_ids_shape = response_ids.shape
            
            # Calculate prompt length
            prompts_shape = data_item.batch["prompts"].shape
            
            # Safe get prompt length
            if len(prompts_shape) >= 2:
                prompt_length = prompts_shape[1]
            elif len(prompts_shape) == 1:
                prompt_length = prompts_shape[0]
            else:
                prompt_length = 0
                logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
                self.file_logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
            
            
            # Get attention mask
            attention_mask_shape = data_item.batch["attention_mask"].shape
            
            # Safe remove batch dimensions
            if len(attention_mask_shape) >= 2:
                attention_mask = data_item.batch["attention_mask"][0]
            elif len(attention_mask_shape) == 1:
                attention_mask = data_item.batch["attention_mask"]
            else:
                logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                self.file_logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                attention_mask = torch.ones(1000, dtype=torch.long)
            
            # Ensure attention_mask is 1D
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            if attention_mask.dim() != 1:
                raise ValueError(f"Invalid attention_mask dimensions: {attention_mask.dim()}")
            
            
            # Calculate valid response length
            if prompt_length > 0 and prompt_length < len(attention_mask):
                valid_response_length = attention_mask[prompt_length:].sum()
            else:
                valid_response_length = 100
            
            # Ensure length is a valid integer
            if hasattr(valid_response_length, 'item'):
                valid_response_length = valid_response_length.item()
                valid_response_length = int(valid_response_length)
            
            
            # Get valid response IDs
            if len(response_ids_shape) >= 2:
                total_len = int(response_ids.shape[1])
                if valid_response_length > 0 and valid_response_length <= total_len:
                    valid_response_ids = response_ids[0, :valid_response_length]
                else:
                    # Fallback to the maximum available length
                    valid_response_ids = response_ids[0, :total_len]
            else:
                total_len = int(len(response_ids))
                if valid_response_length > 0 and valid_response_length <= total_len:
                    valid_response_ids = response_ids[:valid_response_length]
                else:
                    valid_response_ids = response_ids[:total_len]
            
            # Ensure valid_response_ids is 1D tensor
            if valid_response_ids.dim() > 1:
                valid_response_ids = valid_response_ids.flatten()
            
            # Decode response
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")

            # NEW: Extract event ledger from non_tensor_batch (PER-ITEM)
            # Event ledgers are stored as numpy object array in non_tensor_batch['event_ledger']
            # Each item in the batch has its corresponding ledger dict
            event_ledger = None
            if hasattr(data_item, 'non_tensor_batch') and 'event_ledger' in data_item.non_tensor_batch:
                try:
                    # Extract ledger dict for this specific item
                    ledger_data = data_item.non_tensor_batch['event_ledger']
                    
                    # ledger_data should be a dict (from numpy object array element)
                    if isinstance(ledger_data, dict):
                        ledger_dict = ledger_data
                    elif isinstance(ledger_data, (list, np.ndarray)) and len(ledger_data) > 0:
                        # If it's an array/list, take the first element (single item case)
                        ledger_dict = ledger_data[0] if isinstance(ledger_data, list) else ledger_data.tolist()[0]
                    else:
                        ledger_dict = None
                    
                    if ledger_dict:
                        event_ledger = EventLedger.from_dict(ledger_dict)
                        self.file_logger.info(f"[Event Ledger] Successfully loaded ledger with {len(event_ledger)} events")
                        logger.info(f"[Event Ledger] Loaded ledger with {len(event_ledger)} events for item {item_index}")
                    else:
                        self.file_logger.warning(f"[Event Ledger] Invalid ledger data format: {type(ledger_data)}")
                except Exception as e:
                    logger.warning(f"[Event Ledger] Failed to load event ledger for item {item_index}: {e}")
                    self.file_logger.warning(f"[Event Ledger] Failed to load event ledger: {e}")
                    import traceback
                    self.file_logger.warning(f"[Event Ledger] Traceback: {traceback.format_exc()}")
            else:
                self.file_logger.warning(f"[Event Ledger] No event ledger found in data item {item_index} (will use fallback method)")
                logger.warning(f"[Event Ledger] No event ledger in non_tensor_batch for item {item_index}")

            # Robustly resolve data source with fallbacks
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, None)
            try:
                import numpy as _np  # local alias to avoid top-level import assumptions
                if isinstance(data_source, (_np.ndarray, list)) and len(data_source) > 0:
                    data_source = data_source[0]
            except Exception:
                pass
            if not data_source or data_source == "unknown":
                # Common fallbacks
                data_source = (
                    data_item.non_tensor_batch.get("data_source")
                    or (data_item.non_tensor_batch.get("extra_info", {}) or {}).get("split")
                    or data_item.non_tensor_batch.get("ability")
                    or data_item.non_tensor_batch.get("split")
                    or data_item.non_tensor_batch.get("id")
                    or "unknown"
                )
            
            # Try to get question (if exists)
            question = None
            if "question" in data_item.non_tensor_batch:
                question = data_item.non_tensor_batch.get("question", None)
            elif "prompt" in data_item.non_tensor_batch:
                # Extract question content from prompt field
                prompt_data = data_item.non_tensor_batch.get("prompt", None)
                if isinstance(prompt_data, (list, np.ndarray)) and len(prompt_data) > 0:
                    # prompt is message list, extract content of first user message
                    first_message = prompt_data[0] if isinstance(prompt_data, list) else prompt_data.tolist()[0]
                    if isinstance(first_message, dict) and "content" in first_message:
                        question = first_message["content"]
                    elif isinstance(first_message, (list, np.ndarray)) and len(first_message) > 0:
                        # Process nested structure
                        first_content = first_message[0] if isinstance(first_message, list) else first_message.tolist()[0]
                        if isinstance(first_content, dict) and "content" in first_content:
                            question = first_content["content"]
            elif "raw_prompt" in data_item.non_tensor_batch:
                # Extract question content from raw_prompt field
                raw_prompt = data_item.non_tensor_batch.get("raw_prompt", None)
                if isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
                    # raw_prompt is message list, extract content of first user message
                    first_message = raw_prompt[0] if isinstance(raw_prompt, list) else raw_prompt.tolist()[0]
                    if isinstance(first_message, dict) and "content" in first_message:
                        question = first_message["content"]
                    elif isinstance(first_message, (list, np.ndarray)) and len(first_message) > 0:
                        # Process nested structure
                        first_content = first_message[0] if isinstance(first_message, list) else first_message.tolist()[0]
                        if isinstance(first_content, dict) and "content" in first_content:
                            question = first_content["content"]

            
            # logger.info(f"Data source: {data_source}")
            # logger.info(f"Ground truth: {ground_truth}")
            # logger.info(f"Question: {question}")
            # logger.info(f"Response: {response_str}")
            
            # # Log to file
            # self.file_logger.info(f"Data source: {data_source}")
            # self.file_logger.info(f"Ground truth: {ground_truth}")
            # self.file_logger.info(f"Question: {question}")
            # self.file_logger.info(f"Response: {response_str}")
            
            # Parse trajectory components
            components = get_components(
                data_item.batch["responses"],
                data_item.batch["step_ids"],
                data_item.batch["responses_types"],
                data_item.batch["attention_mask"],
                self.tokenizer,
            )
            
            # Analyze entire trajectory (with event ledger for context-invariant rewards and step_ids for turn mapping)
            step_ids_tensor = data_item.batch.get("step_ids")
            feedback = self.analyze_trajectory(components, ground_truth, question, event_ledger, step_ids_tensor)
            
            # Create dense reward tensor with no hard cap on length
            # Align reward length to the actual component spans to prevent clamping
            try:
                max_span = max((c.end_token_idx for c in components), default=0)
                target_length = int(max_span)
                # As a safety net, ensure at least the visible decoded length
                visible_len = int(getattr(valid_response_ids, 'shape', [0])[0]) if hasattr(valid_response_ids, 'shape') else int(valid_response_length)
                if visible_len > 0:
                    target_length = max(target_length, visible_len)
            except Exception:
                # Fallback: use computed valid_response_length if available; otherwise default to 0 (handled downstream)
                target_length = int(valid_response_length) if 'valid_response_length' in locals() else 0
            dense_reward = self.create_dense_reward_tensor(feedback, target_length)
            
            # Print analysis summary
            # self._print_analysis_summary(data_source, item_index, response_str, ground_truth, components, feedback, dense_reward)
            
            self.file_logger.info(f"Item {item_index+1} processing completed successfully")
            
            return {
                "feedback": feedback,
                "reward_tensor": dense_reward
            }
            
        except Exception as e:
            logger.error(f"Error processing single item {item_index}: {e}")
            self.file_logger.error(f"Error processing single item {item_index}: {e}")
            return None
    
    def _print_analysis_summary(self, data_source, item_index, response_str, ground_truth, components, feedback, dense_reward):
        """Print analysis summary"""
        self.file_logger.info(f"\n{'='*80}")
        self.file_logger.info(f"[{data_source}] Hallucination Penalty Analysis for Item {item_index+1}:")
        self.file_logger.info(f"{'='*80}")
        self.file_logger.info(f"Response: {response_str[:300]}...")
        self.file_logger.info(f"Ground Truth: {ground_truth}")
        self.file_logger.info(f"Components Found: {len(components)}")
        for j, comp in enumerate(components):
            self.file_logger.info(f"  {j+1}. {comp.component_type}: {comp.content[:100]}...")
        self.file_logger.info(f"\nScores:")
        self.file_logger.info(f"  Answer Quality: {feedback.answer_quality_score:.3f}")
        self.file_logger.info(f"\nHallucination Analysis:")
        hallucination_meta = getattr(feedback, '_hallucination_analysis', {})
        self.file_logger.info(f"  Information Penalty: {hallucination_meta.get('information_penalty', 0.0):.3f}")
        self.file_logger.info(f"  Information Summary Bonus: {hallucination_meta.get('information_summary_bonus', 0.0):.3f}")
