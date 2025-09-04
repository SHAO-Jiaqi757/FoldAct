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
import asyncio
import threading

from verl import DataProto
from verl.tools.schemas import TrajectoryComponent, TrajectoryFeedback
from verl.utils.reward_score import default_compute_score
from .llm_evaluator import LLMEvaluator

# 设置debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置文件日志记录器
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)

# 创建文件处理器
import os
from datetime import datetime

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 创建带时间戳的日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/reward_manager_{timestamp}.log'

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 设置文件日志格式
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)

# 添加文件处理器到logger
file_logger.addHandler(file_handler)
file_logger.propagate = False  # 避免重复输出到控制台

logger.info(f"Reward manager logging to file: {log_filename}")

class SimpleDenseFeedbackRewardManager:
    """Simplified reward manager focused on grounding, information sufficiency, and refinement."""
    
    def __init__(self, tokenizer, num_examine=2, compute_score=None, reward_fn_key="data_source", 
                 enable_llm_evaluation=True, llm_model="gpt-4.1-mini"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # 强制启用LLM评估
        self.enable_llm_evaluation = True
        self.llm_model = llm_model
        
        # 初始化LLM评估器
        try:
            self.llm_evaluator = LLMEvaluator(model=llm_model)
            logger.info(f"LLM Evaluator initialized with model: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Evaluator: {e}")
            raise RuntimeError(f"LLM Evaluator is required but failed to initialize: {e}")
        
        # 简化的配置
        self.config = self._get_simplified_config()
        
        # 设置日志级别
        self._setup_logging()
        
        # 定义patterns用于解析轨迹组件
        self.patterns = {
            "search": r"<search>(.*?)</search>",
            "information": r"<information>(.*?)</information>",
            "answer": r"<answer>(.*?)</answer>",
            "think": r"<think>(.*?)</think>",
        }
    
        logger.info(f"SimpleDenseFeedbackRewardManager initialized with LLM evaluation enabled")
        logger.info(f"Using patterns for: {list(self.patterns.keys())}")
        logger.info(f"LLM evaluation enabled: {self.enable_llm_evaluation}")
    
    def _get_simplified_config(self):
        """Get simplified configuration"""
        return {
            "max_tool_steps": 5,
            "insufficient_info_penalty": -0.3,        # Penalty for answering directly with insufficient information
            "refinement_bonus": 0.5,                  # Reward for improving from insufficient to sufficient
            "grounding_bonus": 0.3,                   # Reward for correct grounding
            "ungrounded_penalty": 0,                  # Penalty for ungrounded responses
            "format_bonus": 0.3,                      # Format reward: bonus when sequence ends with answer
            # Prevent reward hacking: limits and decay
            "grounding_bonus_max_steps": 2,           # Maximum number of steps to apply grounding bonus (e.g., only reward first 2 times)
            "grounding_bonus_decay": 0.6,             # Decay coefficient for grounding bonus, decreasing by step
            "max_grounding_bonus_total": 1.0,         # Total upper limit for grounding bonus in a trajectory
            "min_component_length": 5,
            "max_log_length": 200,
            "enable_debug_logs": True,
            "reward_allocation_strategy": "component_based",
            "min_reward_value": -1.0,
            "max_reward_value": 2.0,
            "smooth_reward_transition": True,
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
    
    def _run_async(self, coro):
        """Run coroutine in a safe manner in a synchronous environment.
        - If there is no event loop running, use asyncio.run.
        - If there is an event loop running (e.g. in Ray/uvloop environment), create a new event loop in a new thread and run it.
        """
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        
        if not loop_running:
            return asyncio.run(coro)
        
        result_holder = {}
        error_holder = {}
        
        def _thread_runner():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_holder["result"] = new_loop.run_until_complete(coro)
            except Exception as e:
                error_holder["error"] = e
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass
        
        t = threading.Thread(target=_thread_runner, daemon=True)
        t.start()
        t.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")
    
    def parse_trajectory_components(self, response_str: str, response_tokens: List[int] = None) -> List[TrajectoryComponent]:
        """Parse trajectory components with accurate token positioning and fallback parsing."""
        logger.debug(f"Parsing response with {len(response_tokens) if response_tokens else 'unknown'} tokens")
        
        # Log to file
        file_logger.info(f"=== PARSING TRAJECTORY COMPONENTS ===")
        file_logger.info(f"Response string: {response_str}")
        file_logger.info(f"Response tokens count: {len(response_tokens) if response_tokens else 'unknown'}")
        
        # If response_tokens is not provided, tokenize the response
        if response_tokens is None:
            response_tokens = self.tokenizer.encode(response_str, add_special_tokens=False)
            logger.debug(f"Tokenized response with {len(response_tokens)} tokens")
        
        components = []
        step_number = 1
        
        # First parse all known label components
        known_components = []
        for component_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, response_str, re.DOTALL))
            for i, match in enumerate(matches):
                content = match.group(1).strip()
                if len(content) < self.config.get("min_component_length", 5):
                    logger.debug(f"Skipping {component_type} component {i+1} (too short: {len(content)})")
                    continue
                
                # Calculate accurate token positions
                start_char = match.start()
                end_char = match.end()
                
                # Convert character positions to token positions
                start_token_idx = self._char_to_token_position(response_str, response_tokens, start_char)
                end_token_idx = self._char_to_token_position(response_str, response_tokens, end_char)
                
                if start_token_idx is None or end_token_idx is None:
                    logger.warning(f"Could not determine token positions for {component_type} component")
                    continue
                
                component = TrajectoryComponent(
                    component_type=component_type,
                    content=content,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    step_number=step_number
                )
                known_components.append(component)
                step_number += 1
        
                logger.debug(f"Found {component_type} component at tokens {start_token_idx}-{end_token_idx}: {content[:50]}...")
                
                # Log to file
                file_logger.info(f"Component {step_number-1}: {component_type}")
                file_logger.info(f"  Content: {content}")
                file_logger.info(f"  Token range: {start_token_idx}-{end_token_idx}")
                file_logger.info(f"  Step number: {step_number-1}")
        
        # Sort known components by token positions
        known_components.sort(key=lambda x: x.start_token_idx)
        
        # Process remaining content as think components
        think_components = self._extract_think_components(response_str, response_tokens, known_components, step_number)
        
        # Merge all components
        components = known_components + think_components
        
        # If no components are found, log a warning
        if not components:
            logger.warning("No structured components found in response")
        
        # Sort components by token positions
        components.sort(key=lambda x: x.start_token_idx)
        logger.info(f"Parsed {len(components)} trajectory components")
        
        # Log parsed results to file
        file_logger.info(f"Total components parsed: {len(components)}")
        file_logger.info(f"Component types: {[c.component_type for c in components]}")
        file_logger.info("=" * 50)
        
        return components
    
    def _extract_think_components(self, response_str: str, response_tokens: List[int], 
                                 known_components: List[TrajectoryComponent], 
                                 start_step_number: int) -> List[TrajectoryComponent]:
        """Extract remaining content as think components"""
        think_components = []
        step_number = start_step_number
        
        # Get all known components' character position ranges
        covered_ranges = []
        for comp in known_components:
            start_char = self._token_to_char_position(response_str, response_tokens, comp.start_token_idx)
            end_char = self._token_to_char_position(response_str, response_tokens, comp.end_token_idx)
            if start_char is not None and end_char is not None:
                covered_ranges.append((start_char, end_char))
        
        # Merge overlapping ranges
        covered_ranges = self._merge_overlapping_ranges(covered_ranges)
        
        # Find un-covered text segments
        last_end = 0
        for start, end in covered_ranges:
            if last_end < start:
                # Extract think content
                think_content = response_str[last_end:start].strip()
                if len(think_content) >= self.config.get("min_component_length", 5):
                    # Calculate token positions
                    start_token_idx = self._char_to_token_position(response_str, response_tokens, last_end)
                    end_token_idx = self._char_to_token_position(response_str, response_tokens, start)
                    
                    if start_token_idx is not None and end_token_idx is not None:
                        component = TrajectoryComponent(
                            component_type="think",
                            content=think_content,
                            start_token_idx=start_token_idx,
                            end_token_idx=end_token_idx,
                            step_number=step_number
                        )
                        think_components.append(component)
                        step_number += 1
                        
                        logger.debug(f"Found think component at tokens {start_token_idx}-{end_token_idx}: {think_content[:50]}...")
                        
                        # Log to file
                        file_logger.info(f"Component {step_number-1}: think")
                        file_logger.info(f"  Content: {think_content}")
                        file_logger.info(f"  Token range: {start_token_idx}-{end_token_idx}")
                        file_logger.info(f"  Step number: {step_number-1}")
            
            last_end = max(last_end, end)
        
        # Process last segment
        if last_end < len(response_str):
            think_content = response_str[last_end:].strip()
            if len(think_content) >= self.config.get("min_component_length", 5):
                start_token_idx = self._char_to_token_position(response_str, response_tokens, last_end)
                end_token_idx = self._char_to_token_position(response_str, response_tokens, len(response_str))
                
                if start_token_idx is not None and end_token_idx is not None:
                    component = TrajectoryComponent(
                        component_type="think",
                        content=think_content,
                        start_token_idx=start_token_idx,
                        end_token_idx=end_token_idx,
                        step_number=step_number
                    )
                    think_components.append(component)
                    
                    logger.debug(f"Found think component at tokens {start_token_idx}-{end_token_idx}: {think_content[:50]}...")
                    
                    # Log to file
                    file_logger.info(f"Component {step_number}: think")
                    file_logger.info(f"  Content: {think_content}")
                    file_logger.info(f"  Token range: {start_token_idx}-{end_token_idx}")
                    file_logger.info(f"  Step number: {step_number}")
        
        return think_components
    
    def _token_to_char_position(self, text: str, tokens: List[int], token_pos: int) -> Optional[int]:
        """Convert token positions to character positions"""
        try:
            if token_pos >= len(tokens):
                return len(text)
            
            # Decode to specified token positions
            partial_tokens = tokens[:token_pos]
            partial_text = self.tokenizer.decode(partial_tokens, skip_special_tokens=True)
            return len(partial_text)
        except Exception as e:
            logger.warning(f"Error converting token position {token_pos} to char position: {e}")
            return None

    def _char_to_token_position(self, text: str, tokens: List[int], char_pos: int) -> Optional[int]:
        """Convert character positions to token positions"""
        try:
            if char_pos <= 0:
                return 0
            if char_pos >= len(text):
                return len(tokens)
            
            # Binary search to find corresponding token positions
            left, right = 0, len(tokens)
            while left < right:
                mid = (left + right) // 2
                partial_tokens = tokens[:mid]
                partial_text = self.tokenizer.decode(partial_tokens, skip_special_tokens=True)
                partial_length = len(partial_text)
                
                if partial_length < char_pos:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        except Exception as e:
            logger.warning(f'Error converting char position {char_pos} to token position: {e}')
            return None
    
    def _merge_overlapping_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping ranges"""
        if not ranges:
            return []
        
        # Sort by starting position
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        
        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # Overlapping or adjacent
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged
    
  
    def analyze_trajectory_sync(self, components: List[TrajectoryComponent], 
                               ground_truth, question: str = None) -> TrajectoryFeedback:
        """Analyze entire trajectory and calculate feedback (Synchronous version, using fallback evaluation)"""
        logger.info(f"Analyzing trajectory with {len(components)} components (sync version)")
        
        # Log to file
        file_logger.info(f"=== TRAJECTORY ANALYSIS (SYNC) ===")
        file_logger.info(f"Ground truth: {ground_truth}")
        file_logger.info(f"Question: {question}")
        file_logger.info(f"Components count: {len(components)}")
        
        # Extract key information
        search_components = [c for c in components if c.component_type == "search"]
        information_components = [c for c in components if c.component_type == "information"]
        answer_components = [c for c in components if c.component_type == "answer"]
        
        logger.debug(f"Component breakdown: search={len(search_components)}, "
                    f"information={len(information_components)}, answer={len(answer_components)}")
        
        # Log component distribution to file
        file_logger.info(f"Component breakdown:")
        file_logger.info(f"  Search components: {len(search_components)}")
        file_logger.info(f"  Information components: {len(information_components)}")
        file_logger.info(f"  Answer components: {len(answer_components)}")
        
        # Analyze temporal dependencies and information flow (Synchronous fallback evaluation)
        temporal_analysis = self._analyze_temporal_dependencies_sync(components, ground_truth, question)
        
        # Compute component scores with improved logic
        answer_quality_score = self._score_answer_quality_simplified(answer_components, ground_truth)
        
        logger.info(f"Component scores - Answer: {answer_quality_score:.3f}")
        
        # Log评分结果到文件
        file_logger.info(f"Component scores:")
        file_logger.info(f"  Answer quality: {answer_quality_score:.3f}")
        
        # Check for penalties with temporal awareness
        has_insufficient_info = temporal_analysis["has_insufficient_info"]
        has_repeated_searches = self._has_repeated_searches(search_components)
        exceeds_max_steps = len(search_components) > self.config["max_tool_steps"]
        
        logger.info(f"Temporal analysis - Insufficient info: {has_insufficient_info}, "
                   f"Repeated searches: {has_repeated_searches}, Exceeds max steps: {exceeds_max_steps}")
        
        # Log penalty information to file
        file_logger.info(f"Penalty analysis:")
        file_logger.info(f"  Has insufficient info: {has_insufficient_info}")
        file_logger.info(f"  Has repeated searches: {has_repeated_searches}")
        file_logger.info(f"  Exceeds max steps: {exceeds_max_steps}")
        
        # Create trajectory feedback
        feedback = TrajectoryFeedback(
            trajectory_id="trajectory_1",
            components=components,
            final_answer=answer_components[-1].content if answer_components else "",
            ground_truth=ground_truth,
            think_score=0.0,
            answer_quality_score=answer_quality_score,
            has_insufficient_info=has_insufficient_info,
            has_repeated_tools=has_repeated_searches,
            exceeds_max_steps=exceeds_max_steps
        )
        
        # Store temporal analysis results as class attribute
        feedback._temporal_analysis = temporal_analysis
        
        # Log temporal analysis results to file
        file_logger.info(f"Temporal analysis results:")
        for key, value in temporal_analysis.items():
            if key != "llm_evaluation_results":  # Avoid logging too long LLM results
                file_logger.info(f"  {key}: {value}")
        
        # Log fallback evaluation results summary
        if "llm_evaluation_results" in temporal_analysis:
            file_logger.info(f"  Fallback evaluation results count: {len(temporal_analysis['llm_evaluation_results'])}")
            for i, result in enumerate(temporal_analysis["llm_evaluation_results"]):
                file_logger.info(f"    Component {i+1}: {result.get('information_quality', 'Unknown')}")
        
        file_logger.info("=" * 50)
        
        return feedback
    
    def create_dense_reward_tensor(self, feedback: TrajectoryFeedback, 
                                  response_length: int) -> torch.Tensor:
        """Create dense reward tensor with intelligent allocation strategy."""
        try:
            logger.info(f"Creating dense reward tensor for response length: {response_length}")
            
            # Log to file
            file_logger.info(f"=== CREATING DENSE REWARD TENSOR ===")
            file_logger.info(f"Response length: {response_length}")
            file_logger.info(f"Components count: {len(feedback.components)}")
            file_logger.info(f"Reward allocation strategy: {self.config['reward_allocation_strategy']}")
            
            # Parameter validation
            if response_length <= 0:
                logger.warning(f"Invalid response_length: {response_length}, using default 500")
                response_length = 500
            
            # Create reward tensor
            reward_tensor = torch.zeros(response_length, dtype=torch.float32)
            
            if not feedback.components:
                file_logger.info("No components found, applying uniform score: 0.0")
                file_logger.info("=" * 50)
                return reward_tensor
            
            # Use improved reward allocation strategy
            if self.config["reward_allocation_strategy"] == "component_based":
                reward_tensor = self._allocate_rewards_component_based(feedback, response_length)
                file_logger.info("Used component-based reward allocation")
            else:
                reward_tensor = self._allocate_rewards_uniform(feedback, response_length)
                file_logger.info("Used uniform reward allocation")
            
            # Apply smooth transition
            if self.config["smooth_reward_transition"]:
                reward_tensor = self._smooth_reward_transitions(reward_tensor)
                file_logger.info("Applied smooth reward transitions")
            
            # Limit reward range
            reward_tensor = torch.clamp(reward_tensor, 
                                        self.config["min_reward_value"], 
                                        self.config["max_reward_value"])
            
            # Log reward tensor statistics
            self._log_reward_statistics(reward_tensor)
            
            # Log reward tensor details to file
            self._log_reward_tensor_details(reward_tensor, feedback)
            
            return reward_tensor
            
        except Exception as e:
            logger.error(f"Error creating dense reward tensor: {e}")
            # Return default reward tensor
            default_reward = torch.zeros(response_length, dtype=torch.float32)
            default_reward[:] = 0.5
            return default_reward
    
    def _apply_core_reward_adjustments(self, base_score: float, feedback: TrajectoryFeedback, 
                                     component: TrajectoryComponent, sorted_components: List[TrajectoryComponent], 
                                     component_idx: int, temporal_meta: Dict) -> float:
        """Apply reward adjustments for four core aspects (grounding only for search/think, with decay and upper limit)"""
        score = base_score
        
        # 1. Information insufficient penalty
        if temporal_meta.get("has_insufficient_info", False):
            if component.component_type == "answer":
                # Check if there is insufficient information before
                for i in range(component_idx - 1, -1, -1):
                    prev_component = sorted_components[i]
                    if prev_component.component_type == "information":
                        # Apply information insufficient penalty
                        score += self.config["insufficient_info_penalty"]
                        logger.debug(f"Penalizing answer after insufficient info: {score:.3f}")
                        break
                    elif prev_component.component_type == "search":
                        # If previous component is search, means this answer is not directly affected by insufficient information
                        break
        
        # 2. Reasoning grounding evaluation (only for search/think, with decay and total upper limit, to avoid reward hacking)
        if component.component_type in ["search", "think"]:
            reasoning_grounded = temporal_meta.get("reasoning_grounded", True)
            if reasoning_grounded:
                applied_steps = temporal_meta.get("grounding_applied", 0)
                total_bonus = temporal_meta.get("grounding_bonus_total", 0.0)
                max_steps = self.config.get("grounding_bonus_max_steps", 2)
                decay = self.config.get("grounding_bonus_decay", 0.85)
                max_total = self.config.get("max_grounding_bonus_total", 1.0)
                
                if applied_steps < max_steps and total_bonus < max_total:
                    step_decay = decay ** applied_steps
                    raw_bonus = self.config.get("grounding_bonus", 0.4) * step_decay
                    # Ensure total bonus does not exceed upper limit
                    allowed_bonus = min(raw_bonus, max_total - total_bonus)
                    score += allowed_bonus
                    temporal_meta["grounding_applied"] = applied_steps + 1
                    temporal_meta["grounding_bonus_total"] = total_bonus + allowed_bonus
                    logger.debug(f"Rewarding grounded reasoning (step {applied_steps+1}, bonus {allowed_bonus:.3f}): {score:.3f}")
            else:
                score += self.config.get("ungrounded_penalty", -0.3)
                logger.debug(f"Penalizing ungrounded reasoning: {score:.3f}")
        
        # 3. Information improvement reward (from insufficient to sufficient)
        if component.component_type == "information":
            refinement_success = temporal_meta.get("refinement_success", False)
            if refinement_success:
                refinement_steps = temporal_meta.get("refinement_steps", 0)
                # The less steps, the more reward
                refinement_bonus = self.config["refinement_bonus"] * (1.0 - refinement_steps * 0.1)
                score += refinement_bonus
                logger.debug(f"Rewarding information refinement: {score:.3f}")
        
        # 4. Format reward: if sequence ends with answer, give format reward
        if component.component_type == "answer":
            temporal_sequence = temporal_meta.get("temporal_sequence", [])
            if temporal_sequence and temporal_sequence[-1] == "answer":
                # Check if this is the last answer component
                is_last_answer = True
                for j in range(component_idx + 1, len(sorted_components)):
                    if sorted_components[j].component_type == "answer":
                        is_last_answer = False
                        break
                
                if is_last_answer:
                    format_bonus = self.config.get("format_bonus", 0.3)
                    score += format_bonus
                    logger.debug(f"Rewarding proper format (sequence ends with answer): {score:.3f}")
                    file_logger.info(f"Applied format bonus {format_bonus:.3f} for sequence ending with answer")
        
        # 5. Final answer quality (already considered in base_score)
        
        # Ensure score is in a reasonable range
        return max(self.config["min_reward_value"], score)
    
    def _allocate_rewards_component_based(self, feedback: TrajectoryFeedback, response_length: int) -> torch.Tensor:
        """Allocate rewards based on four core aspects: 1. grounding, 2. insufficient info penalty, 3. refinement, 4. final answer"""
        reward_tensor = torch.zeros(response_length, dtype=torch.float32)
        
        # Get temporal analysis results
        temporal_meta = getattr(feedback, '_temporal_analysis', {})
        
        # Initialize grounding reward tracking
        if "grounding_applied" not in temporal_meta:
            temporal_meta["grounding_applied"] = 0
        if "grounding_bonus_total" not in temporal_meta:
            temporal_meta["grounding_bonus_total"] = 0.0
        
        # Sort
        sorted_components = sorted(feedback.components, key=lambda x: x.start_token_idx)
        
        for i, component in enumerate(sorted_components):
            if component.start_token_idx >= response_length or component.end_token_idx > response_length:
                continue
            
            # Base score: only based on final answer quality
            base_score = feedback.answer_quality_score if component.component_type == "answer" else 0.0
            
            # Apply reward adjustments for four core aspects
            final_score = self._apply_core_reward_adjustments(
                base_score, feedback, component, sorted_components, i, temporal_meta
            )
            
            # Allocate reward
            start_idx = min(component.start_token_idx, response_length - 1)
            end_idx = min(component.end_token_idx, response_length)
            reward_tensor[start_idx:end_idx] = final_score
            
            # Log to file
            file_logger.info(f"Component {i+1} ({component.component_type}): base_score={base_score:.3f}, final_score={final_score:.3f}, tokens={start_idx}-{end_idx}")
        
        # Log grounding reward statistics
        grounding_applied = temporal_meta.get("grounding_applied", 0)
        grounding_total = temporal_meta.get("grounding_bonus_total", 0.0)
        file_logger.info(f"Grounding rewards applied: {grounding_applied} steps, total bonus: {grounding_total:.3f}")
        
        return reward_tensor
    
    

    def _smooth_reward_transitions(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        """Smooth reward transitions, to avoid sudden changes"""
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
    
    def _log_reward_statistics(self, reward_tensor: torch.Tensor):
        """Log reward tensor statistics"""
        non_zero_rewards = reward_tensor[reward_tensor != 0]
        if len(non_zero_rewards) > 0:
            logger.info(f"Reward tensor created - Shape: {reward_tensor.shape}, "
                       f"Non-zero rewards: {len(non_zero_rewards)}, "
                       f"Min: {non_zero_rewards.min().item():.3f}, "
                       f"Max: {non_zero_rewards.max().item():.3f}, "
                       f"Mean: {non_zero_rewards.mean().item():.3f}, "
                       f"Std: {non_zero_rewards.std().item():.3f}")
        else:
            logger.warning("No non-zero rewards in tensor")
    
    def _log_reward_tensor_details(self, reward_tensor: torch.Tensor, feedback: TrajectoryFeedback):
        """Log reward tensor details to file"""
        file_logger.info(f"Reward tensor details:")
        file_logger.info(f"  Shape: {reward_tensor.shape}")
        file_logger.info(f"  Min value: {reward_tensor.min().item():.3f}")
        file_logger.info(f"  Max value: {reward_tensor.max().item():.3f}")
        file_logger.info(f"  Mean value: {reward_tensor.mean().item():.3f}")
        
        # Count non-zero rewards and their distribution
        non_zero_rewards = reward_tensor[reward_tensor != 0]
        file_logger.info(f"  Non-zero rewards: {len(non_zero_rewards)}")
        if len(non_zero_rewards) > 0:
            file_logger.info(f"  Non-zero min: {non_zero_rewards.min().item():.3f}")
            file_logger.info(f"  Non-zero max: {non_zero_rewards.max().item():.3f}")
            file_logger.info(f"  Non-zero mean: {non_zero_rewards.mean().item():.3f}")
            file_logger.info(f"  Non-zero std: {non_zero_rewards.std().item():.3f}")
        
        # Log reward allocation for each component
        if feedback.components:
            file_logger.info(f"  Component reward allocation:")
            for i, component in enumerate(feedback.components):
                start_idx = component.start_token_idx
                end_idx = min(component.end_token_idx, len(reward_tensor))
                if start_idx < len(reward_tensor):
                    component_rewards = reward_tensor[start_idx:end_idx]
                    if len(component_rewards) > 0:
                        file_logger.info(f"    {component.component_type} (step {component.step_number}): "
                                       f"tokens {start_idx}-{end_idx}, "
                                       f"reward range [{component_rewards.min().item():.3f}, {component_rewards.max().item():.3f}], "
                                       f"mean {component_rewards.mean().item():.3f}")
        
        file_logger.info("=" * 50)
    
    
    def _score_answer_quality_simplified(self, answer_components: List[TrajectoryComponent], 
                                       ground_truth) -> float:
        """Simplified answer quality score: only consider exact matches"""
        if not answer_components:
            return 0.0
        
        final_answer = answer_components[-1].content.strip()
        logger.debug(f"Scoring final answer: {final_answer[:100]}...")
        
        # Process ground truth format
        if isinstance(ground_truth, str):
            # Process separator format
            if "<|answer_split|>" in ground_truth:
                gt_parts = [gt.strip() for gt in ground_truth.split("<|answer_split|>")]
            else:
                gt_parts = [ground_truth.strip()]
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
            gt_parts = [str(gt).strip() for gt in gt_values]
        elif isinstance(ground_truth, (list, np.ndarray, set, tuple)):
            gt_parts = [str(gt).strip() for gt in ground_truth]
        else:
            gt_parts = [str(ground_truth).strip()]
        
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
    
    def _check_repeated_searches_improved(self, search_components: List[TrajectoryComponent]) -> List[Tuple[int, int, float]]:
        """Improved repeated search detection, return detailed information for completely identical queries"""
        if len(search_components) < 2:
            return []
        
        search_contents = [comp.content.lower().strip() for comp in search_components]
        repeated_pairs = []
        
        for i in range(len(search_contents)):
            for j in range(i + 1, len(search_contents)):
                if search_contents[i] == search_contents[j]:
                    # Completely identical queries, similarity is 1.0
                    repeated_pairs.append((i, j, 1.0))
                    logger.debug(f"Found identical queries: '{search_contents[i]}' and '{search_contents[j]}'")
        
        return repeated_pairs
    
    def _has_repeated_searches(self, search_components: List[TrajectoryComponent]) -> bool:
        """Check if there are repeated searches"""
        return len(self._check_repeated_searches_improved(search_components)) > 0
    

    

    def _evaluate_reasoning_grounding(self, reasoning_text: str, search_evidence: List[Dict], 
                                      question: str) -> Dict[str, Any]:
        """Synchronous wrapper: call LLM evaluator (internal safe running coroutine)."""
        if not self.llm_evaluator:
            raise RuntimeError("LLM evaluator not available")
        try:
            return self._run_async(self.llm_evaluator.evaluate_reasoning_grounding(
                reasoning_text, search_evidence, question
            ))
        except Exception as e:
            logger.error(f"Error in LLM grounding evaluation: {e}")
            return {
                "premise_grounding": "Unspecified",
                "anchor_type": "NONE",
                "evidence_citations": [],
                "unmatched_premises": [],
                "premise_justification": f"Evaluation error: {str(e)}",
                "evaluation_success": False
            }

    def _evaluate_information_quality_batch(self, information_components: List[TrajectoryComponent], 
                                           search_components: List[TrajectoryComponent], 
                                           ground_truth: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper: pass batch requests to LLM evaluator (internal safe running coroutine)"""
        if not self.llm_evaluator:
            raise RuntimeError("LLM evaluator not available")
        evaluation_requests = []
        for i, info_comp in enumerate(information_components):
            search_query = self._find_corresponding_search_query(info_comp, search_components)
            documents = [{"content": info_comp.content}]
            evaluation_requests.append({
                "type": "information_quality",
                "query": search_query,
                "documents": documents,
                "component_index": i
            })
        try:
            return self._run_async(self.llm_evaluator.batch_evaluate(evaluation_requests))
        except Exception as e:
            logger.error(f"Error in batch information quality evaluation: {e}")
            # Maintain original behavior: throw exception to upper layer
            raise RuntimeError(f"Batch information quality evaluation failed: {e}")
    
    def _find_corresponding_search_query(self, info_component: TrajectoryComponent, 
                                       search_components: List[TrajectoryComponent]) -> str:
        """Find corresponding search query"""
        # Simple heuristic: find the nearest search component
        for search_comp in reversed(search_components):
            if search_comp.start_token_idx < info_component.start_token_idx:
                return search_comp.content
        return "unknown query"
    

    

    
    
    
    def _is_information_sufficient_llm(self, evaluation_result: Dict[str, Any]) -> bool:
        """Based on LLM evaluation results, determine if information is sufficient (Synchronous version, not calling LLM)"""
        if not evaluation_result.get("evaluation_success", False):
            return False
        quality = evaluation_result.get("information_quality", "Unspecified")
        return quality == "Sufficient"
    
 
    def _analyze_temporal_dependencies_sync(self, components: List[TrajectoryComponent], 
                                           ground_truth, question: str = None) -> Dict:
        """Analyze temporal dependencies, focusing on four core aspects (using LLM evaluation)"""
        logger.info("Analyzing temporal dependencies with LLM evaluation")
        
        # Sort components by time order
        sorted_components = sorted(components, key=lambda x: x.start_token_idx)
        
        # Extract different types of components
        search_components = [c for c in sorted_components if c.component_type == "search"]
        information_components = [c for c in sorted_components if c.component_type == "information"]
        
        # 1. Information quality analysis (using LLM evaluation)
        info_quality_analysis = self._analyze_information_quality_flow_llm(
            sorted_components, ground_truth, search_components, information_components
        )
        
        # 2. Reasoning grounding analysis (using LLM evaluation)
        grounding_analysis = self._analyze_reasoning_grounding(sorted_components, question, search_components)
        
        # 3. Improved analysis (from insufficient to sufficient)
        refinement_analysis = self._analyze_information_refinement_llm(
            sorted_components, ground_truth, info_quality_analysis["llm_evaluation_results"]
        )
        
        temporal_analysis = {
            "has_insufficient_info": info_quality_analysis["has_insufficient_info"],
            "reasoning_grounded": grounding_analysis["reasoning_grounded"],
            "refinement_success": refinement_analysis["refinement_success"],
            "refinement_steps": refinement_analysis["refinement_steps"],
            "temporal_sequence": [comp.component_type for comp in sorted_components],
            "llm_evaluation_results": info_quality_analysis["llm_evaluation_results"]
        }
        
        logger.info(f"LLM temporal analysis results: {temporal_analysis}")
        return temporal_analysis
    
    def _analyze_information_quality_flow_llm(self, sorted_components: List[TrajectoryComponent], 
                                             ground_truth, search_components: List[TrajectoryComponent], 
                                             information_components: List[TrajectoryComponent]) -> Dict:
        """Use LLM to batch evaluate information quality flow (Synchronous, internal using asyncio.run)"""
        has_insufficient_info = False
        llm_evaluation_results = []
        if information_components:
            llm_evaluation_results = self._evaluate_information_quality_batch(
                information_components, search_components, ground_truth
            )
            for i, (component, eval_result) in enumerate(zip(information_components, llm_evaluation_results)):
                is_sufficient = self._is_information_sufficient_llm(eval_result)
                if not is_sufficient:
                    has_insufficient_info = True
                    logger.debug(f"LLM evaluation: Low quality information at step {i+1}: {eval_result.get('information_quality', 'Unknown')}")
                    file_logger.info(f"LLM evaluation result for {component.component_type} component {i+1}:")
                    file_logger.info(f"  Content: {component.content[:100]}...")
                    file_logger.info(f"  Quality: {eval_result.get('information_quality', 'Unknown')}")
        return {
            "has_insufficient_info": has_insufficient_info,
            "llm_evaluation_results": llm_evaluation_results
        }

    def _analyze_reasoning_grounding(self, sorted_components: List[TrajectoryComponent], 
                                    question: str, search_components: List[TrajectoryComponent]) -> Dict:
        """Analyze reasoning grounding (Synchronous wrapper, internal using asyncio.run to call LLM)"""
        if not self.llm_evaluator:
            raise RuntimeError("LLM evaluator not available")
        search_evidence = []
        for search_comp in search_components:
            search_evidence.append({
                "content": search_comp.content,
                "type": "search_query",
                "step": search_comp.step_number
            })
        grounding_results = []
        for search_comp in search_components:
            if question:
                result = self._evaluate_reasoning_grounding(search_comp.content, search_evidence, question)
            else:
                result = {"premise_grounding": "Directly Grounded", "evaluation_success": True, "fallback": True}
            grounding_results.append(result)
        grounded_count = sum(1 for r in grounding_results if r.get("premise_grounding") == "Directly Grounded")
        total_count = len(grounding_results)
        reasoning_grounded = grounded_count > 0 and (grounded_count / max(1, total_count)) >= 0.5
        return {
            "reasoning_grounded": reasoning_grounded,
            "grounding_details": f"LLM evaluation: {grounded_count}/{total_count} grounded",
            "grounding_results": grounding_results
        }


    
    def _analyze_information_refinement_llm(self, sorted_components: List[TrajectoryComponent], 
                                          ground_truth, llm_evaluation_results: List[Dict[str, Any]]) -> Dict:
        """Based on LLM evaluation results, analyze information improvement"""
        refinement_success = False
        refinement_steps = 0
        
        if not llm_evaluation_results:
            # If there is no LLM evaluation results, return default values
            return {
                "refinement_success": False,
                "refinement_steps": 0
            }
        
        # Find the first information that is insufficient
        first_insufficient_idx = None
        for i, eval_result in enumerate(llm_evaluation_results):
            if eval_result.get("information_quality") == "Insufficient":
                first_insufficient_idx = i
                break
        
        if first_insufficient_idx is not None:
            # Check if there is improvement after
            for i in range(first_insufficient_idx + 1, len(llm_evaluation_results)):
                eval_result = llm_evaluation_results[i]
                if eval_result.get("information_quality") == "Sufficient":
                    refinement_success = True
                    refinement_steps = i - first_insufficient_idx
                    logger.debug(f"LLM evaluation: Information refined after {refinement_steps} steps")
                    break
        
        return {
            "refinement_success": refinement_success,
            "refinement_steps": refinement_steps
        }
    

    
    def __call__(self, data: DataProto, return_dict=False):
        """Process batch data and return dense reward tensors (Synchronous version)"""
        logger.info(f"Processing batch with {len(data)} items")
        
        # Log to file
        file_logger.info(f"=== BATCH PROCESSING START ===")
        file_logger.info(f"Batch size: {len(data)}")
        file_logger.info(f"Return dict: {return_dict}")
        
        # Safe check
        if not data or len(data) == 0:
            logger.warning("Empty data batch received")
            file_logger.warning("Empty data batch received")
            if return_dict:
                return {"reward_tensor": torch.tensor([])}
            else:
                return torch.tensor([])
        
        dense_reward_tensors = []
        try:
            for i in range(len(data)):
                try:
                    data_item = data[i]
                    file_logger.info(f"--- Processing item {i+1}/{len(data)} ---")
                    
                    # Safe check data item
                    if not hasattr(data_item, 'batch') or not hasattr(data_item, 'non_tensor_batch'):
                        logger.warning(f"Data item {i} missing required attributes, skipping")
                        file_logger.warning(f"Data item {i} missing required attributes, skipping")
                        continue
                    
                    # Check necessary fields
                    required_fields = ["prompts", "responses", "attention_mask"]
                    for field in required_fields:
                        if field not in data_item.batch:
                            logger.warning(f"Data item {i} missing required field: {field}, skipping")
                            file_logger.warning(f"Data item {i} missing required field: {field}, skipping")
                            continue
                    
                    if "reward_model" not in data_item.non_tensor_batch:
                        logger.warning(f"Data item {i} missing reward_model field, skipping")
                        file_logger.warning(f"Data item {i} missing reward_model field, skipping")
                        continue
                    
                    # Process single data item (Synchronous call)
                    result = self._process_single_item_sync(data_item, i)
                    if result:
                        dense_reward_tensors.append(result["reward_tensor"])
                        file_logger.info(f"Item {i+1} processed successfully")
                    else:
                        file_logger.warning(f"Item {i+1} processing returned None")
                    
                except Exception as e:
                    logger.error(f"Error processing data item {i}: {e}")
                    file_logger.error(f"Error processing data item {i}: {e}")
                    # Create default reward tensor as fallback
                    default_reward = torch.zeros(500, dtype=torch.float32)
                    dense_reward_tensors.append(default_reward)
                    continue
            
            # Check if there are valid reward tensors
            if not dense_reward_tensors:
                logger.warning("No valid reward tensors generated")
                if return_dict:
                    return {"reward_tensor": torch.tensor([])}
                else:
                    return torch.tensor([])
            
            # Ensure all reward tensor lengths are consistent
            target_length = 500
            normalized_rewards = []
            for reward_tensor in dense_reward_tensors:
                if len(reward_tensor) != target_length:
                    # Adjust length
                    if len(reward_tensor) > target_length:
                        normalized_reward = reward_tensor[:target_length]
                    else:
                        normalized_reward = torch.zeros(target_length, dtype=torch.float32)
                        normalized_reward[:len(reward_tensor)] = reward_tensor
                    normalized_rewards.append(normalized_reward)
                else:
                    normalized_rewards.append(reward_tensor)
            
            # Direct stack, to avoid padding problems
            stacked_rewards = torch.stack(normalized_rewards, dim=0)
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
            # Return default values
            default_reward = torch.zeros((len(data), 500), dtype=torch.float32)
            if return_dict:
                return {"reward_tensor": default_reward}
            else:
                return default_reward
    
    def _process_single_item_sync(self, data_item, item_index):
        """Process single data item, return feedback and reward tensor (Synchronous version)"""
        try:
            # Log to file
            file_logger.info(f"=== PROCESSING SINGLE ITEM {item_index+1} ===")
            
            # Get response data
            response_ids = data_item.batch["responses"]
            response_ids_shape = response_ids.shape
            logger.debug(f"Response IDs shape: {response_ids_shape}")
            file_logger.info(f"Response IDs shape: {response_ids_shape}")
            
            # Calculate prompt length
            prompts_shape = data_item.batch["prompts"].shape
            logger.debug(f"Prompts shape: {prompts_shape}")
            file_logger.info(f"Prompts shape: {prompts_shape}")
            
            # Safe get prompt length
            if len(prompts_shape) >= 2:
                prompt_length = prompts_shape[1]
            elif len(prompts_shape) == 1:
                prompt_length = prompts_shape[0]
            else:
                prompt_length = 0
                logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
                file_logger.warning(f"Unexpected prompts shape: {prompts_shape}, using default prompt_length=0")
            
            logger.debug(f"Prompt length: {prompt_length}")
            file_logger.info(f"Prompt length: {prompt_length}")
            
            # Get attention mask
            attention_mask_shape = data_item.batch["attention_mask"].shape
            logger.debug(f"Attention mask shape: {attention_mask_shape}")
            file_logger.info(f"Attention mask shape: {attention_mask_shape}")
            
            # Safe remove batch dimensions
            if len(attention_mask_shape) >= 2:
                attention_mask = data_item.batch["attention_mask"][0]
            elif len(attention_mask_shape) == 1:
                attention_mask = data_item.batch["attention_mask"]
            else:
                logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                file_logger.warning(f"Unexpected attention_mask shape: {attention_mask_shape}, using default")
                attention_mask = torch.ones(1000, dtype=torch.long)
            
            # Ensure attention_mask is 1D
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            if attention_mask.dim() != 1:
                raise ValueError(f"Invalid attention_mask dimensions: {attention_mask.dim()}")
            
            logger.debug(f"Final attention_mask shape: {attention_mask.shape}")
            file_logger.info(f"Final attention_mask shape: {attention_mask.shape}")
            
            # Calculate valid response length
            if prompt_length > 0 and prompt_length < len(attention_mask):
                valid_response_length = attention_mask[prompt_length:].sum()
            else:
                valid_response_length = 100
            
            # Ensure length is a valid integer
            if hasattr(valid_response_length, 'item'):
                valid_response_length = valid_response_length.item()
                valid_response_length = int(valid_response_length)
            
            logger.info(f"Final valid response length: {valid_response_length}")
            file_logger.info(f"Final valid response length: {valid_response_length}")
            
            # Get valid response IDs
            if len(response_ids_shape) >= 2:
                if valid_response_length > 0 and valid_response_length < response_ids.shape[1]:
                    valid_response_ids = response_ids[0, :valid_response_length]
                else:
                    valid_response_ids = response_ids[0, :100]
            else:
                if valid_response_length > 0 and valid_response_length < len(response_ids):
                    valid_response_ids = response_ids[:valid_response_length]
                else:
                    valid_response_ids = response_ids[:100]
            
            # Ensure valid_response_ids is 1D tensor
            if valid_response_ids.dim() > 1:
                valid_response_ids = valid_response_ids.flatten()
            
            # Decode response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            
            # Try to get question (if exists)
            question = None
            if "question" in data_item.non_tensor_batch:
                question = data_item.non_tensor_batch["question"]
            elif "prompt" in data_item.non_tensor_batch:
                # Extract question content from prompt field
                prompt_data = data_item.non_tensor_batch["prompt"]
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
                raw_prompt = data_item.non_tensor_batch["raw_prompt"]
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

            
            logger.info(f"Data source: {data_source}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"Question: {question}")
            logger.info(f"Response: {response_str[:self.config.get('max_log_length', 200)]}...")
            
            # Log to file
            file_logger.info(f"Data source: {data_source}")
            file_logger.info(f"Ground truth: {ground_truth}")
            file_logger.info(f"Question: {question}")
            file_logger.info(f"Response: {response_str}")
            
            # Parse trajectory components (using improved token position calculation)
            components = self.parse_trajectory_components(response_str, valid_response_ids.tolist())
            
            # Analyze entire trajectory (Synchronous call)
            feedback = self.analyze_trajectory_sync(components, ground_truth, question)
            
            # Create dense reward tensor
            target_length = 500
            dense_reward = self.create_dense_reward_tensor(feedback, target_length)
            
            # Print analysis summary
            if item_index < self.num_examine:
                self._print_analysis_summary(data_source, item_index, response_str, ground_truth, components, feedback, dense_reward)
            
            file_logger.info(f"Item {item_index+1} processing completed successfully")
            
            return {
                "feedback": feedback,
                "reward_tensor": dense_reward
            }
            
        except Exception as e:
            logger.error(f"Error processing single item {item_index}: {e}")
            file_logger.error(f"Error processing single item {item_index}: {e}")
            return None
    
    def _print_analysis_summary(self, data_source, item_index, response_str, ground_truth, components, feedback, dense_reward):
        """Print analysis summary"""
        print(f"\n{'='*80}")
        print(f"[{data_source}] Trajectory Analysis for Item {item_index+1}:")
        print(f"{'='*80}")
        print(f"Response: {response_str[:300]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"Components Found: {len(components)}")
        for j, comp in enumerate(components):
            print(f"  {j+1}. {comp.component_type}: {comp.content[:100]}...")
        print(f"\nScores:")
        print(f"  Answer Quality: {feedback.answer_quality_score:.3f}")
        print(f"\nTemporal Analysis:")
        temporal_meta = getattr(feedback, '_temporal_analysis', {})
        print(f"  Recovery Steps: {temporal_meta.get('recovery_steps', 0)}")
        print(f"  Recovery Success: {temporal_meta.get('recovery_success', False)}")
        print(f"  Repetition Penalty: {temporal_meta.get('repetition_penalty', 0.0):.3f}")
        print(f"  Temporal Sequence: {' -> '.join(temporal_meta.get('temporal_sequence', []))}")
        print(f"\nPenalties:")
        print(f"  Insufficient Info: {feedback.has_insufficient_info}")
        print(f"  Repeated Searches: {feedback.has_repeated_tools}")
        print(f"  Exceeds Max Steps: {feedback.exceeds_max_steps}")
        print(f"\nReward Tensor Shape: {dense_reward.shape}")
        print(f"Non-zero rewards: {(dense_reward != 0).sum().item()}")
        print(f"Reward range: [{dense_reward.min().item():.3f}, {dense_reward.max().item():.3f}]")
        print(f"{'='*80}") 

    def _allocate_rewards_uniform(self, feedback: TrajectoryFeedback, response_length: int) -> torch.Tensor:
        """Uniformly allocate reward to the entire response (backup strategy)"""
        reward_tensor = torch.zeros(response_length, dtype=torch.float32)
        
        # Only use final answer quality as stable uniform signal, to avoid reward hacking caused by accumulation
        avg_score = float(feedback.answer_quality_score) if feedback.answer_quality_score is not None else 0.0
        reward_tensor[:] = avg_score
        logger.info(f"Applied uniform score {avg_score:.3f} to entire response")
        return reward_tensor 