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

"""
Per-Turn Reward Allocation: Distribute rewards at turn granularity.

This module provides utilities to allocate rewards to each turn based on:
1. Event ledger (ground truth events)
2. Step IDs (which tokens belong to which turn)
3. Component types (what actions were taken)

Key insight: Rewards should be allocated per-turn, not per-token, to enable
proper credit assignment with full context policy gradient.
"""

import torch
from typing import Dict, List, Optional
from verl.utils.event_ledger import EventLedger, EventType
import logging

logger = logging.getLogger(__name__)


class PerTurnRewardAllocator:
    """
    Allocate rewards at turn-level granularity for policy gradient training.
    
    Design principles:
    1. Rewards are computed based on event ledger (ground truth)
    2. Rewards are allocated to turns (not individual tokens)
    3. Within each turn, rewards can be distributed evenly or to last token
    """
    
    def __init__(self, distribution_mode: str = "even"):
        """
        Args:
            distribution_mode: How to distribute rewards within a turn
                - "even": Distribute evenly across all tokens in the turn
                - "last_token": Assign all reward to the last token of the turn
                - "first_and_last": Split reward between first and last token
        """
        self.distribution_mode = distribution_mode
    
    def allocate_rewards_per_turn(self,
                                  step_ids: torch.Tensor,  # [seq_len], which turn each token belongs to
                                  per_turn_scores: Dict[int, float],  # turn_id -> reward
                                  response_mask: torch.Tensor,  # [seq_len], valid token mask
                                  device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Allocate per-turn rewards to token-level reward tensor.
        
        Args:
            step_ids: [seq_len] tensor indicating which turn (step) each token belongs to
            per_turn_scores: Dictionary mapping turn_id to its reward score
            response_mask: [seq_len] tensor indicating valid tokens (1 = valid, 0 = padding)
            device: Target device for output tensor
        
        Returns:
            reward_tensor: [seq_len] tensor with rewards allocated to tokens
        """
        if device is None:
            device = step_ids.device
        
        seq_len = step_ids.shape[0]
        reward_tensor = torch.zeros(seq_len, dtype=torch.float32, device=device)
        
        # For each turn with a reward, allocate it to the appropriate tokens
        for turn_id, score in per_turn_scores.items():
            # Find tokens belonging to this turn
            turn_mask = (step_ids == turn_id) & (response_mask > 0)
            num_tokens = turn_mask.sum().item()
            
            if num_tokens == 0:
                logger.warning(f"Turn {turn_id} has reward {score:.3f} but no tokens")
                continue
            
            # Distribute reward according to mode
            if self.distribution_mode == "even":
                # Distribute evenly across all tokens in the turn
                reward_per_token = score / num_tokens
                reward_tensor[turn_mask] = reward_per_token
            
            elif self.distribution_mode == "last_token":
                # Assign all reward to the last token of the turn
                turn_indices = turn_mask.nonzero(as_tuple=True)[0]
                if len(turn_indices) > 0:
                    last_idx = turn_indices[-1]
                    reward_tensor[last_idx] = score
            
            elif self.distribution_mode == "first_and_last":
                # Split reward between first and last token
                turn_indices = turn_mask.nonzero(as_tuple=True)[0]
                if len(turn_indices) > 0:
                    first_idx = turn_indices[0]
                    last_idx = turn_indices[-1]
                    if first_idx == last_idx:
                        reward_tensor[first_idx] = score
                    else:
                        reward_tensor[first_idx] = score / 2
                        reward_tensor[last_idx] = score / 2
            
            else:
                raise ValueError(f"Unknown distribution mode: {self.distribution_mode}")
        
        return reward_tensor
    
    def compute_per_turn_scores_from_ledger(self,
                                           event_ledger: EventLedger,
                                           max_turns: int,
                                           final_answer_correct: bool,
                                           final_answer_score: float,
                                           config: Dict) -> Dict[int, float]:
        """
        Compute reward scores for each turn based on event ledger.
        
        This is the key function that translates events into per-turn rewards.
        
        Args:
            event_ledger: Event ledger containing all trajectory events
            max_turns: Maximum number of turns
            final_answer_correct: Whether the final answer was correct
            final_answer_score: Score for the final answer quality
            config: Configuration dictionary with reward parameters
        
        Returns:
            Dictionary mapping turn_id to reward score
        """
        per_turn_scores = {}
        
        # 1. Process each turn and assign rewards based on actions
        for turn_id in range(max_turns):
            turn_events = event_ledger.get_events_at_turn(turn_id)
            turn_score = 0.0
            
            for event in turn_events:
                if event.event_type == EventType.SEARCH:
                    # Small positive reward for valid search
                    if event.metadata.get('valid', True):
                        turn_score += config.get('search_reward', 0.0)
                
                elif event.event_type == EventType.INFORMATION_SUMMARY:
                    # Check if there was evidence before this turn
                    has_evidence = event_ledger.has_evidence_before_turn(turn_id)
                    if has_evidence:
                        # Reward summary when evidence exists
                        turn_score += config.get('information_summary_bonus', 0.3)
                    else:
                        # Penalize summary without evidence (hallucination)
                        turn_score += config.get('information_summary_penalty', -0.3)
                
                elif event.event_type == EventType.ANSWER:
                    # Assign final answer score to the turn with answer
                    turn_score += final_answer_score
                    
                    # Format bonus if answer is at appropriate time
                    if final_answer_correct:
                        turn_score += config.get('format_bonus', 0.1)
            
            # Only store non-zero scores
            if abs(turn_score) > 1e-6:
                per_turn_scores[turn_id] = turn_score
        
        logger.info(f"Computed per-turn scores: {per_turn_scores}")
        return per_turn_scores
    
    def allocate_with_ledger(self,
                            step_ids: torch.Tensor,
                            response_mask: torch.Tensor,
                            event_ledger: EventLedger,
                            final_answer_score: float,
                            max_turns: int,
                            config: Dict,
                            device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Complete pipeline: compute per-turn scores from ledger and allocate to tokens.
        
        Args:
            step_ids: [seq_len] tensor of turn IDs
            response_mask: [seq_len] tensor of valid token mask
            event_ledger: Event ledger with full trajectory
            final_answer_score: Score for final answer quality
            max_turns: Maximum number of turns
            config: Reward configuration
            device: Target device
        
        Returns:
            reward_tensor: [seq_len] tensor with allocated rewards
        """
        # Compute per-turn scores from ledger
        final_correct = (final_answer_score > 0.5)  # Simple threshold
        per_turn_scores = self.compute_per_turn_scores_from_ledger(
            event_ledger=event_ledger,
            max_turns=max_turns,
            final_answer_correct=final_correct,
            final_answer_score=final_answer_score,
            config=config
        )
        
        # Allocate to tokens
        reward_tensor = self.allocate_rewards_per_turn(
            step_ids=step_ids,
            per_turn_scores=per_turn_scores,
            response_mask=response_mask,
            device=device
        )
        
        return reward_tensor


def create_per_turn_reward_tensor_batch(
    step_ids_batch: torch.Tensor,  # [batch, seq_len]
    response_mask_batch: torch.Tensor,  # [batch, seq_len]
    event_ledgers: List[EventLedger],  # List of ledgers, one per batch item
    final_answer_scores: torch.Tensor,  # [batch]
    max_turns: int,
    config: Dict,
    distribution_mode: str = "even"
) -> torch.Tensor:
    """
    Create per-turn reward tensors for entire batch.
    
    Args:
        step_ids_batch: [batch, seq_len] tensor of turn IDs
        response_mask_batch: [batch, seq_len] tensor of valid token masks
        event_ledgers: List of event ledgers (one per batch item)
        final_answer_scores: [batch] tensor of answer quality scores
        max_turns: Maximum number of turns
        config: Reward configuration
        distribution_mode: How to distribute rewards within each turn
    
    Returns:
        reward_tensor_batch: [batch, seq_len] tensor with allocated rewards
    """
    batch_size, seq_len = step_ids_batch.shape
    device = step_ids_batch.device
    
    allocator = PerTurnRewardAllocator(distribution_mode=distribution_mode)
    reward_batch = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    
    for i in range(batch_size):
        if i < len(event_ledgers):
            reward_batch[i] = allocator.allocate_with_ledger(
                step_ids=step_ids_batch[i],
                response_mask=response_mask_batch[i],
                event_ledger=event_ledgers[i],
                final_answer_score=final_answer_scores[i].item(),
                max_turns=max_turns,
                config=config,
                device=device
            )
        else:
            logger.warning(f"No event ledger for batch item {i}, using zero rewards")
    
    return reward_batch


