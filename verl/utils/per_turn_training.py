"""
Per-Turn Training Utilities

This module provides utilities for per-turn context-aware training.
Solves the context mismatch problem where early turns are trained with wrong context.

Core Problem:
- Generation: Turn i uses context_i
- Training (before): All turns use final compressed context
- Training (after): Each turn uses its actual context_i

Solution:
- Save per-turn contexts during generation
- Use corresponding context for each turn during training
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from verl import DataProto
import logging

logger = logging.getLogger(__name__)


class PerTurnContextManager:
    """
    Manages per-turn contexts for correct log_prob computation.
    
    Usage:
        1. Generation saves per-turn contexts
        2. Training retrieves correct context for each turn
        3. Log_prob computed with matching context
    """
    
    @staticmethod
    def extract_per_turn_contexts(data: DataProto) -> Optional[List[List[Dict]]]:
        """
        Extract per-turn contexts from DataProto.
        
        Args:
            data: DataProto containing per_turn_contexts in non_tensor_batch
            
        Returns:
            List of per-turn contexts for each trajectory, or None if not available
        """
        if not hasattr(data, 'non_tensor_batch'):
            logger.warning("[Per-Turn Training] DataProto has no non_tensor_batch attribute")
            return None
            
        if 'per_turn_contexts' not in data.non_tensor_batch:
            logger.warning("[Per-Turn Training] No per_turn_contexts found in non_tensor_batch")
            return None
        
        per_turn_data = data.non_tensor_batch['per_turn_contexts']
        
        # Convert numpy array to list
        if isinstance(per_turn_data, np.ndarray):
            per_turn_contexts = per_turn_data.tolist()
        else:
            per_turn_contexts = per_turn_data
        
        # ========== DEBUGGING: Extraction Validation ==========
        logger.info(f"[Per-Turn Training] Extracted per-turn contexts for {len(per_turn_contexts)} trajectories")
        
        if per_turn_contexts:
            total_turns = sum(len(traj) for traj in per_turn_contexts)
            logger.info(f"[Per-Turn Training] Total turns across all trajectories: {total_turns}")
            
            # Show first trajectory details
            first_traj = per_turn_contexts[0]
            logger.info(f"[Per-Turn Training] First trajectory: {len(first_traj)} turns")
            for turn_idx, turn_ctx in enumerate(first_traj[:3]):  # Show first 3 turns
                ctx_len = turn_ctx.get('context_length', 0)
                resp_len = turn_ctx.get('response_length', 0)
                logger.info(f"[Per-Turn Training]   Turn {turn_idx}: context={ctx_len}, response={resp_len}")
        
        return per_turn_contexts
    
    @staticmethod
    def reconstruct_per_turn_input_ids(
        per_turn_contexts: List[Dict],
        step_ids: torch.Tensor,
        responses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct input_ids for each turn using saved contexts.
        
        Args:
            per_turn_contexts: List of contexts for each turn
            step_ids: Tensor mapping response tokens to turn IDs
            responses: Full responses tensor
            
        Returns:
            Tuple of (reconstructed_input_ids, turn_boundaries)
            reconstructed_input_ids: Concatenated input_ids for all turns
            turn_boundaries: Indices where each turn's data starts/ends
        """
        all_input_ids = []
        turn_boundaries = []
        current_pos = 0
        
        for turn_idx, turn_ctx in enumerate(per_turn_contexts):
            # Get context for this turn
            context_ids = turn_ctx['input_ids']
            
            # Get response for this turn from full responses
            turn_mask = (step_ids == turn_idx)
            if turn_mask.any():
                turn_response = responses[turn_mask]
                
                # Concatenate context + response
                turn_input_ids = torch.cat([context_ids, turn_response], dim=0)
            else:
                turn_input_ids = context_ids
            
            all_input_ids.append(turn_input_ids)
            turn_boundaries.append((current_pos, current_pos + len(turn_input_ids)))
            current_pos += len(turn_input_ids)
        
        # Concatenate all turns
        reconstructed = torch.cat(all_input_ids, dim=0) if all_input_ids else torch.tensor([])
        
        return reconstructed, turn_boundaries
    
    @staticmethod
    def compute_per_turn_log_probs(
        data: DataProto,
        compute_log_prob_fn,
        use_per_turn_context: bool = True
    ) -> torch.Tensor:
        """
        Compute log probabilities using per-turn contexts.
        
        Args:
            data: DataProto with per_turn_contexts
            compute_log_prob_fn: Function to compute log_prob
            use_per_turn_context: Whether to use per-turn context (if False, use original method)
            
        Returns:
            Log probabilities tensor
        """
        if not use_per_turn_context:
            logger.info("[Per-Turn Training] Using original compressed context")
            return compute_log_prob_fn(data)
        
        # Extract per-turn contexts
        per_turn_contexts_batch = PerTurnContextManager.extract_per_turn_contexts(data)
        
        if per_turn_contexts_batch is None:
            logger.warning("[Per-Turn Training] No per-turn contexts found, falling back to original method")
            return compute_log_prob_fn(data)
        
        logger.info(f"[Per-Turn Training] Computing log_prob with per-turn contexts for {len(per_turn_contexts_batch)} trajectories")
        
        # ========== DEBUGGING: Context Comparison ==========
        original_input_ids = data.batch.get('input_ids', torch.tensor([]))
        if original_input_ids.numel() > 0:
            original_length = original_input_ids.size(1)
            logger.info(f"[DEBUG] Original compressed context length: {original_length} tokens")
        
        # For each trajectory, compute log_prob using correct context
        batch_log_probs = []
        
        for batch_idx, per_turn_contexts in enumerate(per_turn_contexts_batch):
            turn_log_probs = []
            
            # ========== DEBUGGING: Per-Trajectory Analysis ==========
            if batch_idx < 2:  # Debug first 2 trajectories
                logger.info(f"[DEBUG] ========== TRAJECTORY {batch_idx} ANALYSIS ==========")
                logger.info(f"[DEBUG] Trajectory {batch_idx}: {len(per_turn_contexts)} turns")
            
            for turn_idx, turn_ctx in enumerate(per_turn_contexts):
                # Create mini-batch for this single turn
                turn_input_ids = turn_ctx['input_ids'].unsqueeze(0)  # [1, seq_len]
                turn_attention_mask = turn_ctx['attention_mask'].unsqueeze(0)
                turn_response = turn_ctx['response'].unsqueeze(0)
                
                # ========== DEBUGGING: Turn-by-Turn Context Info ==========
                if batch_idx < 2 and turn_idx < 3:  # Debug first 2 trajectories, first 3 turns
                    ctx_len = turn_ctx['context_length']
                    resp_len = turn_ctx['response_length']
                    total_len = ctx_len + resp_len
                    
                    # Compare with original context
                    if original_input_ids.numel() > 0:
                        diff = abs(ctx_len - original_length)
                        match_str = "✅ MATCH" if diff < 100 else f"⚠️ DIFF({diff})"
                    else:
                        match_str = "N/A"
                    
                    logger.info(f"[DEBUG]   Turn {turn_idx}: context={ctx_len}, response={resp_len}, total={total_len} {match_str}")
                
                # Create DataProto for this turn (build position_ids as simple arange)
                turn_seq_len = turn_input_ids.size(1) + turn_response.size(1)
                turn_position_ids = torch.arange(0, turn_seq_len, dtype=torch.long).unsqueeze(0)

                turn_data = DataProto.from_dict({
                    'input_ids': torch.cat([turn_input_ids, turn_response], dim=1),
                    'attention_mask': torch.cat([turn_attention_mask, torch.ones_like(turn_response)], dim=1),
                    'position_ids': turn_position_ids,
                    'responses': turn_response
                }, auto_padding=True)
                
                # Compute log_prob for this turn
                turn_out = compute_log_prob_fn(turn_data)
                # Support both DataProto and raw tensor returns (dedupe DP chunks)
                if isinstance(turn_out, DataProto):
                    turn_log_prob = turn_out.batch.get('old_log_probs', None)
                    if turn_log_prob is None:
                        raise RuntimeError("compute_log_prob_fn returned DataProto without 'old_log_probs'")
                    if turn_log_prob.dim() == 2 and turn_log_prob.size(0) > 1:
                        turn_log_prob = turn_log_prob[:1]
                else:
                    if hasattr(turn_out, 'dim') and turn_out.dim() == 2 and turn_out.size(0) > 1:
                        turn_log_prob = turn_out[:1]
                    else:
                        turn_log_prob = turn_out
                turn_log_probs.append(turn_log_prob)
                
                # ========== DEBUGGING: Log Prob Validation ==========
                if batch_idx < 2 and turn_idx < 3:
                    log_prob_mean = turn_log_prob.mean().item() if turn_log_prob.numel() > 0 else 0
                    log_prob_std = turn_log_prob.std().item() if turn_log_prob.numel() > 0 else 0
                    logger.info(f"[DEBUG]   Turn {turn_idx} log_prob: mean={log_prob_mean:.4f}, std={log_prob_std:.4f}")
            
            # Concatenate log_probs for all turns
            traj_log_probs = torch.cat(turn_log_probs, dim=1) if turn_log_probs else torch.tensor([[]])
            batch_log_probs.append(traj_log_probs)
            
            if batch_idx < 2:
                logger.info(f"[DEBUG] Trajectory {batch_idx} final log_probs shape: {traj_log_probs.shape}")
        
        # Stack to create batch
        result = torch.cat(batch_log_probs, dim=0) if batch_log_probs else torch.tensor([[]])
        
        # ========== DEBUGGING: Final Results ==========
        logger.info(f"[Per-Turn Training] Computed log_probs shape: {result.shape}")
        if result.numel() > 0:
            logger.info(f"[Per-Turn Training] Log_prob statistics: mean={result.mean().item():.4f}, std={result.std().item():.4f}")
            logger.info(f"[Per-Turn Training] Log_prob range: [{result.min().item():.4f}, {result.max().item():.4f}]")
        
        return result

    @staticmethod
    def compute_per_turn_log_probs_batched(
        data: DataProto,
        compute_log_prob_fn,
        use_per_turn_context: bool = True,
        log_debug: bool = False,
        max_turns_per_traj: Optional[int] = None,
    ) -> torch.Tensor:
        """
        OPTIMIZED: Single-batch per-turn compute for maximum efficiency.
        
        Key optimizations:
        1. Single compute call for ALL turns (not grouped by response_length)
        2. Smart padding to handle variable sequence lengths
        3. Efficient result mapping back to trajectories
        4. Minimal memory overhead
        
        Returns: [num_trajectories, sum_turn_tokens_per_traj]
        """
        if not use_per_turn_context:
            return compute_log_prob_fn(data)

        per_turn_contexts_batch = PerTurnContextManager.extract_per_turn_contexts(data)
        if per_turn_contexts_batch is None:
            return compute_log_prob_fn(data)

        num_traj = len(per_turn_contexts_batch)
        
        # ========== FLATTEN ALL TURNS WITH TRACKING ==========
        # Build mapping: (traj_idx, turn_idx) -> (batch_idx, response_length)
        turn_mapping = []  # List of (traj_idx, turn_idx, response_length)
        all_turns = []     # List of turn contexts
        
        for traj_idx, traj in enumerate(per_turn_contexts_batch):
            # SELECTIVE TRAINING: Only use last N turns per trajectory
            if max_turns_per_traj is not None and len(traj) > max_turns_per_traj:
                # Use only the last max_turns_per_traj turns
                selected_turns = traj[-max_turns_per_traj:]
            else:
                selected_turns = traj
            
            for turn_idx, turn_ctx in enumerate(selected_turns):
                # Adjust turn_idx for mapping back to original trajectory
                original_turn_idx = len(traj) - len(selected_turns) + turn_idx if max_turns_per_traj else turn_idx
                turn_mapping.append((traj_idx, original_turn_idx, turn_ctx.get('response_length', turn_ctx['response'].size(0))))
                all_turns.append(turn_ctx)
        
        if not all_turns:
            return torch.tensor([[]])
        
        # ========== BUILD SINGLE LARGE BATCH ==========
        # Collect all sequence lengths for smart padding
        seq_lengths = []
        for turn_ctx in all_turns:
            # Handle reference-based storage (memory optimized)
            if 'input_ids_ref' in turn_ctx and 'attention_mask_ref' in turn_ctx:
                item_idx = turn_ctx.get('item_idx', 0)
                ctx_ids = turn_ctx['input_ids_ref'][item_idx]  # Extract specific item from batch
                ctx_len = ctx_ids.size(0)
            else:
                # Fallback to old format (direct tensors)
                ctx_len = turn_ctx['input_ids'].size(0)
            resp_len = turn_ctx['response'].size(0)
            seq_lengths.append(ctx_len + resp_len)
        
        max_seq_len = max(seq_lengths)
        batch_size = len(all_turns)
        
        # Pre-allocate tensors for efficiency
        # Get device and dtype from first turn (handle both formats)
        first_turn = all_turns[0]
        if 'input_ids_ref' in first_turn and 'attention_mask_ref' in first_turn:
            item_idx = first_turn.get('item_idx', 0)
            device = first_turn['input_ids_ref'][item_idx].device
            dtype = first_turn['input_ids_ref'][item_idx].dtype
        else:
            device = first_turn['input_ids'].device
            dtype = first_turn['input_ids'].dtype
        
        batch_input_ids = torch.zeros(batch_size, max_seq_len, dtype=dtype, device=device)
        batch_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        batch_position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        batch_responses = []
        response_lengths = []
        
        # Fill batch tensors (handle both old and new storage formats)
        for i, turn_ctx in enumerate(all_turns):
            # Handle reference-based storage (memory optimized)
            if 'input_ids_ref' in turn_ctx and 'attention_mask_ref' in turn_ctx:
                item_idx = turn_ctx.get('item_idx', i)
                ctx_ids = turn_ctx['input_ids_ref'][item_idx]  # Extract specific item from batch
                ctx_mask = turn_ctx['attention_mask_ref'][item_idx]
            else:
                # Fallback to old format (direct tensors)
                ctx_ids = turn_ctx['input_ids']
                ctx_mask = turn_ctx['attention_mask']
            
            resp_ids = turn_ctx['response']  # [resp_len]
            
            # Concatenate context + response
            full_seq = torch.cat([ctx_ids, resp_ids], dim=0)  # [ctx_len + resp_len]
            full_mask = torch.cat([ctx_mask, torch.ones_like(resp_ids)], dim=0)
            
            # Pad to max_seq_len
            seq_len = full_seq.size(0)
            batch_input_ids[i, :seq_len] = full_seq
            batch_attention_mask[i, :seq_len] = full_mask
            batch_position_ids[i, :seq_len] = torch.arange(seq_len, dtype=torch.long, device=device)
            
            # Store response info for result mapping
            batch_responses.append(resp_ids)
            response_lengths.append(resp_ids.size(0))
        
        # ========== SINGLE COMPUTE CALL ==========
        # This is the key optimization: ONE call instead of multiple grouped calls
        turn_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids,
            'responses': torch.stack(batch_responses, dim=0)  # [B, max_resp_len]
        }, auto_padding=True)
        
        # Single compute call for ALL turns
        out = compute_log_prob_fn(turn_batch)
        
        # Extract log probabilities
        if isinstance(out, DataProto):
            log_probs = out.batch.get('old_log_probs', None)
            if log_probs is None:
                raise RuntimeError("compute_log_prob_fn returned DataProto without 'old_log_probs'")
        else:
            log_probs = out
        
        # Handle DP duplication (keep first batch_size rows)
        if log_probs.size(0) > batch_size:
            log_probs = log_probs[:batch_size]
        
        # ========== MAP RESULTS BACK TO TRAJECTORIES ==========
        # Group results by trajectory
        traj_turn_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_traj)]
        
        for i, (traj_idx, turn_idx, resp_len) in enumerate(turn_mapping):
            # Extract log_probs for this turn's response
            turn_log_prob = log_probs[i, :resp_len]  # [resp_len]
            traj_turn_outputs[traj_idx].append(turn_log_prob.unsqueeze(0))  # [1, resp_len]
        
        # ========== CONCATENATE PER TRAJECTORY ==========
        per_traj = []
        for traj_idx in range(num_traj):
            if len(traj_turn_outputs[traj_idx]) == 0:
                per_traj.append(torch.empty(1, 0))
            else:
                per_traj.append(torch.cat(traj_turn_outputs[traj_idx], dim=1))
        
        result = torch.cat(per_traj, dim=0) if per_traj else torch.tensor([[]])
        
        if log_debug:
            logger.info(f"[Per-Turn Training][Optimized] Single batch: {batch_size} turns, max_seq_len: {max_seq_len}")
            logger.info(f"[Per-Turn Training][Optimized] Result shape: {result.shape}")
        
        return result


class PerTurnTrainingValidator:
    """
    Validates that per-turn training is working correctly.
    """
    
    @staticmethod
    def validate_context_match(
        data: DataProto,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that saved per-turn contexts match the actual generation.
        
        Returns validation results and statistics.
        """
        per_turn_contexts_batch = PerTurnContextManager.extract_per_turn_contexts(data)
        
        if per_turn_contexts_batch is None:
            return {
                'valid': False,
                'error': 'No per-turn contexts found'
            }
        
        results = {
            'valid': True,
            'num_trajectories': len(per_turn_contexts_batch),
            'trajectory_stats': []
        }
        
        for traj_idx, per_turn_contexts in enumerate(per_turn_contexts_batch):
            traj_stat = {
                'trajectory_id': traj_idx,
                'num_turns': len(per_turn_contexts),
                'turns': []
            }
            
            for turn_idx, turn_ctx in enumerate(per_turn_contexts):
                turn_stat = {
                    'turn_id': turn_idx,
                    'context_length': turn_ctx['context_length'],
                    'response_length': turn_ctx.get('response_length', 0),
                    'has_response': 'response' in turn_ctx
                }
                traj_stat['turns'].append(turn_stat)
            
            results['trajectory_stats'].append(traj_stat)
        
        if verbose:
            logger.info(f"[Validation] Found {results['num_trajectories']} trajectories")
            for traj_stat in results['trajectory_stats'][:3]:  # Show first 3
                logger.info(f"[Validation]   Trajectory {traj_stat['trajectory_id']}: {traj_stat['num_turns']} turns")
                for turn_stat in traj_stat['turns']:
                    logger.info(f"[Validation]     Turn {turn_stat['turn_id']}: "
                              f"context={turn_stat['context_length']}, "
                              f"response={turn_stat['response_length']}")
        
        return results
    
    @staticmethod
    def compare_contexts(
        original_input_ids: torch.Tensor,
        per_turn_contexts: List[Dict],
        step_ids: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compare original (compressed) context with per-turn contexts.
        
        This shows the difference between what was used for generation
        vs what would be used for training without per-turn contexts.
        """
        comparison = {
            'original_length': original_input_ids.size(1),
            'per_turn_lengths': [],
            'context_mismatch_detected': False
        }
        
        for turn_idx, turn_ctx in enumerate(per_turn_contexts):
            turn_length = turn_ctx['context_length']
            comparison['per_turn_lengths'].append(turn_length)
            
            # Check if this turn's context differs from final compressed context
            if turn_idx < len(per_turn_contexts) - 1:
                # For non-final turns, context likely different
                if turn_length != comparison['original_length']:
                    comparison['context_mismatch_detected'] = True
        
        return comparison


def enable_per_turn_training(config):
    """
    Helper function to enable per-turn training in config.
    
    Usage:
        enable_per_turn_training(trainer_config)
    """
    if not hasattr(config, 'actor'):
        config.actor = {}
    
    config.actor['use_per_turn_context'] = True
    logger.info("[Per-Turn Training] Enabled per-turn context training")
    
    return config
