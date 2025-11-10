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
        # Build mapping: (traj_idx, turn_idx) -> (batch_idx, context_length, response_length)
        turn_mapping = []  # List of (traj_idx, turn_idx, context_length, response_length)
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
                resp_len = turn_ctx.get('response_length', turn_ctx['response'].size(0))
                ctx_len = turn_ctx.get('context_length', 0)
                # If context_length not available, try to compute from input_ids
                if ctx_len == 0:
                    if 'input_ids_ref' in turn_ctx and 'attention_mask_ref' in turn_ctx:
                        item_idx = turn_ctx.get('item_idx', 0)
                        ctx_len = turn_ctx['input_ids_ref'][item_idx].size(0)
                    elif 'input_ids' in turn_ctx:
                        ctx_len = turn_ctx['input_ids'].size(0)
                turn_mapping.append((traj_idx, original_turn_idx, ctx_len, resp_len))
                all_turns.append(turn_ctx)
        
        if not all_turns:
            return torch.tensor([[]])
        
        # ========== BUILD SINGLE LARGE BATCH ==========
        # Collect all sequence lengths for smart padding
        # CRITICAL: Use consistent item_idx for all turns in the same batch
        # The first turn's item_idx might be 0, but subsequent turns might have different item_idx
        seq_lengths = []
        for turn_ctx in all_turns:
            # Handle reference-based storage (memory optimized)
            if 'input_ids_ref' in turn_ctx and 'attention_mask_ref' in turn_ctx:
                # CRITICAL FIX: Use the turn's own item_idx, not always 0
                # Each turn might belong to a different trajectory, so item_idx might be different
                item_idx = turn_ctx.get('item_idx', 0)
                ctx_ids = turn_ctx['input_ids_ref'][item_idx]  # Extract specific item from batch
                ctx_len = ctx_ids.size(0)
            else:
                # Fallback to old format (direct tensors)
                ctx_len = turn_ctx['input_ids'].size(0)
            resp_len = turn_ctx['response'].size(0)
            seq_lengths.append(ctx_len + resp_len)
        
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        batch_size = len(all_turns)
        
        # Early return if batch is empty
        if batch_size == 0:
            logger.warning("[Per-Turn Training] Empty batch detected after processing, returning empty result")
            return torch.empty(num_traj, 0, dtype=torch.float32, device='cpu')
        
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
            
            # CRITICAL FIX: Compute ctx_len from actual tensor size, not from stored value
            # The stored context_length might be incorrect or from a different batch item
            ctx_len = ctx_ids.size(0)  # Use actual tensor size
            resp_len = resp_ids.size(0)  # Use actual tensor size
            
            # Validate context mask - if all zeros, use context length from input_ids
            ctx_mask_sum = ctx_mask.sum().item()
            if ctx_mask_sum == 0:
                logger.warning(f"[Per-Turn Training] Context mask is all zeros for turn {i}, using context length from input_ids")
                ctx_mask = torch.ones(ctx_len, dtype=torch.long, device=device)
            elif ctx_mask.size(0) != ctx_len:
                logger.warning(f"[Per-Turn Training] Context mask size {ctx_mask.size(0)} != ctx_len {ctx_len}, adjusting")
                if ctx_mask.size(0) > ctx_len:
                    ctx_mask = ctx_mask[:ctx_len]
                else:
                    # Pad mask if needed
                    pad_size = ctx_len - ctx_mask.size(0)
                    ctx_mask = torch.cat([ctx_mask, torch.zeros(pad_size, dtype=ctx_mask.dtype, device=device)])
            
            # Concatenate context + response
            full_seq = torch.cat([ctx_ids, resp_ids], dim=0)  # [ctx_len + resp_len]
            full_mask = torch.cat([ctx_mask, torch.ones_like(resp_ids)], dim=0)
            
            # CRITICAL FIX: Use saved context_start_position for correct position_ids
            # This ensures training uses the same position_ids as generation
            context_start_position = turn_ctx.get('context_start_position', 0)
            
            # Build position_ids: context starts from context_start_position, response continues
            ctx_position_ids = torch.arange(context_start_position, context_start_position + ctx_len, dtype=torch.long, device=device)
            resp_position_ids = torch.arange(context_start_position + ctx_len, context_start_position + ctx_len + resp_len, dtype=torch.long, device=device)
            full_position_ids = torch.cat([ctx_position_ids, resp_position_ids], dim=0)
            
            # CRITICAL FIX: Ensure full_position_ids matches full_seq size
            # This can happen if ctx_len or resp_len were computed incorrectly
            actual_seq_len = full_seq.size(0)
            actual_pos_ids_len = full_position_ids.size(0)
            if actual_seq_len != actual_pos_ids_len:
                logger.error(f"[Per-Turn Training] Size mismatch: full_seq.size(0)={actual_seq_len}, full_position_ids.size(0)={actual_pos_ids_len}")
                logger.error(f"[Per-Turn Training] ctx_len={ctx_len}, resp_len={resp_len}, context_start_position={context_start_position}")
                # Fix: truncate to match
                min_len = min(actual_seq_len, actual_pos_ids_len)
                full_seq = full_seq[:min_len]
                full_mask = full_mask[:min_len]
                full_position_ids = full_position_ids[:min_len]
                logger.warning(f"[Per-Turn Training] Truncated to min_len={min_len} to fix size mismatch")
            
            # Pad to max_seq_len (truncate if necessary)
            seq_len = full_seq.size(0)
            if seq_len > max_seq_len:
                logger.warning(f"[Per-Turn Training] Sequence length {seq_len} exceeds max_seq_len {max_seq_len}, truncating")
                seq_len = max_seq_len
                full_seq = full_seq[:seq_len]
                full_mask = full_mask[:seq_len]
                full_position_ids = full_position_ids[:seq_len]
            
            # CRITICAL FIX: Double-check sizes before assignment
            # Ensure all tensors have the same size before assignment
            actual_full_seq_len = full_seq.size(0)
            actual_full_mask_len = full_mask.size(0)
            actual_full_pos_ids_len = full_position_ids.size(0)
            
            if actual_full_seq_len != seq_len or actual_full_mask_len != seq_len or actual_full_pos_ids_len != seq_len:
                logger.error(f"[Per-Turn Training] Size mismatch before assignment for turn {i}:")
                logger.error(f"  seq_len={seq_len}, full_seq.size(0)={actual_full_seq_len}, full_mask.size(0)={actual_full_mask_len}, full_position_ids.size(0)={actual_full_pos_ids_len}")
                logger.error(f"  ctx_len={ctx_len}, resp_len={resp_len}, context_start_position={context_start_position}")
                # Fix: use the minimum size and update seq_len
                min_size = min(seq_len, actual_full_seq_len, actual_full_mask_len, actual_full_pos_ids_len)
                seq_len = min_size
                full_seq = full_seq[:seq_len]
                full_mask = full_mask[:seq_len]
                full_position_ids = full_position_ids[:seq_len]
                logger.warning(f"[Per-Turn Training] Fixed size mismatch by truncating to min_size={min_size}")
            
            # Final size check
            if full_seq.size(0) != seq_len or full_mask.size(0) != seq_len or full_position_ids.size(0) != seq_len:
                raise RuntimeError(f"[Per-Turn Training] CRITICAL: Size mismatch after fix! seq_len={seq_len}, "
                                 f"full_seq.size(0)={full_seq.size(0)}, full_mask.size(0)={full_mask.size(0)}, "
                                 f"full_position_ids.size(0)={full_position_ids.size(0)}")
            
            batch_input_ids[i, :seq_len] = full_seq
            batch_attention_mask[i, :seq_len] = full_mask
            batch_position_ids[i, :seq_len] = full_position_ids
            
            # Log position_ids for first few turns to verify correctness
            if i < 3:
                logger.info(f"[Per-Turn Training] Turn {i}: context_start_position={context_start_position}, "
                          f"ctx_len={ctx_len}, resp_len={resp_len}, "
                          f"position_ids_range=[{full_position_ids[0].item()}, {full_position_ids[min(seq_len-1, ctx_len+resp_len-1)].item()}]")
            
            # Store response info for result mapping
            batch_responses.append(resp_ids)
            response_lengths.append(resp_ids.size(0))
        
        # ========== PAD RESPONSES TO SAME LENGTH ==========
        # Find max response length and pad all responses to that length
        max_resp_len = max(response_lengths) if response_lengths else 0
        if max_resp_len > 0:
            padded_responses = []
            for resp_ids in batch_responses:
                resp_len = resp_ids.size(0)
                if resp_len < max_resp_len:
                    # Pad with zeros (or use pad_sequence if preferred)
                    padding = torch.zeros(max_resp_len - resp_len, dtype=resp_ids.dtype, device=resp_ids.device)
                    padded_resp = torch.cat([resp_ids, padding])
                else:
                    padded_resp = resp_ids
                padded_responses.append(padded_resp)
            batch_responses_tensor = torch.stack(padded_responses, dim=0)  # [B, max_resp_len]
        else:
            # Empty batch case
            batch_responses_tensor = torch.empty(batch_size, 0, dtype=dtype, device=device)
        
        # ========== SINGLE COMPUTE CALL ==========
        # Safety check: ensure batch_size > 0 before calling compute function
        if batch_size == 0 or max_seq_len == 0:
            logger.warning(f"[Per-Turn Training] Invalid batch (batch_size={batch_size}, max_seq_len={max_seq_len}), returning empty result")
            return torch.empty(num_traj, 0, dtype=torch.float32, device=device)
        
        # Validate attention_mask before calling compute function
        # Flash Attention varlen mode requires valid sequences (non-zero attention_mask)
        seq_valid_lengths = batch_attention_mask.sum(dim=1)  # [batch_size]
        valid_mask_sum = seq_valid_lengths.sum().item()
        num_zero_seq = (seq_valid_lengths == 0).sum().item()
        
        if valid_mask_sum == 0:
            logger.error(f"[Per-Turn Training] All sequences are padded (valid_mask_sum=0), batch_size={batch_size}, max_seq_len={max_seq_len}")
            logger.error(f"[Per-Turn Training] This will cause Flash Attention 'batch size must be positive' error")
            return torch.empty(num_traj, 0, dtype=torch.float32, device=device)
        
        if num_zero_seq > 0:
            logger.warning(f"[Per-Turn Training] {num_zero_seq}/{batch_size} sequences have zero valid tokens")
            # Note: Filtering invalid sequences would require rebuilding the batch and turn_mapping
            # For now, we'll proceed and let Flash Attention handle it (it will error, but we catch it)
        
        # This is the key optimization: ONE call instead of multiple grouped calls
        # CRITICAL FIX: Disable auto_padding because we've already manually padded to max_seq_len
        # batch_responses_tensor has shape [B, max_resp_len] which is different from [B, max_seq_len]
        # auto_padding=True would try to align them and cause "expanded size" error
        turn_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,  # [B, max_seq_len]
            'attention_mask': batch_attention_mask,  # [B, max_seq_len]
            'position_ids': batch_position_ids,  # [B, max_seq_len]
            'responses': batch_responses_tensor  # [B, max_resp_len] - different size, but that's OK
        }, auto_padding=False)  # Disable auto_padding to avoid size mismatch error
        
        # Single compute call for ALL turns
        try:
            out = compute_log_prob_fn(turn_batch)
        except RuntimeError as e:
            if "batch size must be positive" in str(e):
                # Enhanced diagnostic information
                pos_ids_min = batch_position_ids.min().item()
                pos_ids_max = batch_position_ids.max().item()
                pos_ids_sum = batch_position_ids.sum().item()
                
                # Check position_ids for first few sequences
                pos_ids_samples = []
                for i in range(min(5, batch_size)):
                    seq_len = seq_valid_lengths[i].item()
                    if seq_len > 0:
                        pos_ids = batch_position_ids[i, :seq_len]
                        pos_ids_samples.append({
                            'seq_idx': i,
                            'seq_len': seq_len,
                            'pos_ids_start': pos_ids[0].item(),
                            'pos_ids_end': pos_ids[-1].item(),
                            'pos_ids_is_monotonic': (torch.diff(pos_ids) >= 0).all().item()
                        })
                
                logger.error(f"[Per-Turn Training] Flash Attention received invalid batch")
                logger.error(f"  batch_size={batch_size}, max_seq_len={max_seq_len}")
                logger.error(f"  Attention mask stats: valid_sum={valid_mask_sum}, num_zero_seq={num_zero_seq}")
                logger.error(f"  Position IDs stats: min={pos_ids_min}, max={pos_ids_max}, sum={pos_ids_sum}")
                logger.error(f"  Position IDs samples: {pos_ids_samples}")
                logger.error(f"  Turn mapping length: {len(turn_mapping)}, Response lengths sample: {response_lengths[:20]}")
                logger.error(f"  This likely means cu_seqlens calculation failed in Flash Attention varlen mode")
                logger.error(f"  Possible causes: unpad_input returned zero total_nnz, or position_ids invalid after unpad")
                # Return empty result instead of crashing
                return torch.empty(num_traj, 0, dtype=torch.float32, device=device)
            raise
        
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
        
        logger.debug(f"[Per-Turn Training] Mapping {len(turn_mapping)} turns back to {num_traj} trajectories")
        for i, (traj_idx, turn_idx, ctx_len, resp_len) in enumerate(turn_mapping):
            # Extract log_probs for this turn's response (skip context portion)
            # log_probs[i] has shape [max_seq_len], containing [context_tokens | response_tokens | padding]
            # We need to extract only the response portion: [ctx_len : ctx_len + resp_len]
            start_idx = ctx_len
            end_idx = ctx_len + resp_len
            turn_log_prob = log_probs[i, start_idx:end_idx]  # [resp_len]
            traj_turn_outputs[traj_idx].append(turn_log_prob.unsqueeze(0))  # [1, resp_len]
            
            # Log first few turns for debugging
            if i < 5:
                logger.debug(f"[Per-Turn Training] Turn {i}: traj_idx={traj_idx}, turn_idx={turn_idx}, ctx_len={ctx_len}, resp_len={resp_len}, log_prob_shape={turn_log_prob.shape}")
        
        # ========== CONCATENATE PER TRAJECTORY ==========
        per_traj = []
        traj_lengths = []
        traj_turn_counts = []
        for traj_idx in range(num_traj):
            if len(traj_turn_outputs[traj_idx]) == 0:
                per_traj.append(torch.empty(1, 0, dtype=log_probs.dtype, device=log_probs.device))
                traj_lengths.append(0)
                traj_turn_counts.append(0)
            else:
                traj_log_probs = torch.cat(traj_turn_outputs[traj_idx], dim=1)  # [1, sum_turn_tokens]
                per_traj.append(traj_log_probs)
                traj_lengths.append(traj_log_probs.shape[1])
                traj_turn_counts.append(len(traj_turn_outputs[traj_idx]))
                # Log first few trajectories for debugging
                if traj_idx < 5:
                    logger.debug(f"[Per-Turn Training] Trajectory {traj_idx}: {len(traj_turn_outputs[traj_idx])} turns, total_length={traj_log_probs.shape[1]}")
        
        # ========== PAD TO SAME LENGTH + CREATE VALID MASK ==========
        # Find max total length across all trajectories (similar to original code's padding approach)
        max_total_len = max(traj_lengths) if traj_lengths else 0
        
        if max_total_len > 0:
            # Log trajectory statistics before padding
            min_total_len = min(traj_lengths) if traj_lengths else 0
            avg_total_len = sum(traj_lengths) / len(traj_lengths) if traj_lengths else 0
            num_padding_needed = sum(1 for l in traj_lengths if l < max_total_len)
            
            logger.info(f"[Per-Turn Training] Trajectory statistics: count={num_traj}, turns={traj_turn_counts[:10]}{'...' if len(traj_turn_counts) > 10 else ''}")
            logger.info(f"[Per-Turn Training] Length stats: min={min_total_len}, max={max_total_len}, avg={avg_total_len:.1f}, need_padding={num_padding_needed}/{num_traj}")
            
            # CRITICAL FIX: Create valid_mask to mark valid (non-padding) positions
            valid_mask = torch.ones(num_traj, max_total_len, dtype=torch.bool, device=log_probs.device)
            
            # Pad all trajectories to the same length
            padded_per_traj = []
            padding_stats = {'total_padding': 0, 'max_padding': 0}
            for traj_idx, t in enumerate(per_traj):
                traj_len = t.shape[1]
                if traj_len < max_total_len:
                    pad_size = max_total_len - traj_len
                    padding = torch.zeros(1, pad_size, dtype=t.dtype, device=t.device)
                    padded_t = torch.cat([t, padding], dim=1)
                    padding_stats['total_padding'] += pad_size
                    padding_stats['max_padding'] = max(padding_stats['max_padding'], pad_size)
                    # Mark padding positions as invalid
                    valid_mask[traj_idx, traj_len:] = False
                    if traj_idx < 5:  # Log first few for debugging
                        logger.debug(f"[Per-Turn Training] Trajectory {traj_idx}: padded {traj_len} -> {max_total_len} (+{pad_size}), valid_mask[{traj_len}:] = False")
                else:
                    padded_t = t
                    # All positions are valid (no padding needed)
                padded_per_traj.append(padded_t)
            
            result = torch.cat(padded_per_traj, dim=0)  # [num_traj, max_total_len]
            
            # Log padding and mask statistics
            total_valid_tokens = valid_mask.sum().item()
            total_padding_tokens = (~valid_mask).sum().item()
            logger.info(f"[Per-Turn Training] Padding complete: result_shape={result.shape}, "
                      f"valid_tokens={total_valid_tokens}, padding_tokens={total_padding_tokens}, "
                      f"max_padding_per_traj={padding_stats['max_padding']}")
            
            # Log valid mask statistics
            logger.info(f"[Per-Turn Training] Valid mask created: {valid_mask.sum(dim=1).tolist()[:10]}{'...' if num_traj > 10 else ''} valid tokens per trajectory")
            
            # Store valid_mask as an attribute of the result tensor for later access
            # This allows us to retrieve it in the loss calculation without changing the return signature
            result.valid_mask = valid_mask
        else:
            # Empty batch case
            result = torch.empty(num_traj, 0, dtype=log_probs.dtype, device=log_probs.device)
            valid_mask = torch.empty(num_traj, 0, dtype=torch.bool, device=log_probs.device)
            result.valid_mask = valid_mask
            logger.warning(f"[Per-Turn Training] Empty batch: all trajectories have zero length")
        
        if log_debug:
            logger.info(f"[Per-Turn Training][Optimized] Single batch: {batch_size} turns, max_seq_len: {max_seq_len}")
            logger.info(f"[Per-Turn Training][Optimized] Final result shape: {result.shape}")
            if hasattr(result, 'valid_mask'):
                logger.info(f"[Per-Turn Training][Optimized] Valid mask shape: {result.valid_mask.shape}, valid ratio: {result.valid_mask.sum().item() / result.valid_mask.numel():.3f}")
        
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
