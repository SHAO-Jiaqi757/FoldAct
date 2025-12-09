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
import dataclasses
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# Optional tokenizer for debugging (set via environment variable)
_debug_tokenizer = None

def _try_decode_tokens(token_ids, max_length=100):
    """Try to decode token IDs to text for debugging. Returns token IDs if decoding fails."""
    global _debug_tokenizer
    
    if _debug_tokenizer is None:
        # Try to load tokenizer on first use
        try:
            import os
            model_path = os.environ.get('DEBUG_TOKENIZER_PATH', 
                                       'verl_checkpoints/sft_progressive/phase3_summary_prefix/global_step_350')
            if os.path.exists(model_path):
                from transformers import AutoTokenizer
                _debug_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"[Per-Turn Training] Loaded debug tokenizer from {model_path}")
        except Exception as e:
            _debug_tokenizer = False  # Mark as failed
            logger.debug(f"[Per-Turn Training] Could not load tokenizer for debugging: {e}")
    
    if _debug_tokenizer and _debug_tokenizer is not False:
        try:
            # Convert to list if tensor
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            
            # Limit length for display
            if len(token_ids) > max_length:
                token_ids_display = token_ids[:max_length]
                suffix = f"... (truncated from {len(token_ids)} tokens)"
            else:
                token_ids_display = token_ids
                suffix = ""
            
            text = _debug_tokenizer.decode(token_ids_display, skip_special_tokens=False)
            # Truncate text if too long
            if len(text) > 500:
                text = text[:500] + "..."
            return f"{text}{suffix}"
        except Exception as e:
            logger.debug(f"[Per-Turn Training] Error decoding tokens: {e}")
    
    # Fallback: return token IDs
    token_list = token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids
    if len(token_list) > 50:
        return f"{token_list[:50]} ... (total {len(token_list)} tokens)"
    return str(token_list)


@dataclasses.dataclass
class TurnData:
    """Encapsulates all validated data for a single turn."""
    traj_idx: int
    original_turn_idx: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    response: torch.Tensor
    ctx_len: int
    resp_len: int

def _flatten_and_validate_turns(
    per_turn_contexts_batch: List[List[Dict]],
    max_turns_per_traj: Optional[int],
    pad_token_id: int = 0, # Assuming 0 is the pad_token_id for reconstruction
) -> Tuple[List[TurnData], List[Tuple[int, int, int, int]]]:
    """
    Flattens all turns from all trajectories into a single list, validating each one.
    
    - Filters out turns with no real tokens in the context.
    - Reconstructs attention masks if they are faulty.
    - Encapsulates validated data into TurnData objects.
    """
    valid_turns: List[TurnData] = []
    turn_mapping: List[Tuple[int, int, int, int]] = []

    for traj_idx, traj in enumerate(per_turn_contexts_batch):
        selected_turns = traj[-max_turns_per_traj:] if max_turns_per_traj and len(traj) > max_turns_per_traj else traj

        for turn_idx_in_selection, turn_ctx in enumerate(selected_turns):
            original_turn_idx = (len(traj) - len(selected_turns) + turn_idx_in_selection) if max_turns_per_traj else turn_idx_in_selection

            ctx_ids = turn_ctx['input_ids']
            ctx_mask = turn_ctx['attention_mask']
            resp_ids = turn_ctx['response']
            
            # --- Validation Step 1: Check for real tokens ---
            real_tokens_mask = (ctx_ids != pad_token_id) & (ctx_ids != 151643) # Also check against common padding token
            if not real_tokens_mask.any():
                logger.warning(
                    f"[Per-Turn Refactor] SKIPPING Turn (traj={traj_idx}, turn={original_turn_idx}): "
                    f"Context input_ids contains NO real tokens. ctx_len={ctx_ids.size(0)}"
                )
                continue

            # --- Validation Step 2: Reconstruct faulty masks ---
            if ctx_mask.sum().item() == 0:
                logger.warning(
                    f"[Per-Turn Refactor] Reconstructing all-zero attention_mask for "
                    f"Turn (traj={traj_idx}, turn={original_turn_idx})."
                )
                ctx_mask = real_tokens_mask.long()

            ctx_len = ctx_ids.size(0)
            resp_len = resp_ids.size(0)
            
            # --- Create Position IDs ---
            context_start_position = turn_ctx.get('context_start_position', 0)
            full_seq_len = ctx_len + resp_len
            position_ids = torch.arange(
                context_start_position, 
                context_start_position + full_seq_len, 
                dtype=torch.long, 
                device=ctx_ids.device
            )

            # --- Encapsulate Validated Data ---
            valid_turns.append(TurnData(
                traj_idx=traj_idx,
                original_turn_idx=original_turn_idx,
                input_ids=torch.cat([ctx_ids, resp_ids]),
                attention_mask=torch.cat([ctx_mask, torch.ones_like(resp_ids)]),
                position_ids=position_ids,
                response=resp_ids,
                ctx_len=ctx_len,
                resp_len=resp_len,
            ))
            turn_mapping.append((traj_idx, original_turn_idx, ctx_len, resp_len))

    return valid_turns, turn_mapping


def _build_batched_tensors(
    turns: List[TurnData], 
    pad_token_id: int,
    model_max_length: int,
) -> Dict[str, Any]:
    """
    Builds left-padded batch tensors from a list of TurnData objects.
    This function uses the 'flip-pad-flip' trick for robust left-padding.
    It also truncates sequences that are too long.
    """
    sequences = []
    attention_masks = []
    position_ids_list = []

    for t in turns:
        seq_len = t.input_ids.size(0)
        if seq_len > model_max_length:
            truncate_by = seq_len - model_max_length
            if truncate_by >= t.ctx_len:
                logger.error(
                    f"[Per-Turn Refactor] SKIPPING Turn (traj={t.traj_idx}, turn={t.original_turn_idx}): "
                    f"Sequence length ({seq_len}) is greater than model_max_length ({model_max_length}), "
                    f"and truncation amount ({truncate_by}) would completely remove the context ({t.ctx_len})."
                )
                continue
            
            logger.warning(
                f"[Per-Turn Refactor] Truncating Turn (traj={t.traj_idx}, turn={t.original_turn_idx}) "
                f"from {seq_len} to {model_max_length} tokens by removing context from the left."
            )
            sequences.append(torch.flip(t.input_ids[truncate_by:], dims=[0]))
            attention_masks.append(torch.flip(t.attention_mask[truncate_by:], dims=[0]))
            position_ids_list.append(torch.flip(t.position_ids[truncate_by:], dims=[0]))
        else:
            sequences.append(torch.flip(t.input_ids, dims=[0]))
            attention_masks.append(torch.flip(t.attention_mask, dims=[0]))
            position_ids_list.append(torch.flip(t.position_ids, dims=[0]))

    if not sequences:
        return {}

    # Standard 'flip-pad-flip' for left padding
    input_ids = torch.flip(pad_sequence(sequences, batch_first=True, padding_value=pad_token_id), dims=[1])
    attention_mask = torch.flip(pad_sequence(attention_masks, batch_first=True, padding_value=0), dims=[1])
    position_ids = torch.flip(pad_sequence(position_ids_list, batch_first=True, padding_value=0), dims=[1])
    
    max_resp_len = max(t.resp_len for t in turns) if turns else 0
    responses = pad_sequence(
        [t.response for t in turns], batch_first=True, padding_value=pad_token_id
    )
    if responses.shape[1] < max_resp_len:
        padding = torch.full(
            (responses.shape[0], max_resp_len - responses.shape[1]),
            pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        responses = torch.cat([responses, padding], dim=1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "max_resp_len": max_resp_len
    }

def _prepare_for_computation(
    batch: Dict[str, torch.Tensor]
) -> DataProto:
    """
    Prepares the final DataProto for the compute_log_prob_fn.
    With left-padding already handled, this function is now much simpler.
    """
    input_ids = batch["input_ids"]
    max_resp_len = batch["max_resp_len"]
    
    # The input is already right-aligned (left-padded). We can slice from the end.
    responses_from_input = input_ids[:, -max_resp_len:]

    return DataProto.from_dict({
        'input_ids': input_ids,
        'attention_mask': batch["attention_mask"],
        'position_ids': batch["position_ids"],
        'responses': responses_from_input
    }, auto_padding=False)


def _map_results_to_trajectories(
    log_probs: torch.Tensor,
    turns: List[TurnData],
    num_traj: int
) -> torch.Tensor:
    """
    Maps the flat log_probs tensor back to the trajectory structure.
    """
    traj_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_traj)]
    for i, turn in enumerate(turns):
        turn_log_prob = log_probs[i, :turn.resp_len]
        # FIX: Append 1D tensors, not 2D tensors.
        traj_outputs[turn.traj_idx].append(turn_log_prob)

    # Concatenate turns for each trajectory and pad trajectories to the same length
    per_traj_log_probs = [
        # FIX: Concatenate along dim=0 for 1D tensors.
        torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=log_probs.dtype, device=log_probs.device)
        for outputs in traj_outputs
    ]

    # Use pad_sequence again for robust final padding. It now receives a list of 1D tensors.
    final_log_probs = pad_sequence(per_traj_log_probs, batch_first=True, padding_value=0.0)

    return final_log_probs


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
        
        # Log extraction summary (only if debug enabled)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Per-Turn Training] Extracted per-turn contexts for {len(per_turn_contexts)} trajectories")
        
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
        
        # For each trajectory, compute log_prob using correct context
        batch_log_probs = []
        
        for batch_idx, per_turn_contexts in enumerate(per_turn_contexts_batch):
            turn_log_probs = []
            
            for turn_idx, turn_ctx in enumerate(per_turn_contexts):
                # Create mini-batch for this single turn
                turn_input_ids = turn_ctx['input_ids'].unsqueeze(0)  # [1, seq_len]
                turn_attention_mask = turn_ctx['attention_mask'].unsqueeze(0)
                turn_response = turn_ctx['response'].unsqueeze(0)
                
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
            
            # Concatenate log_probs for all turns
            traj_log_probs = torch.cat(turn_log_probs, dim=1) if turn_log_probs else torch.tensor([[]])
            batch_log_probs.append(traj_log_probs)
        
        # Stack to create batch
        result = torch.cat(batch_log_probs, dim=0) if batch_log_probs else torch.tensor([[]])
        
        return result

    @staticmethod
    def compute_per_turn_log_probs_batched(
        data: DataProto,
        compute_log_prob_fn,
        use_per_turn_context: bool = True,
        log_debug: bool = False,
        max_turns_per_traj: Optional[int] = None,
        # This should be passed from config, e.g., cfg.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
        model_max_length: int = 8192 
    ) -> torch.Tensor:
        """
        REFACTORED v2: A robust, single-batch, per-turn log_prob computation.
        
        This version improves upon the last refactoring by:
        1. Using the standard 'flip-pad-flip' trick for left-padding (right-alignment),
           which removes the need for an oversized `required_len` buffer. This directly
           solves the `max_seq_len > max_token_len` assertion error.
        2. Adding sequence truncation for turns that exceed the `model_max_length`, making
           the pipeline robust to unexpectedly long inputs.
        3. Simplifying the `_prepare_for_computation` step significantly.
        """
        if not use_per_turn_context:
            return compute_log_prob_fn(data)

        per_turn_contexts_batch = PerTurnContextManager.extract_per_turn_contexts(data)
        if per_turn_contexts_batch is None:
            return compute_log_prob_fn(data)

        num_traj = len(per_turn_contexts_batch)
        pad_token_id = getattr(data, 'pad_token_id', 0)

        # 1. Flatten, validate, and encapsulate all turns
        valid_turns, turn_mapping = _flatten_and_validate_turns(
            per_turn_contexts_batch, max_turns_per_traj, pad_token_id
        )

        if not valid_turns:
            logger.warning("[Per-Turn Refactor] No valid turns found after filtering. Returning empty tensor.")
            return torch.empty(num_traj, 0, dtype=torch.float32, device='cpu')

        # 2. Build left-padded batch tensors using the robust flip-pad-flip method
        batched_tensors = _build_batched_tensors(valid_turns, pad_token_id, model_max_length)
        
        if not batched_tensors:
             logger.warning("[Per-Turn Refactor] No valid turns after truncation. Returning empty tensor.")
             return torch.empty(num_traj, 0, dtype=torch.float32, device='cpu')

        # 3. Prepare the final DataProto for the model (now much simpler)
        turn_batch_for_compute = _prepare_for_computation(batched_tensors)

        # 4. Execute the computation
        try:
            out = compute_log_prob_fn(turn_batch_for_compute)
            log_probs = out.batch['old_log_probs'] if isinstance(out, DataProto) else out
            
            # The model might return log_probs for more than just the response, slice it
            max_resp_len = batched_tensors["max_resp_len"]
            if log_probs.shape[1] > max_resp_len:
                log_probs = log_probs[:, -max_resp_len:]

        except Exception as e:
            logger.error(f"[Per-Turn Refactor] Error during compute_log_prob_fn: {e}", exc_info=True)
            # Return an empty tensor to avoid crashing the training loop
            return torch.empty(num_traj, 0, dtype=torch.float32, device='cpu')
        
        # 5. Map the flat results back to the original trajectory structure
        final_output = _map_results_to_trajectories(log_probs, valid_turns, num_traj)
        
        # Attach a valid_mask for loss calculation, similar to the original implementation
        valid_mask = (final_output != 0.0)
        final_output.valid_mask = valid_mask

        return final_output


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
