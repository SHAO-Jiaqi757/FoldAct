# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple, Optional
from contextlib import nullcontext
import numpy as np
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils import torch_dtypes
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.trainer.ppo.per_turn_summary_algos import compute_consistency_loss_from_log_probs
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "align_per_turn_valid_mask"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def align_per_turn_valid_mask(
    valid_mask: torch.Tensor,
    target_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Align (truncate/pad) the per-turn valid_mask to match the current response length.

    Args:
        valid_mask: Tensor of shape [batch, tokens] produced during per-turn log_prob compute.
        target_tokens: Number of tokens in the current response slice.
        device: Device of the response_mask we are aligning to.
        dtype: Desired dtype (matches response_mask dtype to avoid implicit casts).

    Returns:
        Tensor of shape [batch, target_tokens] placed on `device` and cast to `dtype`.
    """
    if valid_mask is None:
        return None

    mask = valid_mask.to(device=device)
    current_tokens = mask.size(1)

    if current_tokens > target_tokens:
        logger.debug(
            "[DP Actor] Truncating per_turn_valid_mask from %d to %d tokens",
            current_tokens,
            target_tokens,
        )
        mask = mask[:, :target_tokens]
    elif current_tokens < target_tokens:
        pad_size = target_tokens - current_tokens
        logger.debug(
            "[DP Actor] Padding per_turn_valid_mask from %d to %d tokens (+%d)",
            current_tokens,
            target_tokens,
            pad_size,
        )
        padding = torch.zeros(mask.size(0), pad_size, dtype=mask.dtype, device=device)
        mask = torch.cat([mask, padding], dim=1)

    mask = mask.to(dtype=dtype)
    valid_ratio = mask.sum().item() / max(mask.numel(), 1)
    logger.info(
        "[DP Actor] per_turn_valid_mask aligned: shape=%s valid_ratio=%.4f",
        tuple(mask.shape),
        valid_ratio,
    )
    return mask


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer  # Store tokenizer for summary mask extraction
        self._warned_missing_responses_types = False
        self._warned_missing_full_context = False

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        assert self.config.dtype in ["float16", "bfloat16", "float32"]
        if self.config.dtype == "float16":
            self.scalar = ShardedGradScaler(growth_interval=400)
        else:
            self.scalar = None

    def _compute_consistency_loss_from_full_context(
        self,
        data: dict,
        log_prob_compressed: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        temperature: float,
    ) -> Optional[torch.Tensor]:
        """
        Compute log-prob consistency loss using injected full-context sequences.
        Returns None if no samples require supervision.
        """
        # Convert TensorDict to dict if needed (for compatibility)
        if not isinstance(data, dict) and hasattr(data, "keys") and hasattr(data, "__getitem__"):
            try:
                # Try to convert TensorDict to dict
                data = {k: data[k] for k in data.keys()}
            except Exception:
                # If conversion fails, keep data as-is (it might be a TensorDict)
                pass
        
        def _safe_get(container, field):
            # Try dict first
            if isinstance(container, dict):
                return container.get(field)
            # Try TensorDict (supports 'in' operator and dict-like access)
            if field in container:
                try:
                    return container[field]
                except (KeyError, TypeError):
                    return None
            # Try get() method (for objects with get method)
            if hasattr(container, "get"):
                try:
                    return container.get(field, None)
                except (KeyError, TypeError):
                    return None
            # Try getattr as last resort
            return getattr(container, field, None)

        # ROBUST SOLUTION: Only use pre-computed full context log_probs
        # This completely avoids activation offload state issues by never doing extra forward passes during training
        full_context_log_probs = _safe_get(data, "full_context_log_probs")
        
        if full_context_log_probs is not None:
            # Use pre-computed log_probs (computed in trainer using compute_log_prob)
            # This is the robust solution that completely avoids activation offload state issues
            logger.debug(f"[DP Actor] Using pre-computed full_context_log_probs: shape={full_context_log_probs.shape}")
            
            # Extract valid rows (samples that have full context)
            full_context_indices = _safe_get(data, "full_context_indices")
            if full_context_indices is not None:
                # Use indices to identify which samples have full context
                valid_rows = full_context_indices >= 0
            else:
                # Fallback: check if log_probs are non-zero
                valid_rows = full_context_log_probs.abs().sum(dim=1) > 0
            
            if not torch.any(valid_rows):
                return None
            
            selected_log_prob = log_prob_compressed[valid_rows]
            selected_log_prob_full = full_context_log_probs[valid_rows].to(selected_log_prob.device)
            selected_response_mask = response_mask[valid_rows]
            
            # Align lengths if needed
            min_len = min(selected_log_prob.size(1), selected_log_prob_full.size(1), selected_response_mask.size(1))
            if min_len <= 0:
                return None
            
            selected_log_prob = selected_log_prob[:, :min_len]
            selected_log_prob_full = selected_log_prob_full[:, :min_len]
            selected_response_mask = selected_response_mask[:, :min_len]
            
            consistency_loss = compute_consistency_loss_from_log_probs(
                log_probs_compressed=selected_log_prob,
                log_probs_full=selected_log_prob_full,
                response_mask=selected_response_mask,
            )
            
            logger.info(
                "[Consistency] Using pre-computed full-context log_probs for %d samples "
                "(tokens=%d) -> loss=%.6f",
                valid_rows.sum().item(),
                int(selected_response_mask.sum().item()),
                float(consistency_loss.detach().cpu().item()),
            )
            return consistency_loss
        
        # CRITICAL: Do NOT fallback to computing log_probs during training
        # This would break activation offload state machine (offloaded_group_count exceeds layer_window_map keys)
        # Instead, log error and return None - the trainer should always provide pre-computed log_probs
        if self.config.get("use_full_context_supervision", False):
            if not self._warned_missing_full_context:
                logger.error(
                    "[DP Actor] CRITICAL: full_context_log_probs not found! "
                    "This should never happen if trainer correctly injects pre-computed log_probs. "
                    "Consistency loss will be skipped to avoid activation offload state corruption.\n"
                    "Possible causes:\n"
                    "  1. Trainer's _inject_full_context_batch was not called\n"
                    "  2. full_log_probs was None when injecting (check trainer logs)\n"
                    "  3. Data was lost during transfer from trainer to dp_actor\n"
                    "  4. full_context_consistency_interval not reached\n"
                    "  5. Event ledgers missing in batch\n"
                    "  6. No indices sampled for full context computation\n"
                    "To debug: Check logs for [ConsistencyDebug] messages in trainer."
                )
                self._warned_missing_full_context = True
            else:
                # Log periodically to track when this happens
                import random
                if random.random() < 0.05:  # Log 5% of the time
                    logger.warning(
                        "[DP Actor] full_context_log_probs still missing! "
                        "This indicates a bug in the trainer's injection logic. "
                        "Check [ConsistencyDebug] logs in trainer."
                    )
        
        # Return None instead of attempting to compute (which would break activation offload)
        return None


    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        
        # DEBUG: Log input shapes and response_length
        input_ids = micro_batch["input_ids"]
        responses = micro_batch["responses"]
        attention_mask = micro_batch["attention_mask"]
        
        if logger.isEnabledFor(logging.WARNING):
            # Log for first micro_batch only (to avoid spam)
            if not hasattr(self, '_debug_logged_forward'):
                self._debug_logged_forward = True
                logger.warning(
                    f"[DP Actor] DEBUG _forward_micro_batch: "
                    f"input_ids.shape={input_ids.shape}, "
                    f"responses.shape={responses.shape}, "
                    f"response_length={response_length}, "
                    f"attention_mask non-zero={(attention_mask.sum(dim=1) > 0).sum().item()}/{input_ids.size(0)}"
                )
                
                # Check alignment for first sample
                if input_ids.size(0) > 0:
                    seq_len = input_ids.size(1)
                    resp_from_input = input_ids[0, -response_length:]
                    resp_given = responses[0]
                    match = torch.equal(resp_from_input, resp_given)
                    resp_from_input_non_zero = (resp_from_input != 0).sum().item()
                    resp_given_non_zero = (resp_given != 0).sum().item()
                    logger.warning(
                        f"[DP Actor] DEBUG Alignment check (sample 0): "
                        f"input_ids[-{response_length}:] vs responses match={match}, "
                        f"resp_from_input non-zero={resp_from_input_non_zero}/{response_length}, "
                        f"resp_given non-zero={resp_given_non_zero}/{response_length}"
                    )
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        from verl.utils.torch_dtypes import PrecisionType
        torch_dtype = PrecisionType.to_dtype(self.config.dtype)
        with torch.autocast(device_type=self.device_name, dtype=torch_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]

            # DIAGNOSIS: Check for all-zero attention masks
            if logger.isEnabledFor(logging.WARNING):
                zero_mask_rows = (attention_mask.sum(dim=1) == 0)
                if zero_mask_rows.any():
                    num_zero_rows = zero_mask_rows.sum().item()
                    logger.error(
                        f"[DP Actor] CRITICAL: Found {num_zero_rows}/{batch_size} rows with all-zero attention_mask in micro_batch! "
                        f"This will result in zero log_probs."
                    )
                    
                    # Show which rows and their input_ids/responses
                    zero_row_indices = zero_mask_rows.nonzero(as_tuple=True)[0]
                    logger.error(f"[DP Actor] Zero attention_mask row indices: {zero_row_indices.tolist()}")
                    
                    for idx in zero_row_indices[:3]:  # Show first 3
                        row_idx = idx.item()
                        input_ids_nonzero = (input_ids[row_idx] != 0).sum().item()
                        resp_nonzero = (responses[row_idx] != 0).sum().item()
                        logger.error(f"[DP Actor] Row {row_idx}: input_ids non-zero={input_ids_nonzero}/{input_ids.shape[1]}, "
                                   f"responses non-zero={resp_nonzero}/{responses.shape[1]}")
                        
                        if input_ids_nonzero > 0:
                            # Show actual token IDs
                            nonzero_input_ids = input_ids[row_idx][input_ids[row_idx] != 0]
                            logger.error(f"[DP Actor] Row {row_idx} input_ids: {nonzero_input_ids.tolist()}")
                            logger.error(f"[DP Actor] Row {row_idx} input_ids: {nonzero_input_ids.tolist()}")
                        else:
                            logger.error(f"[DP Actor] Row {row_idx} input_ids is ENTIRELY ZERO!")
                        
                        if resp_nonzero > 0:
                            nonzero_resp = responses[row_idx][responses[row_idx] != 0]
                            logger.error(f"[DP Actor] Row {row_idx} responses: {nonzero_resp.tolist()}")
                        else:
                            logger.error(f"[DP Actor] Row {row_idx} responses is ENTIRELY ZERO!")

            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                total_nnz = input_ids_rmpad.size(0)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                
                # Basic validation logging
                attention_mask_sum = attention_mask.sum().item()
                seq_valid_lengths = attention_mask.sum(dim=1)
                num_zero_seq = (seq_valid_lengths == 0).sum().item()
                if total_nnz == 0:
                    logger.error(f"[DP Actor] CRITICAL: total_nnz is ZERO! batch_size={batch_size}, seqlen={seqlen}")
                if attention_mask_sum != total_nnz:
                    logger.warning(f"[DP Actor] attention_mask_sum ({attention_mask_sum}) != total_nnz ({total_nnz})")

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
                
                # CRITICAL FIX: Reset position_ids to start from 0 for each sequence
                # This is necessary because Flash Attention varlen mode uses position_ids == 0 to detect sequence boundaries
                # Calculate sequence boundaries from attention_mask
                seq_valid_lengths = attention_mask.sum(dim=1)  # [batch_size]
                cu_seqlens = torch.cat([
                    torch.tensor([0], device=position_ids_rmpad.device, dtype=torch.int32),
                    torch.cumsum(seq_valid_lengths, dim=0, dtype=torch.int32)
                ])  # [batch_size + 1]
                
                # Reset position_ids: for each sequence, subtract the starting position_id
                if position_ids_rmpad.dim() == 2:
                    # Standard case: (1, total_nnz)
                    pos_ids_flat = position_ids_rmpad.flatten()
                    for i in range(batch_size):
                        start_idx = cu_seqlens[i].item()
                        end_idx = cu_seqlens[i+1].item()
                        if end_idx > start_idx:
                            # Get the starting position_id for this sequence
                            seq_start_pos = pos_ids_flat[start_idx].item()
                            # Reset to start from 0
                            pos_ids_flat[start_idx:end_idx] -= seq_start_pos
                    position_ids_rmpad = pos_ids_flat.view(position_ids_rmpad.shape)
                elif position_ids_rmpad.dim() == 3:
                    # Qwen2VL mrope case: (3, 1, total_nnz)
                    for c in range(position_ids_rmpad.size(0)):
                        pos_ids_flat = position_ids_rmpad[c, 0, :].flatten()
                        for i in range(batch_size):
                            start_idx = cu_seqlens[i].item()
                            end_idx = cu_seqlens[i+1].item()
                            if end_idx > start_idx:
                                # Get the starting position_id for this sequence
                                seq_start_pos = pos_ids_flat[start_idx].item()
                                # Reset to start from 0
                                pos_ids_flat[start_idx:end_idx] -= seq_start_pos
                        position_ids_rmpad[c, 0, :] = pos_ids_flat
                
                # Validate position_ids after reset - only log if there's an issue
                if position_ids_rmpad.numel() > 0:
                    pos_ids_flat = position_ids_rmpad.flatten()
                    pos_ids_zero_count = (pos_ids_flat == 0).sum().item()
                    
                    if pos_ids_zero_count == 0:
                        logger.error(f"[DP Actor] CRITICAL: No position_ids == 0 after reset! batch_size={batch_size}, shape={position_ids_rmpad.shape}")
                    elif pos_ids_zero_count < batch_size:
                        logger.warning(f"[DP Actor] position_ids zero_count ({pos_ids_zero_count}) < batch_size ({batch_size})")
                    else:
                        logger.debug(f"[DP Actor] position_ids reset OK: zero_count={pos_ids_zero_count}, batch_size={batch_size}")
                else:
                    logger.error(f"[DP Actor] CRITICAL: position_ids_rmpad is EMPTY!")

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    logger.debug(f"[DP Actor] After Ulysses SP: pad_size={pad_size}")

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
            # DEBUG: Check log_probs output
            if logger.isEnabledFor(logging.WARNING):
                if not hasattr(self, '_debug_logged_logprobs'):
                    self._debug_logged_logprobs = True
                    log_probs_zero_count = (log_probs == 0.0).sum().item()
                    log_probs_total = log_probs.numel()
                    log_probs_zero_ratio = log_probs_zero_count / log_probs_total if log_probs_total > 0 else 0.0
                    
                    # Distinguish between padding zeros and valid position zeros
                    # Use attention_mask to identify valid positions (non-padding)
                    # Note: attention_mask shape is (batch_size, seq_len), log_probs shape is (batch_size, response_length)
                    # We need to extract the response portion from attention_mask
                    full_attention_mask = micro_batch.get("attention_mask", None)
                    if full_attention_mask is not None:
                        # Extract response portion: last response_length positions
                        if full_attention_mask.size(1) >= response_length:
                            response_mask = full_attention_mask[:, -response_length:]
                            if response_mask.shape == log_probs.shape:
                                valid_positions = response_mask.bool()
                                valid_zero_count = ((log_probs == 0.0) & valid_positions).sum().item()
                                valid_total = valid_positions.sum().item()
                                valid_zero_ratio = valid_zero_count / valid_total if valid_total > 0 else 0.0
                                padding_zero_count = ((log_probs == 0.0) & ~valid_positions).sum().item()
                                padding_total = (~valid_positions).sum().item()
                                
                                logger.warning(
                                    f"[DP Actor] DEBUG log_probs output: "
                                    f"shape={log_probs.shape}, "
                                    f"total_zero={log_probs_zero_count}/{log_probs_total} ({log_probs_zero_ratio*100:.1f}%), "
                                    f"valid_zero={valid_zero_count}/{valid_total} ({valid_zero_ratio*100:.1f}%), "
                                    f"padding_zero={padding_zero_count}/{padding_total}, "
                                    f"mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f}"
                                )
                            else:
                                logger.warning(
                                    f"[DP Actor] DEBUG log_probs output: "
                                    f"shape={log_probs.shape}, "
                                    f"zero_count={log_probs_zero_count}/{log_probs_total} ({log_probs_zero_ratio*100:.1f}%), "
                                    f"mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f} "
                                    f"(response_mask shape mismatch: {response_mask.shape} vs {log_probs.shape})"
                                )
                        else:
                            logger.warning(
                                f"[DP Actor] DEBUG log_probs output: "
                                f"shape={log_probs.shape}, "
                                f"zero_count={log_probs_zero_count}/{log_probs_total} ({log_probs_zero_ratio*100:.1f}%), "
                                f"mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f} "
                                f"(attention_mask seq_len={full_attention_mask.size(1)} < response_length={response_length})"
                            )
                    else:
                        logger.warning(
                            f"[DP Actor] DEBUG log_probs output: "
                            f"shape={log_probs.shape}, "
                            f"zero_count={log_probs_zero_count}/{log_probs_total} ({log_probs_zero_ratio*100:.1f}%), "
                            f"mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f} "
                            f"(no attention_mask in micro_batch)"
                        )
                    
                    # Check first sample
                    if log_probs.size(0) > 0:
                        sample_log_prob = log_probs[0]
                        sample_zero_count = (sample_log_prob == 0.0).sum().item()
                        sample_zero_ratio = sample_zero_count / sample_log_prob.numel()
                        
                        # Check valid positions for first sample
                        if full_attention_mask is not None and full_attention_mask.size(1) >= response_length:
                            sample_response_mask = full_attention_mask[0, -response_length:]
                            if sample_response_mask.shape == sample_log_prob.shape:
                                sample_valid_positions = sample_response_mask.bool()
                                sample_valid_zero_count = ((sample_log_prob == 0.0) & sample_valid_positions).sum().item()
                                sample_valid_total = sample_valid_positions.sum().item()
                                sample_valid_zero_ratio = sample_valid_zero_count / sample_valid_total if sample_valid_total > 0 else 0.0
                                
                                logger.warning(
                                    f"[DP Actor] DEBUG Sample 0 log_prob: "
                                    f"total_zero_ratio={sample_zero_ratio*100:.1f}%, "
                                    f"valid_zero_ratio={sample_valid_zero_ratio*100:.1f}%, "
                                    f"mean={sample_log_prob.mean().item():.6f}, "
                                    f"min={sample_log_prob.min().item():.6f}, max={sample_log_prob.max().item():.6f}"
                                )
                            else:
                                logger.warning(
                                    f"[DP Actor] DEBUG Sample 0 log_prob: "
                                    f"zero_ratio={sample_zero_ratio*100:.1f}%, "
                                    f"mean={sample_log_prob.mean().item():.6f}, "
                                    f"min={sample_log_prob.min().item():.6f}, max={sample_log_prob.max().item():.6f}"
                                )
                        else:
                            logger.warning(
                                f"[DP Actor] DEBUG Sample 0 log_prob: "
                                f"zero_ratio={sample_zero_ratio*100:.1f}%, "
                                f"mean={sample_log_prob.mean().item():.6f}, "
                                f"min={sample_log_prob.min().item():.6f}, max={sample_log_prob.max().item():.6f}"
                            )

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if self.scalar is not None:
            self.scalar.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if self.scalar is not None:
            self.scalar.step(self.actor_optimizer)
            self.scalar.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # CRITICAL: Preserve responses_types if available (needed for summary mask computation)
        if "responses_types" in data.batch:
            select_keys.append("responses_types")
        # CRITICAL: Preserve full_context_input_ids and related fields if available
        # These are needed for consistency loss computation
        for key in ["full_context_input_ids", "full_context_attention_mask", "full_context_indices", "full_context_log_probs"]:
            if key in data.batch:
                select_keys.append(key)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        extra_keys = []
        for key in ["consistency_weight"]:
            if key in data.batch.keys():
                extra_keys.append(key)
        select_keys.extend(extra_keys)
        # Add responses_types if available (for optimized summary mask extraction)
        has_responses_types = (
            "responses_types" in data.batch.keys() or
            "responses_types" in data.non_tensor_batch.keys()
        )
        if "responses_types" in data.non_tensor_batch.keys():
            rt_val = data.non_tensor_batch.pop("responses_types")
            if isinstance(rt_val, torch.Tensor):
                pass
            else:
                import numpy as np
                if isinstance(rt_val, np.ndarray):
                    rt_val = torch.from_numpy(rt_val)
                else:
                    rt_val = torch.tensor(rt_val)
            data.batch["responses_types"] = rt_val
        if has_responses_types:
            select_keys.append("responses_types")
        # CRITICAL: Include full-context tensors if trainer injected them
        # This includes full_context_log_probs which is pre-computed to avoid activation offload issues
        for key in ["full_context_input_ids", "full_context_attention_mask", "full_context_indices", "full_context_log_probs"]:
            if key in data.batch.keys():
                select_keys.append(key)
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    device = get_torch_device().current_device()
                    per_turn_valid_mask = None
                    aligned_valid_mask = None
                    if isinstance(data, DataProto):
                        tensor_batch = data.batch.to(device)
                        per_turn_valid_mask = tensor_batch.get("per_turn_valid_mask")
                        data = {**tensor_batch, **data.non_tensor_batch}
                    else:
                        data = data.to(device)  # actor device is cpu when using offload
                        if isinstance(data, dict):
                            per_turn_valid_mask = data.get("per_turn_valid_mask")
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    
                    # DIAGNOSIS: Only log if there's a mismatch (check after actual extraction)
                    if logger.isEnabledFor(logging.WARNING):
                        old_log_prob_shape = old_log_prob.shape
                        if len(old_log_prob_shape) == 0 or (len(old_log_prob_shape) == 2 and old_log_prob_shape[1] != response_length):
                            logger.warning(
                                f"[DP Actor] RESPONSE LENGTH MISMATCH: response_length={response_length}, "
                                f"old_log_prob_shape={old_log_prob_shape}, responses_shape={responses.shape}"
                            )
                    advantages = data["advantages"]
                    
                    # CRITICAL FIX: Apply valid_mask if available (from per-turn training)
                    # This masks out padding positions in old_log_prob
                    if per_turn_valid_mask is None and isinstance(data, dict):
                        per_turn_valid_mask = data.get("per_turn_valid_mask")
                    if per_turn_valid_mask is not None:
                        aligned_valid_mask = align_per_turn_valid_mask(
                            valid_mask=per_turn_valid_mask,
                            target_tokens=response_length,
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        response_mask = response_mask * aligned_valid_mask
                        # Prevent downstream consumers from mistakenly reusing the stale tensor
                        if isinstance(data, dict) and "per_turn_valid_mask" in data:
                            del data["per_turn_valid_mask"]
                    else:
                        aligned_valid_mask = None
                    
                    # Align old_log_prob and advantages to current response_length
                    old_log_prob_original_length = old_log_prob.size(1)
                    if old_log_prob.size(1) != response_length:
                        if old_log_prob.size(1) > response_length:
                            # Truncate if old_log_prob is longer
                            # CRITICAL: This happens when per-turn training computed old_log_prob for all turns
                            # but training uses compressed responses (max_response_length)
                            truncate_size = old_log_prob.size(1) - response_length
                            logger.warning(
                                f"[DP Actor] OLD_LOG_PROB ALIGNMENT: Truncating old_log_prob from {old_log_prob.size(1)} to {response_length} "
                                f"(removing {truncate_size} tokens, batch_idx={batch_idx}, epoch={epoch})"
                            )
                            old_log_prob = old_log_prob[:, :response_length]
                            # Note: response_mask is already aligned to response_length, no need to update
                        else:
                            # Pad with zeros if old_log_prob is shorter
                            pad_size = response_length - old_log_prob.size(1)
                            logger.warning(
                                f"[DP Actor] OLD_LOG_PROB ALIGNMENT: Padding old_log_prob from {old_log_prob.size(1)} to {response_length} "
                                f"(+{pad_size} zeros, batch_idx={batch_idx}, epoch={epoch})"
                            )
                            padding = torch.zeros(old_log_prob.size(0), pad_size, 
                                                dtype=old_log_prob.dtype, device=old_log_prob.device)
                            old_log_prob = torch.cat([old_log_prob, padding], dim=1)
                            
                            # CRITICAL: Update response_mask to mask out the padded positions
                            # The padded positions should not contribute to loss
                            if response_mask.size(1) == response_length:
                                # Mask out the padded positions (set to 0)
                                response_mask[:, old_log_prob_original_length:] = 0
                                logger.warning(
                                    f"[DP Actor] OLD_LOG_PROB ALIGNMENT: Updated response_mask to mask out {pad_size} padded positions"
                                )
                            else:
                                logger.error(
                                    f"[DP Actor] OLD_LOG_PROB ALIGNMENT: response_mask size mismatch! "
                                    f"response_mask.shape={response_mask.shape}, response_length={response_length}"
                                )
                    
                    # Diagnostic: Only log if there's a critical issue (>90% zeros in valid positions)
                    if logger.isEnabledFor(logging.ERROR):
                        valid_zero_count = ((old_log_prob == 0.0) & (response_mask.bool())).sum().item()
                        valid_total = response_mask.sum().item()
                        valid_zero_ratio = valid_zero_count / valid_total if valid_total > 0 else 0.0
                        
                        if valid_zero_ratio > 0.9:
                            logger.error(
                                f"[DP Actor] CRITICAL: {valid_zero_ratio*100:.1f}% of VALID positions have zero old_log_prob! "
                                f"original_length={old_log_prob_original_length}, response_length={response_length}"
                            )
                    
                    if advantages.size(1) != response_length:
                        if advantages.size(1) > response_length:
                            # Truncate if advantages is longer
                            advantages = advantages[:, :response_length]
                        else:
                            # Pad with zeros if advantages is shorter
                            padding = torch.zeros(advantages.size(0), response_length - advantages.size(1), 
                                                dtype=advantages.dtype, device=advantages.device)
                            advantages = torch.cat([advantages, padding], dim=1)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we need logits for consistency loss
                    # Only needed if BOTH use_per_turn_summary AND use_full_context_supervision are enabled
                    use_per_turn_summary = self.config.get('use_per_turn_summary', False)
                    use_full_context_supervision = self.config.get('use_full_context_supervision', False)
                    
                    # Forward pass: get log_probs (no need for logits!)
                    forward_result = self._forward_micro_batch(
                        micro_batch=data, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy)
                    
                    entropy, log_prob = forward_result
                    
                    # Initialize consistency loss (will be computed only if needed)
                    consistency_loss_value = torch.tensor(0.0, device=log_prob.device, dtype=log_prob.dtype)
                    if use_full_context_supervision:
                        full_context_loss = self._compute_consistency_loss_from_full_context(
                            data=data,
                            log_prob_compressed=log_prob,
                            responses=responses,
                            response_mask=response_mask,
                            temperature=temperature,
                        )
                        if full_context_loss is not None:
                            consistency_loss_value = full_context_loss
                    
                
                    # Enable tracing if enable_debug_logs is set (check config, meta_info, or env var)
                    enable_tracing = (
                        self.config.get("enable_debug_logs", False) or
                        data.meta_info.get("enable_debug_logs", False) if hasattr(data, 'meta_info') else False or
                        os.getenv("VERL_ENABLE_PG_TRACE", "false").lower() == "true"
                    )
                    trace_step = epoch * len(dataloader) + batch_idx
                    
                    # use_per_turn_summary already checked above
                    if use_per_turn_summary:
                        # Use Per-Turn + Summary loss
                        from verl.trainer.ppo.per_turn_summary_algos import compute_per_turn_summary_loss_wrapper
                        
                        # Get responses_types from data if available (for optimized summary mask extraction)
                        responses_types = None
                        if isinstance(data, dict) and "responses_types" in data:
                            responses_types = data["responses_types"]
                        elif hasattr(data, 'batch') and 'responses_types' in data.batch:
                            # DataProto case
                            responses_types = data.batch['responses_types']
                        elif 'responses_types' in data:
                            # TensorDict case (data is a TensorDict, not a DataProto)
                            # TensorDict supports 'in' operator and dict-like access
                            responses_types = data['responses_types']
                        if responses_types is not None:
                            if responses_types.device != responses.device:
                                responses_types = responses_types.to(responses.device)
                            # Align to response window if needed
                            if responses_types.shape != responses.shape:
                                if responses_types.size(1) > responses.size(1):
                                    responses_types = responses_types[:, -responses.size(1):]
                                elif responses_types.size(1) < responses.size(1):
                                    # Pad if shorter
                                    pad_size = responses.size(1) - responses_types.size(1)
                                    responses_types = torch.nn.functional.pad(
                                        responses_types, (0, pad_size), value=0
                                    )
                            # Log debug info if enabled
                            enable_debug = (
                                self.config.get("enable_debug_logs", False) or
                                (isinstance(data, dict) and data.get("enable_debug_logs", False))
                            )
                            if enable_debug:
                                unique_types = responses_types.unique().tolist()
                                type_counts = {
                                    t: (responses_types == t).sum().item() 
                                    for t in unique_types
                                }
                                logger.info(f"[DP Actor] Using responses_types for summary mask:")
                                logger.info(f"  Shape: {responses_types.shape}")
                                logger.info(f"  Unique types: {unique_types}")
                                logger.info(f"  Type counts: {type_counts}")
                            else:
                                logger.debug(f"[DP Actor] Using responses_types for summary mask: shape={responses_types.shape}")
                        else:
                            if not self._warned_missing_responses_types:
                                logger.warning(
                                    "[DP Actor] responses_types not found in batch; falling back to tokenizer-based summary mask."
                                )
                                self._warned_missing_responses_types = True
                        
                        pg_loss, loss_metrics = compute_per_turn_summary_loss_wrapper(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            responses=responses,
                            tokenizer=self.tokenizer,
                            config=self.config,
                            per_turn_valid_mask=aligned_valid_mask,
                            consistency_loss=consistency_loss_value,  # Pass pre-computed consistency loss (scalar)
                            responses_types=responses_types,  # Pass pre-computed types for optimization
                        )
                        
                        # Extract metrics for compatibility with standard PPO logging
                        pg_clipfrac = torch.tensor(loss_metrics.get('pg_clipfrac', 0.0), device=pg_loss.device)
                        ppo_kl = torch.tensor(loss_metrics.get('ppo_kl', 0.0), device=pg_loss.device)
                        pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
                        
                        # Log additional Per-Turn + Summary metrics
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(
                                f"[Per-Turn+Summary] summary_loss={loss_metrics['summary_loss']:.4f}, "
                                f"action_loss={loss_metrics['action_loss']:.4f}, "
                                f"consistency_loss={loss_metrics['consistency_loss']:.4f}, "
                                f"summary_tokens={loss_metrics['summary_token_count']}, "
                                f"action_tokens={loss_metrics['action_token_count']}"
                            )
                    else:
                        # Use standard PPO loss
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            enable_tracing=enable_tracing,
                            trace_step=trace_step,
                        )
                        
                        # Auto-enable tracing if pg_loss is suspiciously large (only for standard PPO, not per-turn summary)
                        # CRITICAL: Only check and recompute if using standard PPO loss (not per-turn summary)
                        # Per-turn summary loss already includes all components and should not be recomputed
                        pg_loss_val = pg_loss.detach().item()
                        if not enable_tracing and (abs(pg_loss_val) > 1000 or not torch.isfinite(pg_loss)):
                            logger.warning(f"[PG_LOSS_TRACE] Auto-enabling tracing due to large pg_loss: {pg_loss_val:.6f}")
                            # Recompute with tracing enabled
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                                enable_tracing=True,
                                trace_step=trace_step,
                            )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    
                    # Use scaler.scale() if using mixed precision training
                    if self.scalar is not None:
                        self.scalar.scale(loss).backward()
                    else:
                        loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
