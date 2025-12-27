"""
Per-Turn + Summary Training Algorithms

This module implements the core algorithms for Per-Turn + Summary training:
1. Separate log_prob computation for summary and action tokens
2. Full context supervision with consistency loss

Mathematical formulations are based on:
- PPO (Schulman et al., 2017)
- GAE (Schulman et al., 2016)
- Self-conditioned observation analysis
"""

import torch
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)
_RESPONSES_TYPES_WARNING_EMITTED = False


def extract_summary_mask(
    responses: torch.Tensor,
    tokenizer,
    summary_start_tokens: Optional[list] = None,
    summary_end_tokens: Optional[list] = None
) -> torch.Tensor:
    """
    Extract binary mask indicating which tokens are part of summaries.
    
    Algorithm:
    1. Identify summary start/end token IDs from tokenizer
    2. For each sequence, scan for summary tags
    3. Set mask[i] = 1 if token i is inside <think_summary> or <information_summary>
    
    Args:
        responses: Token IDs tensor of shape [batch_size, seq_len]
        tokenizer: Tokenizer to encode summary tags
        summary_start_tokens: List of token IDs for summary starts (optional)
        summary_end_tokens: List of token IDs for summary ends (optional)
        
    Returns:
        Binary mask of shape [batch_size, seq_len] where 1 = summary token, 0 = action token
        
    Mathematical definition:
        mask[i, j] = 1 if token j is in a summary region, 0 otherwise
    """
    batch_size, seq_len = responses.shape
    device = responses.device
    
    # Get summary tag token IDs
    if summary_start_tokens is None:
        try:
            think_start = tokenizer.encode("<think_summary>", add_special_tokens=False)[0]
            info_start = tokenizer.encode("<information_summary>", add_special_tokens=False)[0]
            summary_start_tokens = [think_start, info_start]
        except:
            logger.warning("[PerTurnSummary] Could not encode summary start tokens")
            return torch.zeros_like(responses, dtype=torch.bool)
    
    if summary_end_tokens is None:
        try:
            think_end = tokenizer.encode("</think_summary>", add_special_tokens=False)[0]
            info_end = tokenizer.encode("</information_summary>", add_special_tokens=False)[0]
            summary_end_tokens = [think_end, info_end]
        except:
            logger.warning("[PerTurnSummary] Could not encode summary end tokens")
            return torch.zeros_like(responses, dtype=torch.bool)
    
    # Initialize mask
    summary_mask = torch.zeros_like(responses, dtype=torch.bool)
    
    # For each sequence, identify summary regions
    for b in range(batch_size):
        in_summary = False
        for t in range(seq_len):
            token_id = responses[b, t].item()
            
            # Check for summary start
            if token_id in summary_start_tokens:
                in_summary = True
                summary_mask[b, t] = True
                
            # Check for summary end
            elif token_id in summary_end_tokens:
                summary_mask[b, t] = True
                in_summary = False
                
            # Inside summary region
            elif in_summary:
                summary_mask[b, t] = True
    
    return summary_mask


def compute_separated_log_probs(
    log_probs: torch.Tensor,
    summary_mask: torch.Tensor,
    response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate log probabilities into summary and action components.
    
    Algorithm:
    1. Apply summary_mask to extract summary log_probs
    2. Apply inverse mask to extract action log_probs
    3. Respect response_mask to exclude padding
    
    Args:
        log_probs: Log probabilities of shape [batch_size, seq_len]
        summary_mask: Binary mask of shape [batch_size, seq_len] (1 = summary)
        response_mask: Binary mask of shape [batch_size, seq_len] (1 = valid token)
        
    Returns:
        (summary_log_probs, action_log_probs): Tuple of tensors with same shape as input
        
    Mathematical definition:
        summary_log_probs[i, j] = log_probs[i, j] if summary_mask[i, j] == 1 else 0
        action_log_probs[i, j] = log_probs[i, j] if summary_mask[i, j] == 0 else 0
        Both respect response_mask (padding tokens are 0)
    """
    # Validate shapes
    assert log_probs.shape == summary_mask.shape == response_mask.shape, \
        f"Shape mismatch: {log_probs.shape} vs {summary_mask.shape} vs {response_mask.shape}"
    
    # Apply masks
    summary_log_probs = log_probs * summary_mask.float() * response_mask.float()
    action_log_probs = log_probs * (~summary_mask).float() * response_mask.float()
    
    return summary_log_probs, action_log_probs



def compute_consistency_loss_from_log_probs(
    log_probs_compressed: torch.Tensor,  # [batch, seq_len] - log_probs for actual tokens under compressed context
    log_probs_full: torch.Tensor,       # [batch, seq_len] - log_probs for same tokens under full context
    response_mask: torch.Tensor,        # [batch, seq_len]
) -> torch.Tensor:
    """
    Compute consistency loss using log_probs difference (MEMORY-EFFICIENT VERSION).
    
    This is similar to PPO's KL approximation: KL(P||Q) ≈ log_probs_P - log_probs_Q
    for the actual generated tokens.
    
    Mathematical formulation:
        For each token position i with actual token t_i:
            log_prob_comp[i] = log P_compressed(t_i | context_compressed)
            log_prob_full[i] = log P_full(t_i | context_full)
            
            # Approximate KL divergence using log_probs difference
            # KL(P_compressed || P_full) ≈ mean(log_prob_comp - log_prob_full)
        
        L_consistency = (1/|valid_tokens|) * Σ_i mask[i] * (log_prob_comp[i] - log_prob_full[i])
    
    Args:
        log_probs_compressed: Log probabilities under compressed context [batch, seq_len]
        log_probs_full: Log probabilities under full context [batch, seq_len]
        response_mask: Valid token mask [batch, seq_len] (1 for valid, 0 for padding)
    
    Returns:
        consistency_loss: Scalar consistency loss (can be negative, but typically positive)
    
    Notes:
        - This is an approximation: full KL requires summing over all vocab tokens
        - However, it's much more memory-efficient (O(batch*seq) vs O(batch*seq*vocab))
        - Similar to how PPO approximates KL using log_probs difference
        - Positive values indicate compressed context assigns lower probability to tokens
    """
    device = log_probs_compressed.device
    dtype = log_probs_compressed.dtype
    
    # Validate inputs
    if log_probs_compressed.shape != log_probs_full.shape:
        raise ValueError(f"Shape mismatch: compressed={log_probs_compressed.shape}, full={log_probs_full.shape}")
    if log_probs_compressed.shape != response_mask.shape:
        raise ValueError(f"Shape mismatch: log_probs={log_probs_compressed.shape}, mask={response_mask.shape}")
    
    # Compute log_probs difference (approximate KL divergence)
    log_prob_diff = log_probs_compressed - log_probs_full  # [batch, seq_len]
    
    # Apply mask
    mask = response_mask.float()
    valid_tokens = mask.sum()
    if valid_tokens <= 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    # Average over valid tokens
    consistency_loss = (log_prob_diff * mask).sum() / valid_tokens
    
    return consistency_loss


def compute_per_turn_summary_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    summary_mask: torch.Tensor,
    response_mask: torch.Tensor,
    consistency_loss: Optional[torch.Tensor] = None,
    summary_loss_weight: float = 0.3,
    action_loss_weight: float = 0.7,
    consistency_loss_weight: float = 0.1,
    clip_ratio: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Compute Per-Turn + Summary training loss with all three mechanisms.
    
    This is the core algorithm that implements:
    1. Separated log_prob computation (summary vs action)
    2. Summary-aware advantage estimation
    3. Full context supervision
    
    Args:
        log_probs: New policy log probs, shape [batch, seq_len]
        advantages: Standard advantages from GAE, shape [batch, seq_len]
        old_log_probs: Old policy log probs, shape [batch, seq_len]
        summary_mask: Binary mask for summary tokens, shape [batch, seq_len]
        response_mask: Binary mask for valid tokens, shape [batch, seq_len]
        consistency_loss: Optional pre-computed consistency loss scalar (log-prob based)
        summary_loss_weight: Weight λ for summary loss (default: 0.3)
        action_loss_weight: Weight (1-λ) for action loss (default: 0.7)
        consistency_loss_weight: Weight for consistency loss (default: 0.0)
        clip_ratio: PPO clip ratio ε (default: 0.2)
        
    Returns:
        Dictionary with loss components:
        - 'total_loss': Combined loss
        - 'summary_loss': PPO loss for summary tokens
        - 'action_loss': PPO loss for action tokens
        - 'consistency_loss': Full context supervision loss (if available)
        - 'summary_ratio_mean': Mean policy ratio for summary tokens
        - 'action_ratio_mean': Mean policy ratio for action tokens
        
    Mathematical formulation:
        1. Separate log probs:
           log_probs_s = log_probs * summary_mask
           log_probs_a = log_probs * (1 - summary_mask)
           
        2. Compute advantages:
           A_s = advantages * summary_mask.float() * response_mask.float()
           A_a = advantages * (~summary_mask).float() * response_mask.float()
           
        3. Compute PPO loss:
           ratio_s = exp(log_probs_s - old_log_probs_s)
           ratio_a = exp(log_probs_a - old_log_probs_a)
           
           L_clip_s = -min(ratio_s * A_s, clip(ratio_s, 1-ε, 1+ε) * A_s)
           L_clip_a = -min(ratio_a * A_a, clip(ratio_a, 1-ε, 1+ε) * A_a)
           
        4. Combine losses:
           L_total = λ * L_clip_s + (1-λ) * L_clip_a + β * L_consistency
    """
    device = log_probs.device
    summary_mask = summary_mask.bool()
    response_mask_bool = response_mask.bool()
    response_mask = response_mask_bool.float()
    
    # Step 1: Separate log probs
    summary_log_probs, action_log_probs = compute_separated_log_probs(
        log_probs, summary_mask, response_mask
    )
    old_summary_log_probs, old_action_log_probs = compute_separated_log_probs(
        old_log_probs, summary_mask, response_mask
    )
    
    # Step 2: Compute advantages
    summary_advantages = advantages * summary_mask.float() * response_mask.float()
    action_advantages = advantages * (~summary_mask).float() * response_mask.float()
    
    # Step 3: Compute policy ratios
    # ratio = exp(new_log_prob - old_log_prob) = P_new / P_old
    
    # For summary tokens
    summary_ratio = torch.exp(summary_log_probs - old_summary_log_probs)
    # Clamp ratio to prevent numerical issues
    summary_ratio = torch.clamp(summary_ratio, 0.0, 10.0)
    
    # For action tokens
    action_ratio = torch.exp(action_log_probs - old_action_log_probs)
    action_ratio = torch.clamp(action_ratio, 0.0, 10.0)
    
    # Step 4: Compute PPO clipped loss
    # L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    
    # Summary loss
    summary_ratio_clipped = torch.clamp(summary_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    summary_loss_unclipped = summary_ratio * summary_advantages
    summary_loss_clipped = summary_ratio_clipped * summary_advantages
    summary_loss_per_token = -torch.min(summary_loss_unclipped, summary_loss_clipped)
    
    # Average over valid summary tokens
    summary_token_mask = summary_mask & response_mask_bool
    summary_token_count_tensor = summary_token_mask.sum()
    summary_loss = (summary_loss_per_token * summary_token_mask.float()).sum() / torch.clamp(summary_token_count_tensor, min=1)
    
    # Action loss
    action_ratio_clipped = torch.clamp(action_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    action_loss_unclipped = action_ratio * action_advantages
    action_loss_clipped = action_ratio_clipped * action_advantages
    action_loss_per_token = -torch.min(action_loss_unclipped, action_loss_clipped)
    
    # Average over valid action tokens
    action_token_mask = (~summary_mask) & response_mask_bool
    action_token_count_tensor = action_token_mask.sum()
    action_loss = (action_loss_per_token * action_token_mask.float()).sum() / torch.clamp(action_token_count_tensor, min=1)
    
    # Step 5: Inject externally-computed consistency loss (if any)
    if consistency_loss is None:
        consistency_loss = torch.tensor(0.0, device=device, dtype=log_probs.dtype)
    else:
        consistency_loss = consistency_loss.to(device=device, dtype=log_probs.dtype)
    
    # Step 6: Combine losses
    # L_total = λ * L_summary + (1-λ) * L_action + β * L_consistency
    total_loss = (
        summary_loss_weight * summary_loss +
        action_loss_weight * action_loss +
        consistency_loss_weight * consistency_loss
    )
    
    # Compute statistics for logging
    with torch.no_grad():
        summary_ratio_mean = (summary_ratio * summary_token_mask.float()).sum() / torch.clamp(summary_token_count_tensor, min=1)
        action_ratio_mean = (action_ratio * action_token_mask.float()).sum() / torch.clamp(action_token_count_tensor, min=1)
    
    return {
        'total_loss': total_loss,
        'summary_loss': summary_loss,
        'action_loss': action_loss,
        'consistency_loss': consistency_loss,
        'summary_ratio_mean': summary_ratio_mean,
        'action_ratio_mean': action_ratio_mean,
        'summary_token_count': int(summary_token_count_tensor.detach().item()),
        'action_token_count': int(action_token_count_tensor.detach().item()),
    }


def extract_summary_mask_from_responses_types(
    responses_types: torch.Tensor,
    enable_debug: bool = False,
) -> torch.Tensor:
    """
    Extract summary mask directly from responses_types (optimized version).
    
    This avoids the need to re-encode and scan token IDs, as responses_types
    is already computed during rollout generation.
    
    Args:
        responses_types: Token-level action types [batch, seq_len]
            - 0: think
            - 1: search
            - 2: answer
            - 3: information (environment token, excluded)
            - 4: information_summary (summary token)
            - 5: think_summary (summary token)
        enable_debug: Whether to log debug information
    
    Returns:
        Binary mask [batch, seq_len] where 1 = summary token, 0 = action token
    """
    global _RESPONSES_TYPES_WARNING_EMITTED
    # Summary tokens are: information_summary (4) and think_summary (5)
    summary_mask = (responses_types == 4) | (responses_types == 5)
    summary_mask = summary_mask.bool()
    
    # Debug logging
    if enable_debug:
        batch_size, seq_len = responses_types.shape
        total_tokens = batch_size * seq_len
        summary_tokens = summary_mask.sum().item()
        action_tokens = total_tokens - summary_tokens
        
        # Count by type
        info_summary_count = (responses_types == 4).sum().item()
        think_summary_count = (responses_types == 5).sum().item()
        think_count = (responses_types == 0).sum().item()
        search_count = (responses_types == 1).sum().item()
        answer_count = (responses_types == 2).sum().item()
        info_count = (responses_types == 3).sum().item()
        
        logger.info(f"[SummaryMask] Extracted summary mask from responses_types:")
        logger.info(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Summary tokens: {summary_tokens} ({summary_tokens/total_tokens*100:.1f}%)")
        logger.info(f"  Action tokens: {action_tokens} ({action_tokens/total_tokens*100:.1f}%)")
        logger.info(f"  Type breakdown:")
        logger.info(f"    - information_summary (4): {info_summary_count} tokens")
        logger.info(f"    - think_summary (5): {think_summary_count} tokens")
        logger.info(f"    - think (0): {think_count} tokens")
        logger.info(f"    - search (1): {search_count} tokens")
        logger.info(f"    - answer (2): {answer_count} tokens")
        logger.info(f"    - information (3): {info_count} tokens")
        
        # Sample-level statistics
        for b in range(min(batch_size, 3)):  # Log first 3 samples
            sample_summary = summary_mask[b].sum().item()
            sample_total = seq_len
            sample_types = responses_types[b].unique().tolist()
            logger.info(f"  Sample {b}: {sample_summary}/{sample_total} summary tokens ({sample_summary/sample_total*100:.1f}%), "
                       f"types present: {sample_types}")
            # Log actual token IDs for debugging summary mask
            try:
                sample_tokens = responses[b]
                summary_tokens_ids = sample_tokens[summary_mask[b]]
                action_tokens_ids = sample_tokens[~summary_mask[b]]
                logger.info(f"    Sample {b} summary_token_ids: {summary_tokens_ids.tolist()[:50]}")
                logger.info(f"    Sample {b} action_token_ids: {action_tokens_ids.tolist()[:50]}")
            except Exception as e:
                logger.debug(f"    Sample {b} token logging failed: {e}")
    
    if enable_debug:
        summary_tokens = int(summary_mask.sum().item())
        total_tokens = summary_mask.numel()
        print(
            f"[PerTurnSummary][MaskCheck] summary_tokens={summary_tokens}/{total_tokens} "
            f"({summary_tokens/total_tokens*100:.2f}%)"
        )

    return summary_mask


def compute_per_turn_summary_loss_wrapper(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor,
    tokenizer=None,  # Deprecated: kept for backward compatibility (only used for fallback summary mask extraction)
    config=None,
    per_turn_valid_mask: Optional[torch.Tensor] = None,
    consistency_loss: Optional[torch.Tensor] = None,  # Pre-computed consistency loss (scalar)
    responses_types: Optional[torch.Tensor] = None,  # Preferred: use this instead of tokenizer
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Wrapper for Per-Turn + Summary loss computation.
    
    This function integrates the Per-Turn + Summary algorithms into the existing
    PPO training pipeline. It handles:
    1. Summary mask extraction (optimized: uses responses_types if available)
    2. Per-Turn + Summary loss computation
    3. Metrics extraction for logging
    
    Args:
        old_log_prob: Old policy log probabilities, shape [batch, seq_len]
        log_prob: New policy log probabilities, shape [batch, seq_len]
        advantages: Advantage estimates, shape [batch, seq_len]
        response_mask: Binary mask for valid tokens, shape [batch, seq_len]
        responses: Token IDs, shape [batch, seq_len]
        tokenizer: Tokenizer for summary tag detection (deprecated, use responses_types instead)
        config: Configuration object with algorithm parameters
        per_turn_valid_mask: Optional binary mask indicating which tokens belong to real turns
        consistency_loss: Pre-computed consistency loss (scalar tensor)
        responses_types: Token-level action types [batch, seq_len] (preferred method)
            - 0: think, 1: search, 2: answer, 3: information, 4: information_summary, 5: think_summary
        
    Returns:
        pg_loss: Policy gradient loss (scalar)
        metrics: Dictionary of metrics for logging
            - 'pg_loss': Policy gradient loss value
            - 'summary_loss': Summary-specific loss
            - 'action_loss': Action-specific loss
            - 'consistency_loss': Consistency loss (if enabled)
            - 'summary_ratio_mean': Mean policy ratio for summary tokens
            - 'action_ratio_mean': Mean policy ratio for action tokens
            - 'pg_clipfrac': Fraction of clipped ratios (approximated)
            - 'ppo_kl': Approximate KL divergence (computed from log ratios)
    """
    # Check if debug logging is enabled
    enable_debug = True
    
    global _RESPONSES_TYPES_WARNING_EMITTED
    # Extract summary mask - prefer responses_types (optimized) over tokenizer (legacy)
    if responses_types is not None:
        # OPTIMIZED: Use responses_types directly (computed during rollout)
        try:
            logger.info(f"[PerTurnSummary] Extracting summary mask from responses_types (optimized method)")
            summary_mask = extract_summary_mask_from_responses_types(responses_types, enable_debug=enable_debug)
            logger.info(f"[PerTurnSummary] Successfully extracted summary mask using responses_types")
        except Exception as e:
            logger.warning(f"[PerTurnSummary] Failed to extract summary mask from responses_types: {e}. Falling back to tokenizer.")
            import traceback
            logger.warning(f"[PerTurnSummary] Traceback: {traceback.format_exc()}")
            summary_mask = None
    else:
        summary_mask = None
        if not _RESPONSES_TYPES_WARNING_EMITTED:
            logger.warning(f"[PerTurnSummary] responses_types not provided, will try tokenizer fallback")
            _RESPONSES_TYPES_WARNING_EMITTED = True
    
    # Fallback to tokenizer-based extraction if responses_types not available
    if summary_mask is None and tokenizer is not None:
        try:
            logger.info(f"[PerTurnSummary] Falling back to tokenizer-based extraction (legacy method)")
            summary_mask = extract_summary_mask(responses, tokenizer)
            logger.info(f"[PerTurnSummary] Successfully extracted summary mask using tokenizer")
            if enable_debug:
                tokens = int(summary_mask.sum().item())
                total = summary_mask.numel()
                print(
                    f"[PerTurnSummary][TokenizerMask] summary_tokens={tokens}/{total} "
                    f"({tokens/total*100:.2f}%)"
                )
        except Exception as e:
            logger.warning(f"[PerTurnSummary] Failed to extract summary mask from tokenizer: {e}. Using all-action mask.")
            import traceback
            logger.warning(f"[PerTurnSummary] Traceback: {traceback.format_exc()}")
            summary_mask = torch.zeros_like(responses, dtype=torch.bool)
    elif summary_mask is None:
        # No responses_types and no tokenizer - fallback to all-action mask
        logger.warning("[PerTurnSummary] No responses_types or tokenizer available. Using all-action mask.")
        summary_mask = torch.zeros_like(responses, dtype=torch.bool)
    
    # Debug: Validate summary mask
    if True:
        # Check alignment with response_mask
        if response_mask.shape == summary_mask.shape:
            valid_summary = (summary_mask & response_mask.bool()).sum().item()
            valid_action = ((~summary_mask) & response_mask.bool()).sum().item()
            valid_total = response_mask.sum().item()
            logger.info(f"  After applying response_mask:")
            logger.info(f"    Valid summary tokens: {valid_summary}/{valid_total} ({valid_summary/valid_total*100:.1f}%)" if valid_total > 0 else "    Valid summary tokens: 0/0")
            logger.info(f"    Valid action tokens: {valid_action}/{valid_total} ({valid_action/valid_total*100:.1f}%)" if valid_total > 0 else "    Valid action tokens: 0/0")
        else:
            logger.warning(f"  Shape mismatch: summary_mask {summary_mask.shape} vs response_mask {response_mask.shape}")
        if tokenizer is not None and summary_mask.size(0) > 0:
            sample_idx = 0
            sample_tokens = responses[sample_idx]
            sample_mask = summary_mask[sample_idx]
            summary_token_ids = sample_tokens[sample_mask]
            action_token_ids = sample_tokens[~sample_mask]
            try:
                summary_text = tokenizer.decode(summary_token_ids.tolist(), skip_special_tokens=False)
                action_text = tokenizer.decode(action_token_ids.tolist(), skip_special_tokens=False)
                logger.info(f"[PerTurnSummary] summary text: {summary_text}")
                logger.info(f"[PerTurnSummary] action text: {action_text}")
            except Exception as e:
                logger.warning(f"[PerTurnSummary] Failed to decode sample {sample_idx}: {e}")

    # Align optional masks to current response window
    def _align_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.shape == response_mask.shape:
            return tensor
        return tensor[:, -response_mask.shape[1]:]

    if per_turn_valid_mask is not None:
        per_turn_valid_mask = _align_optional(per_turn_valid_mask).to(device=response_mask.device, dtype=response_mask.dtype)
        response_mask = response_mask * per_turn_valid_mask
    
    # Compute Per-Turn + Summary loss
    loss_dict = compute_per_turn_summary_loss(
        log_probs=log_prob,
        advantages=advantages,
        old_log_probs=old_log_prob,
        summary_mask=summary_mask,
        response_mask=response_mask,
        consistency_loss=consistency_loss,
        summary_loss_weight=config.get('summary_loss_weight', 0.3),
        action_loss_weight=config.get('action_loss_weight', 0.7),
        consistency_loss_weight=config.get('consistency_loss_weight', 0.1),
        clip_ratio=config.get('clip_ratio', 0.2),
    )
    
    pg_loss = loss_dict['total_loss']
    
    # Compute additional metrics for compatibility with standard PPO logging
    with torch.no_grad():
        # Approximate KL divergence: KL(π_old || π_new) ≈ mean(log(π_old/π_new))
        log_ratio = old_log_prob - log_prob
        valid_mask = response_mask.bool()
        ppo_kl = (log_ratio * response_mask).sum() / max(response_mask.sum(), 1)
        
        # Approximate clipfrac: fraction of ratios outside [1-ε, 1+ε]
        ratio = torch.exp(log_prob - old_log_prob)
        clip_ratio_val = config.get('clip_ratio', 0.2)
        clipped = ((ratio < 1 - clip_ratio_val) | (ratio > 1 + clip_ratio_val)) & valid_mask
        pg_clipfrac = clipped.sum().float() / max(valid_mask.sum(), 1)
    
    metrics = {
        'pg_loss': pg_loss.detach().item(),
        'summary_loss': loss_dict['summary_loss'].detach().item(),
        'action_loss': loss_dict['action_loss'].detach().item(),
        'consistency_loss': loss_dict['consistency_loss'].detach().item(),
        'summary_ratio_mean': loss_dict['summary_ratio_mean'].detach().item(),
        'action_ratio_mean': loss_dict['action_ratio_mean'].detach().item(),
        'pg_clipfrac': pg_clipfrac.detach().item(),
        'ppo_kl': ppo_kl.detach().item(),
        'summary_token_count': loss_dict['summary_token_count'],
        'action_token_count': loss_dict['action_token_count'],
    }
    
    return pg_loss, metrics
