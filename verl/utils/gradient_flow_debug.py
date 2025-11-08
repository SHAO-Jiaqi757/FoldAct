"""
Gradient flow debugging utilities for multi-turn training.

This module provides functions to diagnose gradient flow issues,
particularly in multi-turn training scenarios with GRPO or GAE.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def log_advantage_distribution(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    step: int,
    sample_indices: List[int] = None,
    adv_estimator: str = "unknown",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Log advantage distribution to check if all tokens have the same advantage (GRPO issue).
    
    Args:
        advantages: (bs, seq_len) advantage tensor
        response_mask: (bs, seq_len) mask tensor
        step: current training step
        sample_indices: which samples to analyze (default: first 3)
        adv_estimator: name of advantage estimator (e.g., "GRPO", "GAE")
        verbose: whether to print details
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "step": step,
        "adv_estimator": adv_estimator,
        "all_tokens_same_any": False,
        "samples_with_issue": [],
        "stats": {},
    }
    
    if sample_indices is None:
        sample_indices = list(range(min(3, advantages.shape[0])))
    
    all_tokens_same_count = 0
    all_stats = []
    
    print(f"\n{'='*80}")
    print(f"[GRADIENT FLOW DEBUG] Step {step} - Advantage Distribution Analysis")
    print(f"{'='*80}")
    print(f"Advantage Estimator: {adv_estimator}")
    print(f"Batch size: {advantages.shape[0]}, Sequence length: {advantages.shape[1]}")
    
    for i in sample_indices:
        if i >= advantages.shape[0]:
            continue
        
        valid_mask = response_mask[i] > 0
        valid_advantages = advantages[i][valid_mask]
        
        if len(valid_advantages) == 0:
            if verbose:
                print(f"  ⚠️  Sample {i}: No valid tokens")
            continue
        
        # Check if all tokens have the same advantage
        unique_advantages = valid_advantages.unique()
        num_unique = len(unique_advantages)
        all_tokens_same = num_unique == 1
        
        # Statistics
        stats = {
            "sample_idx": i,
            "num_unique": num_unique,
            "all_tokens_same": all_tokens_same,
            "mean": valid_advantages.mean().item(),
            "std": valid_advantages.std().item(),
            "min": valid_advantages.min().item(),
            "max": valid_advantages.max().item(),
            "num_tokens": len(valid_advantages),
        }
        all_stats.append(stats)
        
        if all_tokens_same:
            all_tokens_same_count += 1
            results["samples_with_issue"].append(i)
            if verbose:
                print(f"\n  ⚠️  Sample {i}: All {stats['num_tokens']} tokens have the SAME advantage")
                print(f"      Advantage value: {unique_advantages[0].item():.6f}")
                if adv_estimator == "GRPO":
                    print(f"      This is expected for GRPO, but may cause credit assignment issues in multi-turn.")
        else:
            if verbose:
                print(f"\n  ✅ Sample {i}: {num_unique} unique advantages")
                print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"      Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                print(f"      Valid tokens: {stats['num_tokens']}")
        
        # Check for very low std (near-identical values)
        if stats['std'] < 1e-6 and not all_tokens_same:
            if verbose:
                print(f"      ⚠️  Very low std ({stats['std']:.2e}) - advantages nearly identical")
    
    results["all_tokens_same_any"] = all_tokens_same_count > 0
    results["stats"] = all_stats
    
    # Summary
    print(f"\n  Summary: {all_tokens_same_count}/{len(sample_indices)} samples have identical advantages")
    if results["all_tokens_same_any"] and adv_estimator == "GRPO":
        print(f"  💡 Recommendation: Consider using GAE for multi-turn training")
    
    return results


def log_reward_distribution(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step: int,
    sample_indices: List[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Log reward distribution to check for reward dilution.
    
    Args:
        token_level_rewards: (bs, seq_len) reward tensor
        response_mask: (bs, seq_len) mask tensor
        step: current training step
        sample_indices: which samples to analyze
        verbose: whether to print details
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "step": step,
        "samples_with_dilution": [],
        "stats": {},
    }
    
    if sample_indices is None:
        sample_indices = list(range(min(3, token_level_rewards.shape[0])))
    
    print(f"\n{'='*80}")
    print(f"[GRADIENT FLOW DEBUG] Step {step} - Reward Distribution Analysis")
    print(f"{'='*80}")
    
    for i in sample_indices:
        if i >= token_level_rewards.shape[0]:
            continue
        
        valid_mask = response_mask[i] > 0
        valid_rewards = token_level_rewards[i][valid_mask]
        
        if len(valid_rewards) == 0:
            if verbose:
                print(f"  ⚠️  Sample {i}: No valid tokens")
            continue
        
        total_reward = valid_rewards.sum().item()
        non_zero_tokens = (valid_rewards > 0).sum().item()
        num_tokens = len(valid_rewards)
        avg_reward_per_token = total_reward / max(num_tokens, 1)
        reward_concentration = non_zero_tokens / max(num_tokens, 1)
        
        stats = {
            "sample_idx": i,
            "total_reward": total_reward,
            "non_zero_tokens": non_zero_tokens,
            "num_tokens": num_tokens,
            "avg_reward_per_token": avg_reward_per_token,
            "reward_concentration": reward_concentration,
            "min": valid_rewards.min().item(),
            "max": valid_rewards.max().item(),
            "mean": valid_rewards.mean().item(),
        }
        results["stats"][i] = stats
        
        # Check for reward dilution
        if reward_concentration < 0.1:  # Less than 10% of tokens have reward
            results["samples_with_dilution"].append(i)
            if verbose:
                print(f"\n  ⚠️  Sample {i}: Reward dilution detected")
                print(f"      Total reward: {total_reward:.6f}")
                print(f"      Non-zero tokens: {non_zero_tokens}/{num_tokens} ({reward_concentration*100:.1f}%)")
                print(f"      Avg reward per token: {avg_reward_per_token:.6f}")
                print(f"      💡 This may cause credit assignment issues in GRPO")
        else:
            if verbose:
                print(f"\n  ✅ Sample {i}: Reward well distributed")
                print(f"      Total reward: {total_reward:.6f}")
                print(f"      Non-zero tokens: {non_zero_tokens}/{num_tokens} ({reward_concentration*100:.1f}%)")
                print(f"      Avg reward per token: {avg_reward_per_token:.6f}")
                print(f"      Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    return results


def log_loss_scale_stability(
    losses: List[float],
    response_lengths: List[int],
    step: int,
    window_size: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Log loss scale stability to check if loss varies with response length.
    
    Args:
        losses: list of recent loss values
        response_lengths: list of corresponding response lengths
        step: current training step
        window_size: number of recent steps to analyze
        verbose: whether to print details
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "step": step,
        "stable": True,
        "correlation": 0.0,
        "potential_issue": False,
    }
    
    if len(losses) < 10:
        return results
    
    # Use recent window
    recent_losses = losses[-window_size:]
    recent_lengths = response_lengths[-window_size:]
    
    if len(recent_losses) < 2:
        return results
    
    # Calculate correlation
    correlation = np.corrcoef(recent_losses, recent_lengths)[0, 1]
    results["correlation"] = correlation
    
    # Check for stability issues
    if abs(correlation) > 0.5:
        results["potential_issue"] = True
        results["stable"] = False
    
    if verbose and results["potential_issue"]:
        print(f"\n{'='*80}")
        print(f"[GRADIENT FLOW DEBUG] Step {step} - Loss Scale Stability Warning")
        print(f"{'='*80}")
        print(f"  ⚠️  Loss scale is correlated with response length (corr={correlation:.3f})")
        print(f"      This suggests loss aggregation normalization issue.")
        print(f"      💡 Recommendation: Use fixed normalization factor in loss aggregation")
    
    return results


def log_gradient_norm(
    model: torch.nn.Module,
    step: int,
    max_norm: float = 10.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Log gradient norm to check for gradient explosion or vanishing.
    
    Args:
        model: the model to check gradients for
        step: current training step
        max_norm: maximum expected gradient norm
        verbose: whether to print details
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "step": step,
        "total_norm": 0.0,
        "max_norm": 0.0,
        "has_nan": False,
        "has_inf": False,
        "potential_issue": False,
    }
    
    total_norm_sq = 0.0
    max_param_norm = 0.0
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            param_norm_sq = param_norm ** 2
            total_norm_sq += param_norm_sq
            max_param_norm = max(max_param_norm, param_norm)
            
            if torch.isnan(param.grad).any():
                has_nan = True
            if torch.isinf(param.grad).any():
                has_inf = True
    
    total_norm = total_norm_sq ** 0.5
    results["total_norm"] = total_norm
    results["max_norm"] = max_param_norm
    results["has_nan"] = has_nan
    results["has_inf"] = has_inf
    
    if total_norm > max_norm or has_nan or has_inf:
        results["potential_issue"] = True
        if verbose:
            print(f"\n{'='*80}")
            print(f"[GRADIENT FLOW DEBUG] Step {step} - Gradient Norm Warning")
            print(f"{'='*80}")
            if total_norm > max_norm:
                print(f"  ⚠️  Gradient norm too large: {total_norm:.6f} > {max_norm}")
                print(f"      💡 Recommendation: Use gradient clipping")
            if has_nan:
                print(f"  ⚠️  NaN detected in gradients!")
            if has_inf:
                print(f"  ⚠️  Inf detected in gradients!")
    
    return results


def log_multi_turn_statistics(
    batch: Any,
    step: int,
    sample_idx: int = 0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Log multi-turn statistics to check token distribution across turns.
    
    Args:
        batch: DataProto batch with step_ids, response_types, etc.
        step: current training step
        sample_idx: which sample to analyze
        verbose: whether to print details
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "step": step,
        "sample_idx": sample_idx,
        "turns": {},
        "total_tokens": 0,
    }
    
    if sample_idx >= batch.batch["responses"].shape[0]:
        return results
    
    if "step_ids" not in batch.batch or "responses_types" not in batch.batch:
        if verbose:
            print(f"\n{'='*80}")
            print(f"[GRADIENT FLOW DEBUG] Step {step} - Multi-turn Statistics (Not Available)")
            print(f"{'='*80}")
            print(f"  ℹ️  step_ids or responses_types not available in batch")
        return results
    
    step_ids = batch.batch["step_ids"][sample_idx]
    responses_types = batch.batch["responses_types"][sample_idx]
    response_mask = batch.batch.get("response_mask", None)
    
    if response_mask is not None:
        valid_mask = response_mask[sample_idx] > 0
    else:
        valid_mask = torch.ones_like(step_ids, dtype=torch.bool)
    
    # Get unique turns
    unique_steps = step_ids.unique().tolist()
    
    print(f"\n{'='*80}")
    print(f"[GRADIENT FLOW DEBUG] Step {step} - Multi-turn Statistics (Sample {sample_idx})")
    print(f"{'='*80}")
    print(f"Total turns: {len(unique_steps)}")
    
    turn_stats = []
    for turn_id in unique_steps[:10]:  # Analyze first 10 turns
        turn_mask = (step_ids == turn_id) & valid_mask
        turn_valid_tokens = turn_mask.sum().item()
        
        if turn_valid_tokens == 0:
            continue
        
        # Count token types in this turn
        turn_types = responses_types[turn_mask]
        type_counts = {}
        for rt in turn_types.unique().tolist():
            type_name = {
                0: 'think', 1: 'search', 2: 'answer',
                3: 'information', 4: 'info_summary', 5: 'think_summary'
            }.get(rt, f'type_{rt}')
            type_counts[type_name] = (turn_types == rt).sum().item()
        
        # Get advantage statistics for this turn (if available)
        turn_adv_stats = {}
        if "advantages" in batch.batch:
            turn_advantages = batch.batch["advantages"][sample_idx][turn_mask]
            turn_adv_stats = {
                "mean": turn_advantages.mean().item() if len(turn_advantages) > 0 else 0.0,
                "std": turn_advantages.std().item() if len(turn_advantages) > 0 else 0.0,
                "min": turn_advantages.min().item() if len(turn_advantages) > 0 else 0.0,
                "max": turn_advantages.max().item() if len(turn_advantages) > 0 else 0.0,
            }
        
        turn_stat = {
            "turn_id": turn_id,
            "num_tokens": turn_valid_tokens,
            "type_counts": type_counts,
            "advantage_stats": turn_adv_stats,
        }
        turn_stats.append(turn_stat)
        
        if verbose:
            type_str = ', '.join([f"{k}:{v}" for k, v in type_counts.items() if v > 0])
            print(f"\n  Turn {turn_id}: {turn_valid_tokens} valid tokens")
            print(f"    Types: {type_str}")
            if turn_adv_stats:
                print(f"    Advantage: mean={turn_adv_stats['mean']:.6f}, std={turn_adv_stats['std']:.6f}")
                print(f"                range=[{turn_adv_stats['min']:.6f}, {turn_adv_stats['max']:.6f}]")
    
    results["turns"] = turn_stats
    results["total_tokens"] = valid_mask.sum().item()
    
    if len(unique_steps) > 10:
        print(f"\n  ... ({len(unique_steps) - 10} more turns)")
    
    return results


def comprehensive_gradient_flow_check(
    batch: Any,
    step: int,
    adv_estimator: str = "unknown",
    model: Optional[torch.nn.Module] = None,
    check_loss_history: Optional[List[Tuple[float, int]]] = None,
    sample_indices: List[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive gradient flow check covering all diagnostic items.
    
    Args:
        batch: DataProto batch
        step: current training step
        adv_estimator: advantage estimator name
        model: model to check gradients (optional)
        check_loss_history: list of (loss, response_length) tuples (optional)
        sample_indices: which samples to analyze
        verbose: whether to print details
        
    Returns:
        Dictionary with all check results
    """
    # Check frequency from environment variable
    check_freq = int(os.environ.get("GRADIENT_FLOW_CHECK_FREQ", "1"))
    
    if step % check_freq != 0:
        return {}
    
    print(f"\n{'#'*80}")
    print(f"# [GRADIENT FLOW DEBUG] Step {step} - Comprehensive Check")
    print(f"{'#'*80}")
    
    results = {
        "step": step,
        "checks": {},
    }
    
    # 1. Advantage distribution
    if "advantages" in batch.batch and "response_mask" in batch.batch:
        results["checks"]["advantage_distribution"] = log_advantage_distribution(
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            step=step,
            sample_indices=sample_indices,
            adv_estimator=adv_estimator,
            verbose=verbose
        )
    
    # 2. Reward distribution
    if "token_level_rewards" in batch.batch and "response_mask" in batch.batch:
        results["checks"]["reward_distribution"] = log_reward_distribution(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            step=step,
            sample_indices=sample_indices,
            verbose=verbose
        )
    
    # 3. Loss scale stability
    if check_loss_history is not None:
        losses = [l for l, _ in check_loss_history]
        lengths = [rl for _, rl in check_loss_history]
        results["checks"]["loss_scale"] = log_loss_scale_stability(
            losses=losses,
            response_lengths=lengths,
            step=step,
            verbose=verbose
        )
    
    # 4. Gradient norm
    if model is not None:
        results["checks"]["gradient_norm"] = log_gradient_norm(
            model=model,
            step=step,
            verbose=verbose
        )
    
    # 5. Multi-turn statistics
    if sample_indices is None:
        sample_indices = [0]
    for sample_idx in sample_indices[:1]:  # Only check first sample for multi-turn stats
        results["checks"][f"multi_turn_sample_{sample_idx}"] = log_multi_turn_statistics(
            batch=batch,
            step=step,
            sample_idx=sample_idx,
            verbose=verbose
        )
    
    # 6. Mask coverage (basic check)
    if "response_mask" in batch.batch and "responses" in batch.batch:
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        
        if response_mask.shape == responses.shape:
            mask_coverage = response_mask.sum().item() / (response_mask.shape[0] * response_mask.shape[1])
            results["checks"]["mask_coverage"] = {
                "coverage_ratio": mask_coverage,
                "valid_tokens": response_mask.sum().item(),
                "total_tokens": response_mask.numel(),
            }
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"[GRADIENT FLOW DEBUG] Step {step} - Mask Coverage")
                print(f"{'='*80}")
                print(f"  Valid tokens: {response_mask.sum().item()}/{response_mask.numel()}")
                print(f"  Coverage ratio: {mask_coverage:.2%}")
    
    print(f"\n{'#'*80}\n")
    
    return results

