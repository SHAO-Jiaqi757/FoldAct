"""
简洁的 responses 和 response_mask 检查工具
"""

import torch
from typing import Dict, Optional
from verl import DataProto


def check_responses_and_mask(
    batch: DataProto,
    sample_idx: int = 0,
    verbose: bool = True,
    tokenizer=None
) -> Dict[str, bool]:
    """
    检查 responses 和 response_mask 的正确性
    
    Args:
        batch: 数据批次
        sample_idx: 采样检查的样本索引
        verbose: 是否打印详细信息
        
    Returns:
        检查结果字典，包含各项检查的通过状态
    """
    results = {
        "responses_exists": False,
        "response_mask_exists": False,
        "shapes_match": False,
        "mask_valid": False,
        "responses_in_range": False,
        "all_passed": False,
    }
    
    # 强制输出 batch keys（帮助调试）
    print(f"Batch keys: {list(batch.batch.keys())}")
    
    if "responses" not in batch.batch:
        print("❌ responses not found in batch")
        print(f"Available keys: {list(batch.batch.keys())}")
        return results
    
    responses = batch.batch["responses"]
    results["responses_exists"] = True
    
    # 强制输出基本信息（即使 verbose=False）
    print(f"📊 Responses check:")
    print(f"  Shape: {responses.shape}")
    print(f"  Dtype: {responses.dtype}")
    try:
        print(f"  Min: {responses.min().item()}, Max: {responses.max().item()}")
    except Exception as e:
        print(f"  ⚠️  Failed to get min/max: {e}")
    
    # 检查值域（token IDs 应该是非负整数）
    if responses.min() >= 0:
        results["responses_in_range"] = True
    elif verbose:
        print(f"  ⚠️  WARNING: Negative token IDs found!")
    
    # 检查 response_mask
    if "response_mask" in batch.batch:
        response_mask = batch.batch["response_mask"]
        results["response_mask_exists"] = True
        
        # 强制输出基本信息
        print(f"\n📊 Response mask check:")
        print(f"  Shape: {response_mask.shape}")
        print(f"  Dtype: {response_mask.dtype}")
        
        # 检查 mask 来源和方法
        mask_source = "unknown"
        if "responses_types" in batch.batch:
            mask_source = "responses_types (multi-turn aware)"
        elif "info_mask" in batch.batch:
            mask_source = "info_mask"
        elif "attention_mask" in batch.batch:
            mask_source = "attention_mask"
        
        print(f"\n  🔍 Mask source: {mask_source}")
        
        # 对比 info_mask 和 attention_mask（如果存在）
        if "info_mask" in batch.batch and "attention_mask" in batch.batch:
            info_mask_full = batch.batch["info_mask"]
            attention_mask_full = batch.batch["attention_mask"]
            response_length = responses.shape[1]
            
            if info_mask_full.shape[1] >= response_length and attention_mask_full.shape[1] >= response_length:
                info_mask_response = info_mask_full[:, -response_length:]
                attention_mask_response = attention_mask_full[:, -response_length:]
                
                # 比较第一个样本
                if sample_idx < info_mask_response.shape[0]:
                    info_sample = info_mask_response[sample_idx]
                    attn_sample = attention_mask_response[sample_idx]
                    mask_sample = response_mask[sample_idx]
                    
                    info_valid = info_sample.sum().item()
                    attn_valid = attn_sample.sum().item()
                    mask_valid = mask_sample.sum().item()
                    
                    print(f"  📊 Mask comparison (Sample {sample_idx}):")
                    print(f"    info_mask valid tokens: {info_valid}")
                    print(f"    attention_mask valid tokens: {attn_valid}")
                    print(f"    response_mask valid tokens: {mask_valid}")
                    
                    if mask_source == "responses_types (multi-turn aware)":
                        if mask_valid > info_valid:
                            print(f"    ✅ response_mask uses responses_types - includes {mask_valid - info_valid} more assistant tokens from earlier turns")
                        elif mask_valid == info_valid:
                            print(f"    ℹ️  response_mask matches info_mask (may indicate only last turn has assistant tokens)")
                        else:
                            print(f"    ⚠️  Unexpected: response_mask has fewer tokens than info_mask")
                    elif info_valid != attn_valid:
                        print(f"    ⚠️  Difference detected: info_mask excludes {attn_valid - info_valid} information tokens")
                        if mask_valid == info_valid:
                            print(f"    ✅ response_mask correctly uses info_mask")
                        elif mask_valid == attn_valid:
                            print(f"    ❌ response_mask incorrectly uses attention_mask (BUG!)")
                        else:
                            print(f"    ⚠️  response_mask doesn't match either mask (unexpected)")
                    else:
                        print(f"    ℹ️  info_mask and attention_mask are identical (no information blocks)")
        
        # 检查形状匹配
        if responses.shape == response_mask.shape:
            results["shapes_match"] = True
        elif verbose:
            print(f"  ❌ Shape mismatch: responses {responses.shape} vs mask {response_mask.shape}")
        
        # 检查 mask 有效性
        if results["shapes_match"]:
            mask_sum = response_mask.sum(dim=-1)  # (bs,)
            valid_lengths = mask_sum
            
            if verbose:
                print(f"  Valid tokens per sample: {valid_lengths.tolist()[:5]}...")
                print(f"  Mean valid ratio: {(mask_sum / responses.shape[1]).mean().item():.2%}")
            
            # 检查是否有无效的 mask（全0或全1不合理）
            if (mask_sum > 0).all() and (mask_sum < responses.shape[1]).any():
                results["mask_valid"] = True
            elif verbose:
                if (mask_sum == 0).any():
                    print(f"  ⚠️  WARNING: Some samples have zero valid tokens!")
                if (mask_sum == responses.shape[1]).all():
                    print(f"  ⚠️  WARNING: All tokens are marked as valid (unusual)")
        
        # 检查样本级别的对齐 - 总是显示至少一个样本的详细信息
        if sample_idx < responses.shape[0] and results["shapes_match"]:
            resp_sample = responses[sample_idx]
            mask_sample = response_mask[sample_idx]
            valid_len = mask_sample.sum().item()
            
            # 总是显示基本信息（不管 verbose 设置）
            print(f"\n📋 Sample {sample_idx} details:")
            print(f"  Response length: {len(resp_sample)}")
            print(f"  Valid tokens: {valid_len}")
            
            # 显示多轮信息（如果可用）
            if "step_ids" in batch.batch and "responses_types" in batch.batch:
                step_ids_sample = batch.batch["step_ids"][sample_idx]
                responses_types_sample = batch.batch["responses_types"][sample_idx]
                
                # 统计每个 step 的有效 token 数
                unique_steps = step_ids_sample.unique().tolist()
                print(f"\n  🔄 Multi-turn analysis:")
                print(f"    Total steps: {len(unique_steps)}")
                
                for step_id in unique_steps[:10]:  # 最多显示前10个step
                    step_mask = (step_ids_sample == step_id) & (mask_sample > 0)
                    step_valid_count = step_mask.sum().item()
                    
                    if step_valid_count > 0:
                        # 统计这个 step 的 token 类型（只统计有效的 assistant tokens，排除 information）
                        # CRITICAL FIX: Only count tokens that are both in this step AND valid in response_mask
                        step_types = responses_types_sample[(step_ids_sample == step_id) & (mask_sample > 0)]
                        type_counts = {}
                        for rt in step_types.unique().tolist():
                            type_name = {0: 'think', 1: 'search', 2: 'answer', 
                                        3: 'information', 4: 'info_summary', 5: 'think_summary'}.get(rt, f'type_{rt}')
                            type_counts[type_name] = (step_types == rt).sum().item()
                        
                        type_str = ', '.join([f"{k}:{v}" for k, v in type_counts.items() if v > 0])
                        print(f"    Step {step_id}: {step_valid_count} valid tokens ({type_str})")
                
                if len(unique_steps) > 10:
                    print(f"    ... ({len(unique_steps) - 10} more steps)")
            
            # 显示 mask 模式（总是显示）
            mask_list = mask_sample.tolist()
            first_valid_idx = next((i for i, m in enumerate(mask_list) if m > 0), None)
            last_valid_idx = next((i for i in range(len(mask_list)-1, -1, -1) if mask_list[i] > 0), None)
            if first_valid_idx is not None and last_valid_idx is not None:
                print(f"\n  📊 Mask pattern:")
                print(f"    Valid token range: [{first_valid_idx}, {last_valid_idx}]")
                print(f"    Mask pattern (first 50): {''.join(['1' if m > 0 else '0' for m in mask_list[:50]])}")
                
                # 显示 mask 模式（中间段）
                if len(mask_list) > 100:
                    mid_start = len(mask_list) // 2 - 25
                    mid_end = mid_start + 50
                    print(f"    Mask pattern (middle): {''.join(['1' if m > 0 else '0' for m in mask_list[mid_start:mid_end]])}")
                
                # 显示 mask 模式（末尾）
                print(f"    Mask pattern (last 50): {''.join(['1' if m > 0 else '0' for m in mask_list[-50:]])}")
            
            # 显示实际响应文本（如果有 tokenizer，总是显示）
            if tokenizer is not None:
                try:
                    # 只解码有效 token
                    valid_tokens = resp_sample[mask_sample.bool() if mask_sample.dtype == torch.bool else mask_sample > 0]
                    if len(valid_tokens) > 0:
                        decoded_text = tokenizer.decode(valid_tokens.tolist(), skip_special_tokens=False)
                        print(f"\n  📝 Decoded response (valid tokens only, {len(valid_tokens)} tokens):")
                        print(f"     {decoded_text}")
                    
                    # 如果 verbose，显示更多信息
                    if verbose:
                        # 显示完整序列（包括无效部分）
                        full_decoded = tokenizer.decode(resp_sample.tolist(), skip_special_tokens=False)
                        print(f"\n  📝 Decoded full sequence:")
                        print(f"     {full_decoded[:500]}{'...' if len(full_decoded) > 500 else ''}")
                        
                        # 显示第一个和最后一个有效 token
                        if len(valid_tokens) > 0:
                            print(f"\n  First 10 valid tokens: {valid_tokens[:10].tolist()}")
                            print(f"  Last 10 valid tokens: {valid_tokens[-10:].tolist()}")
                except Exception as e:
                    print(f"  ⚠️  Failed to decode: {e}")
                    import traceback
                    traceback.print_exc()
            
            if verbose:
                # 只在 verbose 模式下显示原始 token IDs
                print(f"\n  First 20 tokens: {resp_sample[:20].tolist()}")
                print(f"  First 20 mask: {mask_sample[:20].tolist()}")
                
                # 检查 mask 是否为 0/1
                mask_unique = mask_sample.unique().tolist()
                if mask_sample.dtype == torch.bool or (set(mask_unique).issubset({0, 1})):
                    print(f"  ✅ Mask values are binary: {mask_unique}")
                else:
                    print(f"  ⚠️  WARNING: Mask contains non-binary values: {mask_unique}")
    
    elif "attention_mask" in batch.batch:
        # 尝试从 attention_mask 推断 response_mask
        attention_mask = batch.batch["attention_mask"]
        response_length = responses.shape[1]
        
        # 强制输出
        print(f"\n📊 Inferring response_mask from attention_mask:")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Response length: {response_length}")
        
        if attention_mask.shape[1] >= response_length:
            inferred_mask = attention_mask[:, -response_length:]
            print(f"  ✅ Inferred mask shape: {inferred_mask.shape}")
            try:
                print(f"  Valid tokens: {inferred_mask.sum(dim=-1).tolist()[:5]}...")
            except Exception as e:
                print(f"  ⚠️  Failed to compute valid tokens: {e}")
    else:
        print(f"\n⚠️  Neither response_mask nor attention_mask found in batch")
    
    # 综合结果
    results["all_passed"] = all([
        results["responses_exists"],
        results["response_mask_exists"] or "attention_mask" in batch.batch,
        results["shapes_match"] if results["response_mask_exists"] else True,
        results["mask_valid"] if results["shapes_match"] else True,
        results["responses_in_range"],
    ])
    
    # 强制输出结果（即使 verbose=False）
    print(f"\n{'✅' if results['all_passed'] else '❌'} Overall check: {'PASSED' if results['all_passed'] else 'FAILED'}")
    print(f"  Responses exists: {results['responses_exists']}")
    print(f"  Response mask exists: {results['response_mask_exists']}")
    print(f"  Shapes match: {results['shapes_match']}")
    print(f"  Mask valid: {results['mask_valid']}")
    print(f"  Responses in range: {results['responses_in_range']}")
    
    return results


def check_responses_in_training(
    batch: DataProto,
    step: int = 0,
    check_freq: int = 10,
    verbose: bool = True,
    tokenizer=None
) -> bool:
    """
    在训练循环中使用的便捷函数
    
    Args:
        batch: 数据批次
        step: 当前训练步数
        check_freq: 检查频率
        verbose: 是否详细输出
        tokenizer: Tokenizer for decoding (optional)
        
    Returns:
        True if checks passed
    """
    if check_freq <= 0 or (step % check_freq != 0):
        return True
    
    try:
        # 强制输出检查开始标记
        print(f"\n{'='*80}")
        print(f"[RESPONSES CHECK] Step {step}")
        print(f"{'='*80}")
        
        results = check_responses_and_mask(batch, sample_idx=0, verbose=verbose, tokenizer=tokenizer)
        
        # 强制输出检查结果
        print(f"{'='*80}\n")
        
        return results["all_passed"]
    except Exception as e:
        print(f"⚠️  [RESPONSES CHECK] Step {step} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

