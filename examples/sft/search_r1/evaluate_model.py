#!/usr/bin/env python3
"""
Evaluate trained model performance on:
1. Summary-Reasoning: Model generates summary from full context, then reasons
2. Reasoning from Summary: Model reasons directly from given summary
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer


def load_test_data(test_file: str, limit: int = None) -> List[Dict]:
    """Load test data from JSONL file."""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def test_summary_reasoning(
    model,
    tokenizer,
    sample: Dict,
    max_new_tokens: int = 512
) -> Dict[str, str]:
    """
    Test 1: Summary-Reasoning
    Input: Full context (question + reasoning + search results)
    Expected: Generate summary first, then reasoning/answer
    """
    # Use the prompt content (consistent with training)
    full_context = sample['prompt'][0]['content']
    
    # Prepare input - tokenize directly
    messages = [{"role": "user", "content": full_context}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract only the generated tokens (token-level slicing)
    generated_tokens = outputs[0, input_length:]
    
    # Decode only the generated part
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return {
        "input": full_context,
        "generated": response,
        "expected_summary": sample.get('summary', 'N/A'),
        "expected_answer": sample['extra_info']['answer']
    }


def test_reasoning_from_summary(
    model,
    tokenizer,
    sample: Dict,
    max_new_tokens: int = 256
) -> Dict[str, str]:
    """
    Test 2: Reasoning from Summary
    Input: Summary only (concise context)
    Expected: Generate reasoning and answer directly
    """
    # Use summary as input
    summary = sample.get('summary', '')
    
    if not summary:
        return {"error": "No summary available"}
    
    # Prepare input - tokenize directly
    messages = [{"role": "user", "content": summary}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract only the generated tokens (token-level slicing)
    generated_tokens = outputs[0, input_length:]
    
    # Decode only the generated part
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return {
        "input_summary": summary,
        "generated": response,
        "expected_answer": sample['extra_info']['answer']
    }


def evaluate_model(
    model_path: str,
    test_file: str,
    output_file: str,
    num_samples: int = 10,
    max_new_tokens: int = 512
):
    """Main evaluation function."""
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load test data
    print(f"\nLoading test data from {test_file}...")
    test_samples = load_test_data(test_file, limit=num_samples)
    print(f"✓ Loaded {len(test_samples)} test samples")
    
    # Detect which tests to run based on data structure
    # Check if prompt contains full context (not summary format)
    has_full_context = False
    has_summary = False
    
    if test_samples:
        # Check if prompt exists and contains full context (has <search_results> not <search_results_summary>)
        if 'prompt' in test_samples[0]:
            prompt_content = test_samples[0]['prompt'][0]['content']
            # If it contains <search_results_summary>, it's a summary format, not full context
            has_full_context = '<search_results_summary>' not in prompt_content and '<reasoning_summary>' not in prompt_content
        
        # Check if summary field exists
        has_summary = 'summary' in test_samples[0] and test_samples[0].get('summary')
    
    print(f"\n📊 Data analysis:")
    print(f"  - Full context available: {'✓' if has_full_context else '✗'}")
    print(f"  - Summary available: {'✓' if has_summary else '✗'}")
    
    results = {
        "model_path": model_path,
        "test_file": test_file,
        "num_samples": len(test_samples),
        "summary_reasoning_results": [],
        "reasoning_from_summary_results": []
    }
    
    print("\n" + "="*80)
    print("EVALUATION STARTED")
    print("="*80)
    
    # Test 1: Summary-Reasoning (only if full context is available)
    if has_full_context:
        print("\n📝 Test 1: Summary-Reasoning (Full Context → Summary + Answer)")
        print("-"*80)
        
        for i, sample in enumerate(tqdm(test_samples, desc="Summary-Reasoning")):
            result = test_summary_reasoning(
                model, tokenizer, sample, max_new_tokens
            )
            result['sample_id'] = i
            results['summary_reasoning_results'].append(result)
            
            if i == 0:  # Show first example
                print(f"\n【Example 1】")
                print(f"Input (first 200 chars): {result['input'][:200]}...")
                print(f"\nGenerated:\n{result['generated'][:400]}...")
                print(f"\nExpected Summary:\n{result['expected_summary'][:200]}...")
    else:
        print("\n⊘ Test 1: Summary-Reasoning - SKIPPED (no full context in data)")
    
    # Test 2: Reasoning from Summary (only if summary is available)
    if has_summary:
        print("\n\n🤔 Test 2: Reasoning from Summary (Summary → Answer)")
        print("-"*80)
        
        for i, sample in enumerate(tqdm(test_samples, desc="Reasoning from Summary")):
            result = test_reasoning_from_summary(
                model, tokenizer, sample, max_new_tokens
            )
            result['sample_id'] = i
            results['reasoning_from_summary_results'].append(result)
            
            if i == 0:  # Show first example
                print(f"\n【Example 1】")
                print(f"Input Summary (first 200 chars): {result['input_summary'][:200]}...")
                print(f"\nGenerated:\n{result['generated'][:400]}...")
                print(f"\nExpected:\n{result['expected_answer'][:200]}...")
    else:
        print("\n⊘ Test 2: Reasoning from Summary - SKIPPED (no summary in data)")
    
    # Save results
    print(f"\n\n💾 Saving results to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Results saved!")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Results saved to: {output_file}")
    print("\nTests performed:")
    if has_full_context:
        print(f"  ✓ Test 1 - Summary-Reasoning: {len(results['summary_reasoning_results'])} samples")
    else:
        print(f"  ⊘ Test 1 - Summary-Reasoning: SKIPPED")
    if has_summary:
        print(f"  ✓ Test 2 - Reasoning from Summary: {len(results['reasoning_from_summary_results'])} samples")
    else:
        print(f"  ⊘ Test 2 - Reasoning from Summary: SKIPPED")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on summary and reasoning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on 10 samples
  python3 evaluate_model.py \\
      --model_path verl_checkpoints/sft_best/phase2_summary_prefix/global_step_XXX \\
      --test_file data/sft_compress/sft_train_with_summary.jsonl \\
      --output results/evaluation_results.json \\
      --num_samples 10

  # Evaluate on 50 samples with longer generation
  python3 evaluate_model.py \\
      --model_path /tmp/sft_output/global_step_XXX \\
      --test_file data/sft_compress/sft_train_with_summary.jsonl \\
      --output results/evaluation_results.json \\
      --num_samples 50 \\
      --max_new_tokens 768
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test data (JSONL with summary field)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

