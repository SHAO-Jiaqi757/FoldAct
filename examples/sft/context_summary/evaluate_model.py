#!/usr/bin/env python3
"""
Evaluate trained model performance on any dataset.
Unified evaluation that processes samples with their prompts and generates responses.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

# Disable vLLM v1 which has issues
os.environ['VLLM_USE_V1'] = '0'

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def remove_assistant_from_context(messages: List[Dict]) -> List[Dict]:
    """Remove assistant messages from context for evaluation."""
    # Remove the last assistant message if it exists (added during dataset creation)
    if messages and messages[-1].get('role') == 'assistant':
        return messages[:-1]
    return messages


def evaluate_with_vllm(
    model_path: str,
    test_file: str,
    output_file: str,
    num_samples: int = 10,
    max_new_tokens: int = 1024
):
    """Evaluate using vLLM with training-matched parameters."""
    
 
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Tokenizer loaded")
    
    # Initialize vLLM with training-matched config
    print("\nInitializing vLLM engine...")
    print("⚠️  Using vLLM (non-v1) to match training inference")
    
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=8192,  # Reduced from 12288 for single GPU
        gpu_memory_utilization=0.7,
        enable_chunked_prefill=False,  # Disable for single GPU simplicity
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    
    print("✓ vLLM engine initialized")
    
    # Load test data
    print(f"\nLoading test data from {test_file}...")
    test_samples = load_test_data(test_file, limit=num_samples)
    print(f"✓ Loaded {len(test_samples)} test samples")
    
    # Create sampling params matching training
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    
    results = {
        "model_path": model_path,
        "test_file": test_file,
        "num_samples": len(test_samples),
        "inference_backend": "vLLM (single GPU)",
        "config": {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": max_new_tokens,
            "dtype": "float16",
        },
        "evaluation_results": []
    }
    
    print("\n" + "="*80)
    print("EVALUATION STARTED")
    print("="*80)
    
    # Prepare prompts and track raw formats
    prompts = []
    prompt_details = []  # Track both readable and raw formats
    
    for sample in test_samples:
        messages = sample['prompt'].copy()
        if messages and messages[-1].get('role') == 'assistant':
            messages = messages[:-1]
        
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt_text)
        
        # Also get tokenized version to show raw format
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )
        
        # Decode with and without special tokens
        prompt_readable = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        
        prompt_details.append({
            "readable": prompt_readable,
            "token_count": len(prompt_ids[0]),
        })
    
    # Generate with vLLM
    print(f"\n📝 Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Process results
    for i, (sample, output, prompt_detail) in enumerate(zip(test_samples, outputs, prompt_details)):
        generated = output.outputs[0].text.strip()
        
        result = {
            "input": sample['prompt'],
            "input_readable": prompt_detail['readable'],
            "input_raw": prompt_detail['raw'],
            "input_token_count": prompt_detail['token_count'],
            "generated": generated,
            "expected_summary": sample.get('summary', 'N/A'),
            "expected_answer": sample['answer'],
            "context_length": len(sample['prompt']),
            "sample_id": i
        }
        results['evaluation_results'].append(result)
        
        if i == 0:  # Show first example
            print(f"\n【Example 1】")
            print(f"Context length: {result['context_length']} messages")
            print(f"Input (first 200 chars): {result['input'][:200]}...")
            print(f"\nGenerated:\n{result['generated'][:400]}...")
            print(f"\nExpected Summary:\n{result['expected_summary'][:200]}...")
            print(f"\nExpected Answer:\n{result['expected_answer'][:200]}...")
    
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
    print(f"  ✓ Evaluation completed: {len(results['evaluation_results'])} samples")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple vLLM evaluation matching training inference",
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
        help="Path to test data (JSONL file)"
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
    
    evaluate_with_vllm(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

