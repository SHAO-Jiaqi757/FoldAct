#!/usr/bin/env python3
"""
Split summary dataset into two variants:
1. Dataset 1: Keep original prompt, answer = summary + original answer
2. Dataset 2: prompt = summary only (discard original prompt), keep original answer
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def create_dataset1(sample: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Dataset 1: Keep original prompt, answer = summary + original answer
    """
    new_sample = sample.copy()
    
    # Get summary and original answer
    summary = sample.get("summary", "")
    original_answer = sample["extra_info"]["answer"]
    
    # Combine summary + answer
    combined_answer = f"{summary}\n\n{original_answer}"
    
    # Update extra_info
    new_sample["extra_info"] = sample["extra_info"].copy()
    new_sample["extra_info"]["answer"] = combined_answer
    new_sample["extra_info"]["has_summary_prefix"] = True
    
    return new_sample


def create_dataset2(sample: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Dataset 2: prompt = summary only (discard original prompt), keep original answer
    """
    new_sample = sample.copy()
    
    # Get summary
    summary = sample.get("summary", "")
    
    # Replace prompt with summary
    new_sample["prompt"] = [
        {
            "role": "user",
            "content": summary
        }
    ]
    
    # Update extra_info question to use summary
    new_sample["extra_info"] = sample["extra_info"].copy()
    new_sample["extra_info"]["question"] = summary
    new_sample["extra_info"]["original_question_discarded"] = True
    
    return new_sample


def process_file(
    input_file: str,
    output_dataset1: str,
    output_dataset2: str,
    limit: int = None
):
    """Process input file and create two output datasets."""
    
    samples = []
    
    # Read input
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    # Check if summary exists
                    if "summary" not in sample or not sample["summary"]:
                        print(f"Warning: Line {i+1} has no summary, skipping")
                        continue
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i+1}: {e}")
                    continue
    
    print(f"Loaded {len(samples)} samples with summaries")
    
    if not samples:
        print("No samples to process!")
        return
    
    # Create Dataset 1
    print(f"\nCreating Dataset 1: answer = summary + original_answer")
    dataset1 = [create_dataset1(s) for s in samples]
    
    # Create Dataset 2
    print(f"Creating Dataset 2: prompt = summary, discard original prompt")
    dataset2 = [create_dataset2(s) for s in samples]
    
    # Save Dataset 1
    print(f"\nSaving Dataset 1 to {output_dataset1}...")
    Path(output_dataset1).parent.mkdir(parents=True, exist_ok=True)
    with open(output_dataset1, 'w', encoding='utf-8') as f:
        for sample in dataset1:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save Dataset 2
    print(f"Saving Dataset 2 to {output_dataset2}...")
    Path(output_dataset2).parent.mkdir(parents=True, exist_ok=True)
    with open(output_dataset2, 'w', encoding='utf-8') as f:
        for sample in dataset2:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Done!")
    print(f"  Dataset 1: {len(dataset1)} samples → {output_dataset1}")
    print(f"  Dataset 2: {len(dataset2)} samples → {output_dataset2}")
    
    # Show examples
    print("\n" + "="*80)
    print("DATASET 1 EXAMPLE (answer = summary + original_answer)")
    print("="*80)
    print(f"Prompt (first 200 chars):\n{dataset1[0]['extra_info']['question'][:200]}...")
    print(f"\nAnswer (first 300 chars):\n{dataset1[0]['extra_info']['answer'][:300]}...")
    
    print("\n" + "="*80)
    print("DATASET 2 EXAMPLE (prompt = summary only)")
    print("="*80)
    print(f"Prompt (first 300 chars):\n{dataset2[0]['extra_info']['question'][:300]}...")
    print(f"\nAnswer (first 200 chars):\n{dataset2[0]['extra_info']['answer'][:200]}...")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Split summary dataset into two training variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples
  python3 split_summary_datasets.py \\
      --input_file data/sft_compress/sft_train_with_summary.jsonl \\
      --output_dataset1 data/sft_compress/sft_train_summary_prefix.jsonl \\
      --output_dataset2 data/sft_compress/sft_train_summary_only.jsonl

  # Test with first 10 samples
  python3 split_summary_datasets.py \\
      --input_file data/sft_compress/sft_train_with_summary.jsonl \\
      --output_dataset1 data/sft_compress/test_dataset1.jsonl \\
      --output_dataset2 data/sft_compress/test_dataset2.jsonl \\
      --limit 10

Dataset 1: Keep original context, prepend summary to answer
  - Good for: Teaching model to first summarize then answer
  - prompt: [original question + reasoning + search results]
  - answer: [summary] + [original answer]

Dataset 2: Use summary as prompt, original answer
  - Good for: Training with condensed context
  - prompt: [summary only]
  - answer: [original answer]
        """
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with summaries"
    )
    parser.add_argument(
        "--output_dataset1",
        type=str,
        required=True,
        help="Output file for Dataset 1 (answer = summary + original_answer)"
    )
    parser.add_argument(
        "--output_dataset2",
        type=str,
        required=True,
        help="Output file for Dataset 2 (prompt = summary only)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N samples (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        process_file(
            input_file=args.input_file,
            output_dataset1=args.output_dataset1,
            output_dataset2=args.output_dataset2,
            limit=args.limit
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



