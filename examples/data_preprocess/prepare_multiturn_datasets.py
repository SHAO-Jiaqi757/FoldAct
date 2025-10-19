#!/usr/bin/env python3
"""
Prepare multiturn datasets for MultiTurnSFTDataset training.

This script:
1. Processes the JSONL files to create proper multiturn format
2. Splits into train/val datasets
3. Converts to parquet format for MultiTurnSFTDataset
4. Creates the necessary directory structure
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import random


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of samples."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    return samples


def save_jsonl(samples: List[Dict[str, Any]], file_path: str) -> None:
    """Save samples to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def convert_to_parquet(jsonl_file: str, parquet_file: str) -> None:
    """Convert JSONL file to parquet format for MultiTurnSFTDataset."""
    print(f"Converting {jsonl_file} to {parquet_file}...")
    
    # Load JSONL data
    samples = load_jsonl(jsonl_file)
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Save as parquet
    Path(parquet_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_file, index=False)
    
    print(f"✓ Converted {len(samples)} samples to {parquet_file}")


def split_train_val(samples: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Split samples into train and validation sets."""
    random.seed(seed)
    random.shuffle(samples)
    
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    return train_samples, val_samples


def process_summary_prefix_dataset(input_file: str, output_dir: str, val_ratio: float = 0.1) -> None:
    """Process the summary prefix dataset (Dataset 1)."""
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY PREFIX DATASET (Dataset 1)")
    print(f"{'='*60}")
    
    # Load samples
    samples = load_jsonl(input_file)
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Split into train/val
    train_samples, val_samples = split_train_val(samples, val_ratio)
    print(f"Split: {len(train_samples)} train, {len(val_samples)} val")
    
    # Save JSONL files
    train_jsonl = f"{output_dir}/sft_train_summary_prefix_train.jsonl"
    val_jsonl = f"{output_dir}/sft_train_summary_prefix_val.jsonl"
    
    save_jsonl(train_samples, train_jsonl)
    save_jsonl(val_samples, val_jsonl)
    
    # Convert to parquet
    train_parquet = f"{output_dir}/sft_train_summary_prefix_train.parquet"
    val_parquet = f"{output_dir}/sft_train_summary_prefix_val.parquet"
    
    convert_to_parquet(train_jsonl, train_parquet)
    convert_to_parquet(val_jsonl, val_parquet)
    
    print(f"✓ Summary prefix dataset ready:")
    print(f"  Train: {train_jsonl} + {train_parquet}")
    print(f"  Val:   {val_jsonl} + {val_parquet}")


def process_summary_only_dataset(input_file: str, output_dir: str, val_ratio: float = 0.1) -> None:
    """Process the summary only dataset (Dataset 2)."""
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY ONLY DATASET (Dataset 2)")
    print(f"{'='*60}")
    
    # Load samples
    samples = load_jsonl(input_file)
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Split into train/val
    train_samples, val_samples = split_train_val(samples, val_ratio)
    print(f"Split: {len(train_samples)} train, {len(val_samples)} val")
    
    # Save JSONL files
    train_jsonl = f"{output_dir}/sft_train_summary_only_train.jsonl"
    val_jsonl = f"{output_dir}/sft_train_summary_only_val.jsonl"
    
    save_jsonl(train_samples, train_jsonl)
    save_jsonl(val_samples, val_jsonl)
    
    # Convert to parquet
    train_parquet = f"{output_dir}/sft_train_summary_only_train.parquet"
    val_parquet = f"{output_dir}/sft_train_summary_only_val.parquet"
    
    convert_to_parquet(train_jsonl, train_parquet)
    convert_to_parquet(val_jsonl, val_parquet)
    
    print(f"✓ Summary only dataset ready:")
    print(f"  Train: {train_jsonl} + {train_parquet}")
    print(f"  Val:   {val_jsonl} + {val_parquet}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multiturn datasets for MultiTurnSFTDataset training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both datasets
  python3 prepare_multiturn_datasets.py \\
      --summary_prefix_file data/sft_compress/sft_train_summary_prefix.jsonl \\
      --summary_only_file data/sft_compress/sft_train_summary_only.jsonl \\
      --output_dir data/sft_compress \\
      --val_ratio 0.1

  # Process only summary prefix dataset
  python3 prepare_multiturn_datasets.py \\
      --summary_prefix_file data/sft_compress/sft_train_summary_prefix.jsonl \\
      --output_dir data/sft_compress
        """
    )
    
    parser.add_argument(
        "--summary_prefix_file",
        type=str,
        help="Path to summary prefix JSONL file (Dataset 1)"
    )
    parser.add_argument(
        "--summary_only_file", 
        type=str,
        help="Path to summary only JSONL file (Dataset 2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sft_compress",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    if not args.summary_prefix_file and not args.summary_only_file:
        print("Error: At least one dataset file must be provided")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process summary prefix dataset (Dataset 1)
    if args.summary_prefix_file:
        if not Path(args.summary_prefix_file).exists():
            print(f"Error: Summary prefix file not found: {args.summary_prefix_file}")
            return 1
        process_summary_prefix_dataset(
            args.summary_prefix_file, 
            args.output_dir, 
            args.val_ratio
        )
    
    # Process summary only dataset (Dataset 2)
    if args.summary_only_file:
        if not Path(args.summary_only_file).exists():
            print(f"Error: Summary only file not found: {args.summary_only_file}")
            return 1
        process_summary_only_dataset(
            args.summary_only_file, 
            args.output_dir, 
            args.val_ratio
        )
    
    print(f"\n{'='*60}")
    print("DATASET PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print("\nFiles created:")
    for file_path in Path(args.output_dir).rglob("*"):
        if file_path.is_file():
            print(f"  {file_path}")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Update your training scripts to use MultiTurnSFTDataset")
    print("2. Set data.messages_key='prompt' in your config")
    print("3. Use the parquet files for training")
    print("4. The datasets are now compatible with MultiTurnSFTDataset")
    
    return 0


if __name__ == "__main__":
    exit(main())

