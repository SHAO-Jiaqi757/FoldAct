#!/usr/bin/env python3
"""
Split JSONL datasets into train and validation sets.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[Any, Any]], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_dataset(
    input_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        train_file: Path to output training JSONL file
        val_file: Path to output validation JSONL file
        val_ratio: Ratio of validation set (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Total samples: {len(data)}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split point
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"\nSplit summary:")
    print(f"  Train samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val samples: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    
    # Save train set
    print(f"\nSaving train set to {train_file}...")
    save_jsonl(train_data, train_file)
    
    # Save val set
    print(f"Saving val set to {val_file}...")
    save_jsonl(val_data, val_file)
    
    print("\n✓ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL dataset into train and validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split summary_only dataset (default 90/10 split)
  python3 split_train_val.py \\
      --input data/sft_compress/sft_train_summary_only.jsonl \\
      --train data/sft_compress/sft_train_summary_only_train.jsonl \\
      --val data/sft_compress/sft_train_summary_only_val.jsonl

  # Split summary_prefix dataset with custom split ratio
  python3 split_train_val.py \\
      --input data/sft_compress/sft_train_summary_prefix.jsonl \\
      --train data/sft_compress/sft_train_summary_prefix_train.jsonl \\
      --val data/sft_compress/sft_train_summary_prefix_val.jsonl \\
      --val_ratio 0.15

  # Use different random seed
  python3 split_train_val.py \\
      --input data/sft_compress/sft_train_summary_only.jsonl \\
      --train data/sft_compress/sft_train_summary_only_train.jsonl \\
      --val data/sft_compress/sft_train_summary_only_val.jsonl \\
      --seed 123
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to output training JSONL file"
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to output validation JSONL file"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of validation set (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate val_ratio
    if not 0 < args.val_ratio < 1:
        parser.error("val_ratio must be between 0 and 1")
    
    split_dataset(
        input_file=args.input,
        train_file=args.train,
        val_file=args.val,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

