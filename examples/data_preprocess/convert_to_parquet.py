#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format for training.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def convert_jsonl_to_parquet(input_file: str, output_file: str):
    """Convert JSONL to Parquet."""
    
    print(f"Reading from {input_file}...")
    samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i}: {e}")
    
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No samples to convert!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Save to Parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    print(f"✓ Done! Converted {len(samples)} samples to {output_file}")
    
    # Verify
    df_verify = pd.read_parquet(output_file)
    print(f"✓ Verified: {len(df_verify)} samples in parquet file")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python3 convert_to_parquet.py \\
      --input data/sft_compress/sft_train_summary_prefix.jsonl \\
      --output data/sft_compress/sft_train_summary_prefix.parquet

  # Batch convert
  python3 convert_to_parquet.py \\
      --input data/sft_compress/sft_train_summary_only.jsonl \\
      --output data/sft_compress/sft_train_summary_only.parquet
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Parquet file"
    )
    
    args = parser.parse_args()
    
    try:
        convert_jsonl_to_parquet(args.input, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



