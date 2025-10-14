#!/usr/bin/env bash
# Prepare train/val splits for both summary datasets

set -e

# Configuration
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-42}
BASE_DIR="data/sft_compress"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Prepare Train/Val Dataset Splits                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Validation ratio: ${VAL_RATIO} ($(echo "$VAL_RATIO * 100" | bc)%)"
echo "  Random seed: ${SEED}"
echo "  Base directory: ${BASE_DIR}"
echo ""
echo "════════════════════════════════════════════════════════════════"

# Check if input files exist
if [ ! -f "${BASE_DIR}/sft_train_summary_only.jsonl" ]; then
    echo "Error: ${BASE_DIR}/sft_train_summary_only.jsonl not found!"
    echo "Please run the data processing pipeline first."
    exit 1
fi

if [ ! -f "${BASE_DIR}/sft_train_summary_prefix.jsonl" ]; then
    echo "Error: ${BASE_DIR}/sft_train_summary_prefix.jsonl not found!"
    echo "Please run the data processing pipeline first."
    exit 1
fi

# Split Dataset 1: Summary Prefix
echo ""
echo "【Step 1】Splitting Dataset 1 (Summary Prefix)"
echo "════════════════════════════════════════════════════════════════"

python3 examples/data_preprocess/split_train_val.py \
    --input ${BASE_DIR}/sft_train_summary_prefix.jsonl \
    --train ${BASE_DIR}/sft_train_summary_prefix_train.jsonl \
    --val ${BASE_DIR}/sft_train_summary_prefix_val.jsonl \
    --val_ratio ${VAL_RATIO} \
    --seed ${SEED}

echo ""
echo "✓ Dataset 1 split complete!"

# Split Dataset 2: Summary Only
echo ""
echo "【Step 2】Splitting Dataset 2 (Summary Only)"
echo "════════════════════════════════════════════════════════════════"

python3 examples/data_preprocess/split_train_val.py \
    --input ${BASE_DIR}/sft_train_summary_only.jsonl \
    --train ${BASE_DIR}/sft_train_summary_only_train.jsonl \
    --val ${BASE_DIR}/sft_train_summary_only_val.jsonl \
    --val_ratio ${VAL_RATIO} \
    --seed ${SEED}

echo ""
echo "✓ Dataset 2 split complete!"

# Convert to Parquet
echo ""
echo "【Step 3】Converting to Parquet format"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "Converting Dataset 1 (Summary Prefix) - Train..."
python3 examples/data_preprocess/convert_to_parquet.py \
    --input ${BASE_DIR}/sft_train_summary_prefix_train.jsonl \
    --output ${BASE_DIR}/sft_train_summary_prefix_train.parquet

echo ""
echo "Converting Dataset 1 (Summary Prefix) - Val..."
python3 examples/data_preprocess/convert_to_parquet.py \
    --input ${BASE_DIR}/sft_train_summary_prefix_val.jsonl \
    --output ${BASE_DIR}/sft_train_summary_prefix_val.parquet

echo ""
echo "Converting Dataset 2 (Summary Only) - Train..."
python3 examples/data_preprocess/convert_to_parquet.py \
    --input ${BASE_DIR}/sft_train_summary_only_train.jsonl \
    --output ${BASE_DIR}/sft_train_summary_only_train.parquet

echo ""
echo "Converting Dataset 2 (Summary Only) - Val..."
python3 examples/data_preprocess/convert_to_parquet.py \
    --input ${BASE_DIR}/sft_train_summary_only_val.jsonl \
    --output ${BASE_DIR}/sft_train_summary_only_val.parquet

echo ""
echo "✓ All conversions complete!"

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     All Done!                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo ""
echo "Dataset 1 (Summary Prefix):"
echo "  📄 JSONL - Train: ${BASE_DIR}/sft_train_summary_prefix_train.jsonl"
echo "  📄 JSONL - Val:   ${BASE_DIR}/sft_train_summary_prefix_val.jsonl"
echo "  📦 Parquet - Train: ${BASE_DIR}/sft_train_summary_prefix_train.parquet"
echo "  📦 Parquet - Val:   ${BASE_DIR}/sft_train_summary_prefix_val.parquet"
echo ""
echo "Dataset 2 (Summary Only):"
echo "  📄 JSONL - Train: ${BASE_DIR}/sft_train_summary_only_train.jsonl"
echo "  📄 JSONL - Val:   ${BASE_DIR}/sft_train_summary_only_val.jsonl"
echo "  📦 Parquet - Train: ${BASE_DIR}/sft_train_summary_only_train.parquet"
echo "  📦 Parquet - Val:   ${BASE_DIR}/sft_train_summary_only_val.parquet"
echo ""

# Show file sizes
echo "File sizes:"
ls -lh ${BASE_DIR}/sft_train_summary_prefix_train.* 2>/dev/null | awk '{print "  " $9 ": " $5}'
ls -lh ${BASE_DIR}/sft_train_summary_prefix_val.* 2>/dev/null | awk '{print "  " $9 ": " $5}'
ls -lh ${BASE_DIR}/sft_train_summary_only_train.* 2>/dev/null | awk '{print "  " $9 ": " $5}'
ls -lh ${BASE_DIR}/sft_train_summary_only_val.* 2>/dev/null | awk '{print "  " $9 ": " $5}'
echo ""

echo "Ready for training! 🚀"
echo ""


