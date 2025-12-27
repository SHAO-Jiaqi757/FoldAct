#!/bin/bash
# Complete pipeline to prepare multiturn datasets for MultiTurnSFTDataset

set -e

echo "=================================================="
echo "MULTITURN DATASET PREPARATION PIPELINE"
echo "=================================================="

# Configuration
INPUT_FILE="data/sft_compress/sft_train_multiturn_with_summary.jsonl"
OUTPUT_DIR="data/sft_compress"
VAL_RATIO=0.1

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Please ensure the input file exists before running this pipeline."
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Validation ratio: $VAL_RATIO"
echo ""

# Step 1: Create both datasets in one step
echo "Step 1: Creating both summary prefix and summary only datasets..."
python3 examples/data_preprocess/create_prev_turn_context_dataset.py \
    --input_file "$INPUT_FILE" \
    --output_prefix_file "$OUTPUT_DIR/sft_train_summary_prefix.jsonl" \
    --output_only_file "$OUTPUT_DIR/sft_train_summary_only.jsonl"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create datasets"
    exit 1
fi

echo "✓ Both datasets created"

# Step 2: Prepare datasets for MultiTurnSFTDataset
echo ""
echo "Step 2: Preparing datasets for MultiTurnSFTDataset..."
python3 examples/data_preprocess/prepare_multiturn_datasets.py \
    --summary_prefix_file "$OUTPUT_DIR/sft_train_summary_prefix.jsonl" \
    --summary_only_file "$OUTPUT_DIR/sft_train_summary_only.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --val_ratio "$VAL_RATIO"

if [ $? -ne 0 ]; then
    echo "Error: Failed to prepare multiturn datasets"
    exit 1
fi

echo "✓ Multiturn datasets prepared"

# Step 3: Show final structure
echo ""
echo "=================================================="
echo "PIPELINE COMPLETE!"
echo "=================================================="
echo ""
echo "Created files:"
find "$OUTPUT_DIR" -type f -name "*.jsonl" -o -name "*.parquet" | sort

echo ""
echo "=================================================="
echo "USAGE INSTRUCTIONS"
echo "=================================================="
echo ""
echo "For MultiTurnSFTDataset training:"
echo ""
echo "1. Summary Prefix Dataset (Dataset 1):"
echo "   Train: $OUTPUT_DIR/sft_train_summary_prefix_train.parquet"
echo "   Val:   $OUTPUT_DIR/sft_train_summary_prefix_val.parquet"
echo ""
echo "2. Summary Only Dataset (Dataset 2):"
echo "   Train: $OUTPUT_DIR/sft_train_summary_only_train.parquet"
echo "   Val:   $OUTPUT_DIR/sft_train_summary_only_val.parquet"
echo ""
