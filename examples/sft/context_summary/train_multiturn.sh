#!/bin/bash
# Train search-agent multi-turn SFT model
# This script trains on the multi-turn conversation format where the model learns to:
# - Generate <think>reasoning</think> and <search>query</search> based on context
# - Process <information>search results</information> from user turns
# - Eventually provide <answer>final answer</answer>

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: train_multiturn.sh <nproc_per_node> <save_path> [other_configs...]"
    echo ""
    echo "Example:"
    echo "  bash train_multiturn.sh 8 /tmp/sft_multiturn"
    echo "  bash train_multiturn.sh 8 /tmp/sft_multiturn trainer.total_epochs=10"
    echo ""
    echo "Arguments:"
    echo "  nproc_per_node: Number of GPUs to use"
    echo "  save_path: Directory to save checkpoints"
    echo "  other_configs: Additional Hydra config overrides (optional)"
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Default model (can be overridden with MODEL env var)
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Data paths
TRAIN_FILE="$PROJECT_ROOT/data/sft_compress/sft_train_multiturn.parquet"
VAL_FILE="$PROJECT_ROOT/data/sft_compress/sft_train_multiturn.parquet"  # Using same for now

echo "================================================"
echo "search-agent Multi-Turn SFT Training"
echo "================================================"
echo "Model: $MODEL"
echo "GPUs: $nproc_per_node"
echo "Save path: $save_path"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "================================================"

# Check if data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    echo "Please run: python3 examples/data_preprocess/process_search_agent_multiturn.py --input_file data/sft_compress/filtered_results_sample_200.jsonl --output_file data/sft_compress/sft_train_multiturn.jsonl --output_format parquet"
    exit 1
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    data.max_length=2048 \
    data.truncation=left \
    data.micro_batch_size=1 \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    ++model.torch_dtype=bfloat16 \
    ++model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=search_agent_multiturn_sft \
    trainer.experiment_name=search_agent_multiturn_${MODEL##*/} \
    trainer.logger=['console'] \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    $@

echo "================================================"
echo "Training completed!"
echo "Checkpoints saved in: $save_path"
echo "================================================"

