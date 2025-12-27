#!/bin/bash

set -e

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CUDA_FUSER_DISABLE_FALLBACK=1
export TORCH_DISTRIBUTED_USE_COALESCED_COLLECTIVES=0
export NCCL_ASYNC_ERROR_HANDLING=1


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

# Base path and model configuration
base_save_path="/path/to/sft_web_compress_progressive"
MODEL="/path/to/Qwen2.5-7B-Instruct"
export MODEL

nproc_per_node=2  # Recommended to set this according to your actual GPU count

# =============================
# Phase 1: Multi-turn SFT
# =============================
phase1_path="${base_save_path}/phase1_multiturn"
mkdir -p "$phase1_path"
TRAIN_FILE="$PROJECT_ROOT/data/sft_web_compress/sft_train_multiturn.parquet"
VAL_FILE="$PROJECT_ROOT/data/sft_web_compress/sft_train_multiturn.parquet"

echo "Phase 1: Multi-turn SFT"
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    data.max_length=20480 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=true \
    +model.fsdp_config.reshard_after_forward=true \
    +model.fsdp_config.fsdp_size=8 \
    ++model.torch_dtype=float16 \
    ++model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$phase1_path \
    trainer.project_name=search_agent_web_compress_phase1_multiturn \
    trainer.experiment_name=progressive_multiturn_${MODEL##*/} \
    trainer.logger=['console'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine

phase1_ckpt=$(ls -td ${phase1_path}/global_step_* 2>/dev/null | head -1)
echo "Phase 1 checkpoint: $phase1_ckpt"


# =============================
# Phase 2: Summary Only
# =============================
phase2_path="${base_save_path}/phase2_summary_only"
mkdir -p "$phase2_path"
TRAIN_FILES="$PROJECT_ROOT/data/sft_web_compress/sft_train_summary_only_train.parquet"
VAL_FILES="$PROJECT_ROOT/data/sft_web_compress/sft_train_summary_only_val.parquet"

echo "Phase 2: Summary Only"
bash examples/sft/context_summary/train_summary_only.sh $nproc_per_node $phase2_path \
    model.partial_pretrain=$phase1_ckpt \
    trainer.total_epochs=5 \
    optim.lr=1e-5

phase2_ckpt=$(ls -td ${phase2_path}/global_step_* 2>/dev/null | head -1)
echo "Phase 2 checkpoint: $phase2_ckpt"

# =============================
# Phase 3: Prev-Turn Context
# =============================
phase3_path="${base_save_path}/phase3_summary_prefix"
mkdir -p "$phase3_path"
TRAIN_FILES="$PROJECT_ROOT/data/sft_web_compress/sft_train_summary_prefix_train.parquet"
VAL_FILES="$PROJECT_ROOT/data/sft_web_compress/sft_train_summary_prefix_val.parquet"

echo "Phase 3: Prev-Turn Context"
bash examples/sft/context_summary/train_summary_prefix.sh $nproc_per_node $phase3_path \
    model.partial_pretrain=$phase2_ckpt \
    trainer.total_epochs=5 \
    optim.lr=5e-6

echo ""
echo "============== Training Complete =============="
echo "Phase 1 Checkpoint: $phase1_ckpt"
echo "Phase 2 Checkpoint: $phase2_ckpt"
echo "Phase 3 Checkpoint: $(ls -td ${phase3_path}/global_step_* 2>/dev/null | head -1)"
echo ""
echo "Three-stage training pipeline:"
echo "  1. Multi-turn full-chain reasoning"
echo "  2. Summary Only efficient reasoning"
echo "  3. Contextual reasoning based on historical summaries"
echo ""
echo "web_compress dataset path: data/sft_web_compress/"
echo "It is recommended to use the final Phase 3 model for inference or subsequent RL"

