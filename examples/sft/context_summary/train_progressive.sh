#!/usr/bin/env bash
# Progressive three-stage training strategy
# Phase 1: Multi-turn SFT (learning complete reasoning chains)
# Phase 2: Summary Only (starting from Phase 1 checkpoint, learning efficient reasoning)
# Phase 3: Prev-Turn Context (starting from Phase 2 checkpoint, learning summary generation with previous-turn context)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: train_progressive.sh <nproc_per_node> <base_save_path> [extra_hydra_overrides...]"
    echo ""
    echo "Example:"
    echo "  bash train_progressive.sh 5 /tmp/sft_progressive"
    echo "  bash train_progressive.sh 5 /tmp/sft_progressive trainer.logger=['console','wandb']"
    echo ""
    echo "This script runs three training phases:"
    echo "  Phase 1: Multi-turn SFT (full reasoning chains)"
    echo "  Phase 2: Summary Only (efficient reasoning from summaries)"
    echo "  Phase 3: Prev-Turn Context (summary generation grounded on previous turn)"
    exit 1
fi

nproc_per_node=$1
base_save_path=$2

# Allow users to append Hydra overrides
shift 2
EXTRA_ARGS_RAW=("$@")
EXTRA_ENV=()
EXTRA_HYDRA=()
for arg in "${EXTRA_ARGS_RAW[@]}"; do
  if [[ "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    EXTRA_ENV+=("$arg")
  else
    EXTRA_HYDRA+=("$arg")
  fi
done
if [ ${#EXTRA_ENV[@]} -gt 0 ]; then
  echo "Exporting env overrides: ${EXTRA_ENV[*]}"
  for kv in "${EXTRA_ENV[@]}"; do
    export "$kv"
  done
fi

# If CUDA_VISIBLE_DEVICES is not explicitly set, automatically select based on nproc_per_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node-1)))
  export CUDA_VISIBLE_DEVICES
fi

# GPU count check
IFS=',' read -r -a _vis <<< "$CUDA_VISIBLE_DEVICES"
num_visible=${#_vis[@]}
if [ "$num_visible" -lt "$nproc_per_node" ]; then
  echo "Error: nproc_per_node=$nproc_per_node but visible GPUs=$num_visible"
  exit 1
fi

# Model configuration
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
export MODEL

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Progressive Three-Stage Training - Multi-turn + Mixed ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Phase 1: Multi-turn SFT (complete reasoning chains)"
echo "Phase 2: Summary Only (efficient reasoning)"
echo "Phase 3: Prev-Turn Context (summary generation with previous turn context)"
echo ""
echo "Model: $MODEL"
echo "GPUs: $nproc_per_node (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Save path: $base_save_path"
echo ""
echo "════════════════════════════════════════════════════════════════"

# ============================================================================
# Phase 1: Multi-turn SFT Training
# ============================================================================
echo ""
echo "【Phase 1】Starting Training - Multi-turn SFT"
echo "════════════════════════════════════════════════════════════════"

phase1_path="${base_save_path}/phase1_multiturn"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Phase 1 cmd: bash examples/sft/context_summary/train_multiturn.sh $nproc_per_node $phase1_path \\
    model.partial_pretrain=$MODEL \\
    trainer.total_epochs=5 \\
    optim.lr=1e-5 \\
    ${EXTRA_HYDRA[*]}"
else
  # Run multi-turn training directly
  # Note: We can't nest bash script calls here because CUDA_VISIBLE_DEVICES is already set
  # We directly call torchrun
  
  # Get project root
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  
  TRAIN_FILE="$PROJECT_ROOT/data/sft_compress/sft_train_multiturn.parquet"
  VAL_FILE="$PROJECT_ROOT/data/sft_compress/sft_train_multiturn.parquet"
  
  if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Multi-turn training file does not exist: $TRAIN_FILE"
    exit 1
  fi
  
  echo "Multi-turn training data: $TRAIN_FILE"
  
  torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    data.max_length=4096 \
    data.micro_batch_size=24 \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    ++model.torch_dtype=bfloat16 \
    ++model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$phase1_path \
    trainer.project_name=search_agent_progressive_phase1_multiturn \
    trainer.experiment_name=progressive_multiturn_${MODEL##*/} \
    trainer.logger=['console'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    "${EXTRA_HYDRA[@]}"
fi

echo ""
echo "✓ Phase 1 (Multi-turn) completed!"

# Find the latest checkpoint from phase 1
phase1_ckpt=$(ls -td ${phase1_path}/global_step_* 2>/dev/null | head -1)

if [ -z "$phase1_ckpt" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] Will look for the latest Phase 1 checkpoint here"
        phase1_ckpt="<PHASE1_MULTITURN_CKPT>"
    else
        echo "Error: Could not find Phase 1 checkpoint"
        exit 1
    fi
fi

echo "Phase 1 checkpoint: $phase1_ckpt"

# ============================================================================
# Phase 2: Summary Only Training (starting from Phase 1 checkpoint)
# ============================================================================
echo ""
echo "【Phase 2】Starting Training - Summary Only"
echo "════════════════════════════════════════════════════════════════"
echo "Continuing training from checkpoint: $phase1_ckpt"

phase2_path="${base_save_path}/phase2_summary_only"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Phase 2 cmd: bash examples/sft/context_summary/train_summary_only.sh $nproc_per_node $phase2_path \\
    model.partial_pretrain=$phase1_ckpt \\
    trainer.total_epochs=5 \\
    optim.lr=1e-5 \\
    ${EXTRA_HYDRA[*]}"
else
  bash examples/sft/context_summary/train_summary_only.sh $nproc_per_node $phase2_path \
      model.partial_pretrain=$phase1_ckpt \
      trainer.total_epochs=5 \
      optim.lr=1e-5 \
      "${EXTRA_HYDRA[@]}"
fi

echo ""
echo "✓ Phase 2 (Summary Only) completed!"

# Find the latest checkpoint from phase 2
phase2_ckpt=$(ls -td ${phase2_path}/global_step_* 2>/dev/null | head -1)

if [ -z "$phase2_ckpt" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] Will look for the latest Phase 2 checkpoint here"
        phase2_ckpt="<PHASE2_SUMMARY_ONLY_CKPT>"
    else
        echo "Error: Could not find Phase 2 checkpoint"
        exit 1
    fi
fi

echo "Phase 2 checkpoint: $phase2_ckpt"

# ============================================================================
# Phase 3: Prev-Turn Context Training (starting from Phase 2 checkpoint)
# ============================================================================
echo ""
echo "【Phase 3】Starting Fine-tuning - Prev-Turn Context"
echo "════════════════════════════════════════════════════════════════"
echo "Continuing training from checkpoint: $phase2_ckpt"

phase3_path="${base_save_path}/phase3_summary_prefix"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Phase 3 cmd: bash examples/sft/context_summary/train_summary_prefix.sh $nproc_per_node $phase3_path \\
    model.partial_pretrain=$phase2_ckpt \\
    trainer.total_epochs=5 \\
    optim.lr=5e-6 \\
    ${EXTRA_HYDRA[*]}"
  exit 0
else
  bash examples/sft/context_summary/train_summary_prefix.sh $nproc_per_node $phase3_path \
      model.partial_pretrain=$phase2_ckpt \
      trainer.total_epochs=5 \
      optim.lr=5e-6 \
      "${EXTRA_HYDRA[@]}"
fi

echo ""
echo "✓ Phase 3 (Prev-Turn Context) completed!"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Progressive Three-Stage Training Complete!       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Phase 1 Model (Multi-turn): $phase1_path/global_step_*"
echo "Phase 2 Model (Summary Only): $phase2_path/global_step_*"
echo "Phase 3 Model (Prev-Turn Context - Final): $phase3_path/global_step_*"
echo ""
echo "Training Strategy Explanation:"
echo "  Phase 1: Learn complete multi-turn reasoning chains"
echo "  Phase 2: Learn efficient reasoning from summaries"
echo "  Phase 3: Learn summary generation grounded on previous-turn context"
echo ""
echo "Recommended to use the final model from Phase 3 for inference or subsequent RL training"
echo ""
