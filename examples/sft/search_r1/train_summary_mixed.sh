#!/usr/bin/env bash
# 混合训练策略：先用数据集 2 快速训练，再用数据集 1 精调
# 这种策略能获得最佳效果
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,5,6,7
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: train_summary_mixed.sh <nproc_per_node> <base_save_path> [extra_hydra_overrides...]"
    echo "Example: bash train_summary_mixed.sh 8 /tmp/sft_summary_mixed trainer.logger=['console','wandb'] optim.lr_scheduler=cosine"
    exit 1
fi

nproc_per_node=$1
base_save_path=$2

# 允许用户在 mix 脚本后追加任意 Hydra 覆盖，这些会被同时传递给两个阶段
shift 2
EXTRA_ARGS_RAW=("$@")
# Split env-style VAR=VALUE from real Hydra overrides
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

# 若未显式设置 CUDA_VISIBLE_DEVICES，则按 nproc_per_node 自动选择前 n 张卡
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node-1)))
  export CUDA_VISIBLE_DEVICES
fi

# 基础健壮性检查：可见 GPU 数需与 nproc_per_node 一致
IFS=',' read -r -a _vis <<< "$CUDA_VISIBLE_DEVICES"
num_visible=${#_vis[@]}
if [ "$num_visible" -lt "$nproc_per_node" ]; then
  echo "错误: nproc_per_node=$nproc_per_node 但可见 GPU=$num_visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
  echo "请调整 CUDA_VISIBLE_DEVICES 或 nproc_per_node 保持一致。"
  exit 1
fi

# Model configuration
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
export MODEL

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           混合训练策略 - 两阶段训练                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "阶段 1: 使用数据集 2 (summary only) 快速训练"
echo "阶段 2: 使用数据集 1 (summary prefix) 精调"
echo ""
echo "模型: $MODEL"
echo "保存路径: $base_save_path"
echo ""
echo "════════════════════════════════════════════════════════════════"

# Phase 1: Train with dataset 2 (summary only)
echo ""
echo "【阶段 1】开始训练 - 数据集 2 (Summary Only)"
echo "════════════════════════════════════════════════════════════════"

phase1_path="${base_save_path}/phase1_summary_only"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Phase 1 cmd: bash examples/sft/search_r1/train_summary_only.sh $nproc_per_node $phase1_path \\
    model.partial_pretrain=$MODEL \\
    trainer.total_epochs=10 \\
    optim.lr=1e-5 \\
    ${EXTRA_HYDRA[*]}"
else
  bash examples/sft/search_r1/train_summary_only.sh $nproc_per_node $phase1_path \
      model.partial_pretrain=$MODEL \
      trainer.total_epochs=10 \
      optim.lr=1e-5 \
      "${EXTRA_HYDRA[@]}"
fi

echo ""
echo "✓ 阶段 1 完成！"

# Find the latest checkpoint from phase 1
latest_ckpt=$(ls -td ${phase1_path}/global_step_* 2>/dev/null | head -1)

if [ -z "$latest_ckpt" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] 将在此处查找 Phase 1 最新 checkpoint 并传递给 Phase 2"
        latest_ckpt="<PHASE1_GLOBAL_STEP_DIR>"
    else
        echo "错误: 找不到阶段 1 的 checkpoint"
        exit 1
    fi
fi

echo "阶段 1 checkpoint: $latest_ckpt"

# Phase 2: Fine-tune with dataset 1 (summary prefix)
echo ""
echo "【阶段 2】开始精调 - 数据集 1 (Summary Prefix)"
echo "════════════════════════════════════════════════════════════════"
echo "从 checkpoint 继续训练: $latest_ckpt"

phase2_path="${base_save_path}/phase2_summary_prefix"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Phase 2 cmd: bash examples/sft/search_r1/train_summary_prefix.sh $nproc_per_node $phase2_path \\
    model.partial_pretrain=$latest_ckpt \\
    trainer.total_epochs=5 \\
    optim.lr=5e-6 \\
    ${EXTRA_HYDRA[*]}"
  exit 0
else
  bash examples/sft/search_r1/train_summary_prefix.sh $nproc_per_node $phase2_path \
      model.partial_pretrain=$latest_ckpt \
      trainer.total_epochs=5 \
      optim.lr=5e-6 \
      "${EXTRA_HYDRA[@]}"
fi

echo ""
echo "✓ 阶段 2 完成！"

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   混合训练完成！                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "阶段 1 模型: $phase1_path/global_step_*"
echo "阶段 2 模型 (最终): $phase2_path/global_step_*"
echo ""
echo "推荐使用阶段 2 的最终模型进行推理或 RL 训练"
echo ""
