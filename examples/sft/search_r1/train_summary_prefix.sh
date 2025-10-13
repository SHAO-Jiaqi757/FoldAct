#!/usr/bin/env bash
# 训练脚本 - 数据集 1: Summary 前缀模式
# answer = summary + 原始 answer
# 模型学习: 先总结，再推理回答

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: train_summary_prefix.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "Example: bash train_summary_prefix.sh 8 /tmp/sft_summary_prefix"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Data paths (pre-split files under data/sft_compress)
TRAIN_FILES=${TRAIN_FILES:-data/sft_compress/sft_train_summary_prefix_train.parquet}
VAL_FILES=${VAL_FILES:-data/sft_compress/sft_train_summary_prefix_val.parquet}

if [ ! -f "$TRAIN_FILES" ]; then
  echo "错误: 训练文件不存在: $TRAIN_FILES"
  exit 1
fi
if [ ! -f "$VAL_FILES" ]; then
  echo "错误: 验证文件不存在: $VAL_FILES"
  exit 1
fi

# Model configuration
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}

# Compute batch sizes compatible with dp_size and micro batch
MICRO_BSZ=${MICRO_BSZ:-1}
ACC_STEPS=${ACC_STEPS:-6}
GLOBAL_TBS=$(( MICRO_BSZ * nproc_per_node * ACC_STEPS ))

echo "=================================================="
echo "训练配置 - 数据集 1: Summary 前缀模式"
echo "=================================================="
echo "数据集: answer = summary + 原始 answer"
echo "模型: $MODEL"
echo "训练文件: $TRAIN_FILES"
echo "验证文件: $VAL_FILES"
echo "GPU 数量: $nproc_per_node"
echo "保存路径: $save_path"
echo "计算批量: train_batch_size=$GLOBAL_TBS micro_batch_size_per_gpu=$MICRO_BSZ (acc_steps=$ACC_STEPS)"
echo "=================================================="

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.max_length=4096 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=$MICRO_BSZ \
    data.train_batch_size=$GLOBAL_TBS \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.strategy=${FSDP_STRATEGY:-fsdp2} \
    ++model.torch_dtype=${TORCH_DTYPE:-bfloat16} \
    ++model.attn_implementation=${ATTN_IMPL:-flash_attention_2} \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=$save_path \
    trainer.project_name=search-r1-sft-summary-prefix \
    trainer.experiment_name=summary-prefix-$(basename $MODEL) \
    trainer.total_epochs=5 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false $@

echo ""
echo "=================================================="
echo "训练完成！"
echo "模型保存在: $save_path/global_step_*"
echo "=================================================="

# Optional cleanup: keep only the latest checkpoint to save space
CLEAN_OLD_CKPTS=${CLEAN_OLD_CKPTS:-1}
if [ "$CLEAN_OLD_CKPTS" = "1" ]; then
  latest_dir=$(ls -td "$save_path"/global_step_* 2>/dev/null | head -1)
  if [ -n "$latest_dir" ]; then
    echo "清理旧 checkpoint，仅保留: $latest_dir"
    for d in "$save_path"/global_step_*; do
      [ "$d" = "$latest_dir" ] && continue
      [ -d "$d" ] && rm -rf -- "$d"
    done
  fi
fi
