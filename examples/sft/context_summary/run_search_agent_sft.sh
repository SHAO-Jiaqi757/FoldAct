#!/usr/bin/env bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_search_agent_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Data paths
TRAIN_FILES=${TRAIN_FILES:-$HOME/project/Search_R1/data/sft_compress/sft_train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/project/Search_R1/data/sft_compress/sft_train.parquet}

# Model configuration
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

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
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=64 \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=search-agent-sft \
    trainer.experiment_name=search-agent-sft-$(basename $MODEL) \
    trainer.total_epochs=3 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false $@



