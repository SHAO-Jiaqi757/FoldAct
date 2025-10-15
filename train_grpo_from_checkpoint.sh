#!/bin/bash
################################################################################
# Training Script with Context Length Monitoring and KL-Aware Training
# 
# Key Features:
# - Initialize from checkpoint: verl_checkpoints/sft_best/phase2_summary_prefix/global_step_110
# - Enable context window compression (sliding window)
# - Enable KL-aware training (10% full context, 90% compressed)
# - Enable context length monitoring for WandB
################################################################################

# GPU Configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

# Kill all processes using GPUs to ensure clean state
echo "[INFO] Killing all processes on target GPUs..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 1,2,5,6,7 | xargs -r kill -9 2>/dev/null || true
sleep 2

# Environment Variables
export VERL_LOGGING_LEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1

# PyTorch & CUDA Configuration
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Data Directories
export DATA_DIR='/datapool/data/deepresearcher'
export TRAIN_DATA_DIR='/datapool/data/deepresearcher'
export TEST_DATA_DIR='/datapool/data/deepresearcher'

# WandB Configuration
WAND_PROJECT='search-agent-Context-Monitoring'

# Ray Configuration
export RAY_TMPDIR=/datapool/ray_tmp

################################################################################
# MODEL CONFIGURATION - Initialize from Checkpoint
################################################################################

# ⭐ IMPORTANT: Set BASE_MODEL to the checkpoint path
# This checkpoint contains the model weights from SFT phase2
# The directory structure shows it's already in HuggingFace format:
#   - config.json, generation_config.json
#   - model-*.safetensors files
#   - tokenizer files (tokenizer.json, vocab.json, etc.)
export CHECKPOINT_DIR="/home/SHAO-Jiaqi757/project/Search_R1/verl_checkpoints/sft_progressive/phase3_summary_prefix/global_step_550"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "[INFO] Please verify the checkpoint path exists"
    exit 1
fi

# Verify it's a valid HuggingFace checkpoint
if [ ! -f "$CHECKPOINT_DIR/config.json" ]; then
    echo "[ERROR] config.json not found in $CHECKPOINT_DIR"
    echo "[INFO] This doesn't appear to be a valid HuggingFace checkpoint"
    exit 1
fi

echo "[INFO] ✓ Found HuggingFace checkpoint at: $CHECKPOINT_DIR"
export BASE_MODEL="$CHECKPOINT_DIR"

# Display checkpoint info
echo "[INFO] Checkpoint contents:"
ls -lh "$CHECKPOINT_DIR" | head -10

# Experiment Name
export EXPERIMENT_NAME=nq-search-agent-grpo-qwen2.5-3b-it-context-monitoring

################################################################################
# vLLM Configuration
################################################################################
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_MODELSCOPE=false

################################################################################
# Checkpoint Storage Configuration
################################################################################
TARGET_CKPT_ROOT="${TARGET_CKPT_ROOT:-/datapool/verl_checkpoints}"
SRC_CKPT_ROOT="verl_checkpoints"

# Ensure target directory exists
mkdir -p "$TARGET_CKPT_ROOT"

# Create symlink if not exists
if [ ! -L "$SRC_CKPT_ROOT" ]; then
    rm -rf "$SRC_CKPT_ROOT"
    ln -s "$TARGET_CKPT_ROOT" "$SRC_CKPT_ROOT"
fi

################################################################################
# NCCL Configuration for Single-Node Training
################################################################################
# Prevent accidentally connecting to a remote Ray cluster
unset RAY_ADDRESS

# Clear any stale torch.distributed env from previous runs
for v in MASTER_ADDR MASTER_PORT NODE_RANK RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE; do
  unset "$v"
done

# Force local rendezvous
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29549}

# NCCL Configuration
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME='^lo,docker0'
export NCCL_SOCKET_DISABLE_IPV6=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Avoid NCCL peer access attempts between non-P2P GPU pairs
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

echo "[Env] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[Env] BASE_MODEL=$BASE_MODEL"
echo "[Env] EXPERIMENT_NAME=$EXPERIMENT_NAME"

################################################################################
# GPU Compute Mode
################################################################################
nvidia-smi -i 4,5 -c EXCLUSIVE_PROCESS

################################################################################
# Logging Configuration
################################################################################
export HYDRA_FULL_ERROR=1
LOG_TO_FILE=${LOG_TO_FILE:-1}  # Default to logging to file
LOG_DIR=${LOG_DIR:-logs}

if [ "$LOG_TO_FILE" = "1" ]; then
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S).log"
  # Prune logs older than 7 days
  find "$LOG_DIR" -type f -name "${EXPERIMENT_NAME}-*.log*" -mtime +7 -delete 2>/dev/null || true
  echo "[INFO] Logging to $LOG_FILE"
fi

################################################################################
# TRAINING SCRIPT
################################################################################

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train_transformed.parquet \
    data.val_files=$TEST_DATA_DIR/test_transformed.parquet \
    data.reward_fn_key=data_source \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=grpo \
    \
    `# ========== MODEL CONFIGURATION (from checkpoint) ==========` \
    `# Actor model uses the checkpoint for training` \
    +actor_rollout_ref.actor.model.path=$BASE_MODEL \
    `# Rollout model uses base HuggingFace model for vLLM compatibility` \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.trust_remote_code=true \
    \
    `# ========== ACTOR OPTIMIZATION ==========` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    \
    `# ========== ACTOR FSDP & BATCH SIZE ==========` \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    \
    `# ========== ROLLOUT CONFIGURATION (vLLM Async) ==========` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=search_r1.async_runtime.naive_chat_scheduler.NaiveChatCompletionScheduler \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.external_executor=false \
    actor_rollout_ref.rollout.cuda_visible_devices=[5] \
    +actor_rollout_ref.rollout.dp_size_override=1 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=16 \
    actor_rollout_ref.rollout.n_agent=1 \
    env.rollout.n=5 \
    \
    `# ========== LOG PROB & DYNAMIC BATCH SIZE ==========` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=6 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    \
    `# ========== REFERENCE MODEL CONFIGURATION ==========` \
    actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# ========== TRAINER CONFIGURATION ==========` \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=402 \
    trainer.save_freq=1 \
    trainer.test_freq=100 \
    trainer.resume_mode=enable \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    \
    `# ========== CHECKPOINT CONTENTS ==========` \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    \
    `# ========== MULTI-TURN GENERATION CONFIGURATION ==========` \
    max_turns=6 \
    final_turn_do_search=true \
    algorithm.no_think_rl=false \
    \
    `# ========== RETRIEVER CONFIGURATION ==========` \
    +retriever.url="http://10.201.8.114:8000/retrieve" \
    retriever.num_workers=5 \
    retriever.rate_limit=120 \
    retriever.timeout=30 \
    retriever.enable_global_rate_limit=true \
    retriever.topk=3 \
    \
    `# ========== DATA CONFIGURATION ==========` \
    +data.train_data_num=3072 \
    +data.val_data_num=256 \
    data.max_start_length=512 \
    data.max_obs_length=1900 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    +data.shuffle_train_dataloader=True \
    \
    `# ========== 🆕 CONTEXT WINDOW & KL-AWARE TRAINING ==========` \
    `# Enable sliding window: keep only the most recent 1 turn` \
    +context_window_turns=1 \
    `# KL-Aware Training: 10% full context, 90% compressed` \
    +full_context_ratio=0.1 \
    `# Enable KL baseline computation for compressed rollouts` \
    +enable_kl_baseline=true \
    \
    2>&1 | { if [ "$LOG_TO_FILE" = "1" ]; then tee -a "$LOG_FILE"; else cat; fi; }

################################################################################
# Post-Training Summary
################################################################################
echo ""
echo "======================================================================="
echo "Training completed!"
echo "======================================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoints saved to: verl_checkpoints/$EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"
