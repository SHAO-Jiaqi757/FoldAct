#!/bin/bash
################################################################################
# Training Script with Context Length Monitoring and KL-Aware Training
# 
# Key Features:
# - Initialize from checkpoint: verl_checkpoints/sft_progressive/phase3_summary_prefix/global_step_275
# - Enable context window compression (sliding window)
# - Enable KL-aware training (10% full context, 90% compressed)
# - Enable context length monitoring for WandB
################################################################################
bash set_proxy.sh

# GPU Configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# Kill all processes using GPUs to ensure clean state

# Environment Variables
export VERL_LOGGING_LEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1

# PyTorch & CUDA Configuration
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Data Directories
export DATA_DIR='/datapool/data/ASearcher'
export TRAIN_DATA_DIR='/datapool/data/ASearcher'
export TEST_DATA_DIR='/datapool/data/ASearcher'

# WandB Configuration
WAND_PROJECT='search-agent-Context-Monitoring'

# Ray Configuration
# Use user-writable directory instead of /mnt/ray_tmp (permission issues)
RAY_TMPDIR=/mnt/ray_tmp
export RAY_TMPDIR

################################################################################
# MODEL CONFIGURATION - Initialize from Checkpoint
################################################################################

# ⭐ IMPORTANT: Set BASE_MODEL to the checkpoint path
# This checkpoint contains the model weights from SFT phase3
# The directory structure shows it's already in HuggingFace format:
#   - config.json, generation_config.json
#   - model-*.safetensors files
#   - tokenizer files (tokenizer.json, vocab.json, etc.)
export CHECKPOINT_DIR="verl_checkpoints/sft_progressive/phase3_summary_prefix/global_step_350"

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
export EXPERIMENT_NAME=consistency-loss-agent-gae-qwen2.5-3b-it-context-monitoring

################################################################################
# vLLM Configuration
################################################################################
export VLLM_USE_V1=0

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
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    +enable_debug_logs=true \
    data.reward_fn_key=data_source \
    reward_model.reward_manager=hallucination_penalty \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=3500 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=gae \
    \
    `# ========== MODEL CONFIGURATION (from checkpoint) ==========` \
    `# Actor model uses the checkpoint for training` \
    +actor_rollout_ref.actor.model.path=$BASE_MODEL \
    `# Rollout model also resumes from the checkpoint for vLLM generation` \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.trust_remote_code=true \
    \
    `# ========== ACTOR OPTIMIZATION ==========` \
    `# Reduced LR for stability (was 1e-6, now 3e-7 to prevent pg_loss explosion)` \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    `# Stronger gradient clipping to prevent NaN (was default 1.0, now 0.5)` \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    `# Increased KL coef for stability (was 0.001, now 0.01 to stabilize ppo_kl)` \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    `# Tighter clip ratios to prevent large policy updates` \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    `# Increased dual-clip lower bound for negative advantages (was 3.0, now 5.0)` \
    actor_rollout_ref.actor.clip_ratio_c=5.0 \
    \
    `# ========== ACTOR FSDP & BATCH SIZE ==========` \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    +actor_rollout_ref.actor.dtype=float16 \
    \
    `# ========== ROLLOUT CONFIGURATION (vLLM Sync) ==========` \
    ++actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.n_agent=1 \
    env.rollout.n=6 \
    \
    `# ========== LOG PROB & DYNAMIC BATCH SIZE ==========` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    \
    `# ========== REFERENCE MODEL CONFIGURATION ==========` \
    +actor_rollout_ref.ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.ref.dtype=float16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    `# ========== CRITIC MODEL CONFIGURATION ==========` \
    critic.model.path=$BASE_MODEL \
    `# ========== TRAINER CONFIGURATION ==========` \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=402 \
    trainer.save_freq=30 \
    trainer.test_freq=100 \
    trainer.resume_mode=enable \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.enable_experiment_logging=true \
    trainer.experiment_log_dir=logs/paper_experiments \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.log_val_generations=5 \
    \
    `# ========== CHECKPOINT CONTENTS ==========` \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    \
    `# ========== MULTI-TURN GENERATION CONFIGURATION ==========` \
    max_turns=6 \
    algorithm.no_think_rl=false \
    \
    `# ========== RETRIEVER CONFIGURATION ==========` \
    +retriever.url="http://10.200.14.82:5000/retrieve" \
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
    data.filter_overlong_prompts=false \
    data.truncation=right \
    data.shuffle=true \
    \
    `# ========== 🆕 CONTEXT WINDOW & KL-AWARE TRAINING ==========` \
    `# Enable sliding window: keep only the most recent 1 turn` \
    +use_summary=true \
    \
    `# ========== 🚀 OPTIMIZED PER-TURN TRAINING ==========` \
    `# Selective per-turn: only use for final 2 turns (efficiency optimization)` \
    +per_turn_dropout_prob=0.5 \
    \
    `# ========== 🆕 PER-TURN + SUMMARY TRAINING ==========` \
    `# Enable Per-Turn + Summary method (separated log_prob computation)` \
    `# +actor_rollout_ref.actor.use_separated_loss=true (Auto-enabled by use_summary=true)` \
    `# Enable Full Context Supervision for consistency loss` \
    +actor_rollout_ref.actor.use_full_context_supervision=true \
    `# Loss weights: λ for summary, (1-λ) for action, β for consistency` \
    +actor_rollout_ref.actor.summary_loss_weight=1 \
    +actor_rollout_ref.actor.action_loss_weight=1 \
    +actor_rollout_ref.actor.consistency_loss_weight=1 \
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
