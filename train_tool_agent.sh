#!/bin/bash
# Training Script for Tool Call Agent (Non-SLURM version)
# Run directly: bash train_tool_agent.sh

################################################################################
# Training Script for Tool Call Agent
# 
# This script trains an agent that can use multiple tools:
# - search: Web search (Jina API or local retriever)
# - visit: Visit web pages
# - PythonInterpreter: Execute Python code in sandbox
# - google_scholar: Search academic publications
# - parse_file: Parse file contents (placeholder)
#
# Key Features:
# - Multi-turn tool calling with <tool_call> JSON format
# - Context compression (sliding window)
# - Per-turn context tracking for correct credit assignment
# - Event ledger for context-invariant rewards
# - Comprehensive tool processing debugging
#
# Environment Variables Required:
# - JINA_API_KEYS: For Jina search API (optional, falls back to local retriever)
# - SANDBOX_FUSION_ENDPOINT: For Python interpreter (optional, uses default)
################################################################################

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export OPENAI_API_KEY=sk-53PlxjlriDLgjpOzRNf8ySQ4FVWKU7gnF8afQDlPHvOaI98V
export OPENAI_BASE_URL=https://35.aigcbest.top/v1
# OPENAI_BASE=https://api2.aigcbest.top/v1
# OPENAI_API_KEY=sk-lI8zwf1hGdjQIy0XD09439BfEfBd4a55A1Fd53A5527fAd96

USE_REMOTE_OPENAI=1

# Export CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=0,1


# ========== CONFIGURATION ==========
# Model configuration - Initialize from Checkpoint
# ⭐ IMPORTANT: Set CHECKPOINT_DIR to train from a checkpoint
# This checkpoint contains the model weights from previous training phase
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-verl_checkpoints/sft_progressive/web_compress_3B/phase3_summary_prefix/global_step_20}"


# export BASE_MODEL="$CHECKPOINT_DIR"
# Alternative: Use base model directly (uncomment to use)
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"  # Adjust as needed
# BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # For 7B model

# WandB configuration
WAND_PROJECT='tool-call-agent'
# Allow overriding EXPERIMENT_NAME for resuming training
EXPERIMENT_NAME="tool-call-agent"
export EXPERIMENT_NAME

# Checkpoint directory (can be overridden via environment variable)
TARGET_CKPT_ROOT="${TARGET_CKPT_ROOT:-verl_checkpoints}"

# Data Directories
# For sft_web_compress dataset
DATA_BASE_DIR="${DATA_BASE_DIR:-data/sft_web_compress}"

# Tool configuration
TOOL_CONFIG_PATH="verl/trainer/config/tool_config_web_compress.yaml"

# Verify data files exist
TRAIN_FILE="$DATA_BASE_DIR/train.parquet"
VAL_FILE="$DATA_BASE_DIR/test.parquet"

echo "=========================================="
echo "Tool Call Agent Training Configuration"
echo "=========================================="
echo "Base Model/Checkpoint: $BASE_MODEL"
if [ "$BASE_MODEL" = "$CHECKPOINT_DIR" ]; then
    echo "  → Training from checkpoint: $CHECKPOINT_DIR"
else
    echo "  → Training from base model"
fi
echo "WandB Project: $WAND_PROJECT"
echo "Experiment: $EXPERIMENT_NAME"
echo "Tool Config: $TOOL_CONFIG_PATH"
echo "Train Data: $TRAIN_FILE"
echo "Val Data: $VAL_FILE"
echo "=========================================="

# Suppress warnings
export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED=1

# Ray configuration
export RAY_DISABLE_IMPORT_WARNING=1

################################################################################
# Logging Configuration
################################################################################
export HYDRA_FULL_ERROR=1
LOG_TO_FILE=${LOG_TO_FILE:-1}  # Default to logging to file
LOG_DIR=${LOG_DIR:-logs}

if [ "$LOG_TO_FILE" = "1" ]; then
  mkdir -p "$LOG_DIR"
  # Create log file with timestamp (EXPERIMENT_NAME already includes timestamp, so just add one more)
  LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S).log"
  # Prune logs older than 7 days
  find "$LOG_DIR" -type f -name "${EXPERIMENT_NAME}-*.log*" -mtime +7 -delete 2>/dev/null || true

fi

# Start training
echo "[INFO] Starting Tool Call Agent training..."

# Ensure outputs directory exists
mkdir -p outputs
echo "[INFO] Current working directory: $(pwd)"
echo "[INFO] Outputs directory: outputs (permissions: $(ls -ld outputs 2>/dev/null | awk '{print $1, $3, $4}'))"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_BASE_DIR/train.parquet \
    data.val_files=$DATA_BASE_DIR/test.parquet \
    data.reward_fn_key=data_source \
    reward_model.reward_manager=hallucination_penalty \
    `# Batch sizes optimized for tool call agent` \
    data.train_batch_size=16 \
    data.val_batch_size=4 \
    data.max_prompt_length=12000 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=gae \
    \
    `# ========== MODEL CONFIGURATION ==========` \
    `# Actor model uses the base model for training` \
    +actor_rollout_ref.actor.model.path=$BASE_MODEL \
    `# Rollout model also uses the base model for sglang generation` \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.trust_remote_code=true \
    \
    `# ========== ACTOR OPTIMIZATION ==========` \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    `# Dual-clip for negative advantages` \
    actor_rollout_ref.actor.clip_ratio_c=5.0 \
    \
    `# ========== ACTOR FSDP & BATCH SIZE ==========` \
    `# Optimized for L20 GPU (1 node, 8 GPUs total)` \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    `# Max token length for longer tool responses` \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    +actor_rollout_ref.actor.dtype=float16 \
    \
    `# ========== ROLLOUT CONFIGURATION (vLLM Sync) ==========` \
    `# Use vLLM for rollout (same as reference script)` \
    actor_rollout_ref.rollout.name=vllm \
    `# Optimized for L20 GPU (1 node, 8 GPUs total)` \
    ++actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    `# Reduced max_model_len to 8096 to reduce KV cache memory requirements (from 16384)` \
    actor_rollout_ref.rollout.max_model_len=8096 \
    `# Ensure max_num_batched_tokens >= max_model_len to satisfy chunked prefill` \
    actor_rollout_ref.rollout.max_num_batched_tokens=8096 \
    `# GPU memory utilization for H20 GPU (set to 0.7 for optimal performance on clean node)` \
    `# REDUCED to 0.3 to avoid OOM during training (vLLM shares GPU with Actor)` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.n_agent=1 \
    `# CRITICAL: In verl+env, keep actor_rollout_ref.rollout.n=1, use env.rollout.n for GRPO` \
    actor_rollout_ref.rollout.n=1 \
    env.rollout.n=6 \
    \
    `# ========== LOG PROB CONFIGURATION ==========` \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    `# Reduced to match max_model_len=8096 to reduce memory requirements` \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    \
    `# ========== REFERENCE MODEL CONFIGURATION ==========` \
    +actor_rollout_ref.ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.ref.dtype=float16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    `# Reduced to match max_model_len=12288 to reduce memory requirements` \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
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
    trainer.save_freq=40 \
    trainer.test_freq=100 \
    trainer.resume_mode=enable \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$TARGET_CKPT_ROOT/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.log_val_generations=5 \
    trainer.enable_experiment_logging=true \
    `# Full context consistency check interval (set to 1 to check every step)` \
    trainer.full_context_consistency_interval=1 \
    \
    `# ========== CHECKPOINT CONTENTS ==========` \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    \
    `# ========== MULTI-TURN GENERATION CONFIGURATION ==========` \
    max_turns=6 \
    algorithm.no_think_rl=false \
    \
    `# ========== SANDBOX FUSION CONFIGURATION ==========` \
    `# Python interpreter sandbox service` \
    +sandbox_fusion.url="http://10.200.14.82:10080/run_code" \
    \
    `# ========== DATA CONFIGURATION ==========` \
    data.max_start_length=512 \
    `# Increased max_obs_length for tool responses (Jina search returns 5-10x longer than local retriever)` \
    `# Jina responses can be 2000-5000 tokens, so we need larger observation length` \
    data.max_obs_length=4000 \
    `# Reduced max_prompt_length to 8192 to save memory (was 12000)` \
    data.max_prompt_length=8192 \
    data.filter_overlong_prompts=false \
    data.truncation=right \
    data.shuffle=true \
    \
    `# ========== 🆕 PER-TURN + SUMMARY TRAINING (FoldAct) ==========` \
    `# Enable Per-Turn + Summary method (separated log_prob computation)` \
    `# Enable Full Context Supervision for consistency loss` \
    +actor_rollout_ref.actor.use_full_context_supervision=true \
    `# Loss weights: λ for summary, (1-λ) for action, β for consistency` \
    +actor_rollout_ref.actor.summary_loss_weight=1 \
    +actor_rollout_ref.actor.action_loss_weight=1 \
    +actor_rollout_ref.actor.consistency_loss_weight=1 \
    `# Enable summary compression (auto-enables separated loss)` \
    +use_summary=true \
    `# Selective segment training dropout probability` \
    +per_turn_dropout_prob=0.5 \
    \
    `# ========== REWARD MANAGER CONFIGURATION ==========` \
    `# Use hallucination penalty reward manager for tool call agent` \
    data.reward_fn_key=data_source \
    reward_model.reward_manager=hallucination_penalty \
    `# LLM Judge configuration (for web tool correctness checking)` \
    `# Enable LLM as judge for answer quality evaluation (different from local RAG)` \
    +reward_model.reward_kwargs.use_llm_judge=true \
    \
    `# ========== GENERATION CONFIGURATION ==========` \
    `# Maximum model context length (must match actor_rollout_ref.rollout.max_model_len)` \
    `# Reduced to 8096 to reduce KV cache memory requirements (from 16384)` \
    +max_model_len=8096 \
    `# Enable debug logs (use ++ to force add to struct config)` \
    ++enable_debug_logs=true \
    `# Sandbox fusion URL (can be overridden by environment variable)` \
    +sandbox_fusion_url="http://10.200.14.82:10080/run_code" \
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
echo "Checkpoints saved to: $TARGET_CKPT_ROOT/$EXPERIMENT_NAME"
if [ "$LOG_TO_FILE" = "1" ] && [ -n "$LOG_FILE" ]; then
  echo "Log file: $LOG_FILE"
  if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Log file size: $LOG_SIZE ($LOG_LINES lines)"
    if [ "$LOG_LINES" -eq 0 ] || [ "$LOG_SIZE" = "0" ]; then
      echo "⚠️  WARNING: Log file is empty! This may indicate the training command produced no output."
      echo "⚠️  Check if the Python command ran successfully or if there were errors."
    else
      echo "✓ Log file contains output"
    fi
  else
    echo "⚠️  WARNING: Log file was not created: $LOG_FILE"
  fi
fi

