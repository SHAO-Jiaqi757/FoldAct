
# Use A800s only by physical index and align CUDA order to PCIe bus IDs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,6,7
# Kill all processes using GPUs 4,5 and 1,2,6,7 to ensure clean state
echo "[INFO] Killing all processes on GPUs 4,5 and 1,2,6,7..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 4,5,1,2,6,7 | xargs -r kill -9 2>/dev/null || true
sleep 2

# Enable memory logging and debugging
export VERL_LOGGING_LEVEL=WARNING
export CUDA_LAUNCH_BLOCKING=1


# PyTorch 扩展/第三方 JIT 用
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
# FlashInfer 自身的 AOT/JIT 用（空格分隔）
# export FLASHINFER_CUDA_ARCH_LIST="8.0 8.6"
# 让注意力后端用 FlashInfer（尤其 FP8 KV / PageAttention）
# export VLLM_ATTENTION_BACKEND=FLASHINFER
# 开启 FlashInfer 采样内核
# export VLLM_USE_FLASHINFER_SAMPLER=1

export DATA_DIR='/datapool/data/deepresearcher'
# Avoid NCCL peer access attempts between non-P2P GPU pairs
export NCCL_P2P_DISABLE=1
# Disable CUDA IPC SHM to avoid peer access errors on mixed/isolated PCIe topologies
export NCCL_SHM_DISABLE=1
WAND_PROJECT='Search-R1'
export RAY_TMPDIR=/datapool/ray_tmp
# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-40percent
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em
export TRAIN_DATA_DIR='/datapool/data/deepresearcher'
export TEST_DATA_DIR='/datapool/data/deepresearcher'
# set -x
# Use vLLM V1 engine for AsyncLLM; ensure no legacy attention backend is enforced
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Not compatible with vLLM memory pool
export VLLM_USE_MODELSCOPE=false

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Store checkpoints on /datapool via symlink so path remains stable
# Allow override via $TARGET_CKPT_ROOT for testing.
TARGET_CKPT_ROOT="${TARGET_CKPT_ROOT:-/datapool/verl_checkpoints}"
SRC_CKPT_ROOT="verl_checkpoints"

# 确保目标目录存在
mkdir -p "$TARGET_CKPT_ROOT"

# 如果源目录是符号链接就跳过，否则删除再建符号链接
if [ ! -L "$SRC_CKPT_ROOT" ]; then
    rm -rf "$SRC_CKPT_ROOT"
    ln -s "$TARGET_CKPT_ROOT" "$SRC_CKPT_ROOT"
fi


# If a previous HF-format checkpoint exists, prefer it as BASE_MODEL so we can
# resume on a different world_size (e.g., 3090 rollout) without shard mismatch.
CKPT_DIR="${SRC_CKPT_ROOT}/${EXPERIMENT_NAME}"
TRACKER_FILE="${CKPT_DIR}/latest_checkpointed_iteration.txt"
if [ -f "$TRACKER_FILE" ]; then
  LAST_STEP=$(cat "$TRACKER_FILE" 2>/dev/null || echo "")
  if [ -n "$LAST_STEP" ]; then
    HF_DIR="${CKPT_DIR}/global_step_${LAST_STEP}/actor/huggingface"
    if [ -d "$HF_DIR" ]; then
      echo "[INFO] Found HF checkpoint at: $HF_DIR; overriding BASE_MODEL."
      export BASE_MODEL="$HF_DIR"
    else
      echo "[INFO] No HF checkpoint under $HF_DIR; using $BASE_MODEL"
    fi
  fi
fi



# ========= NCCL & 环境变量 =========
###############################################################################
# Distributed and NCCL safety/debug settings (single-node friendly)
###############################################################################
# Prevent accidentally connecting to a remote Ray cluster
unset RAY_ADDRESS

# Clear any stale torch.distributed env from previous multi-node runs
for v in MASTER_ADDR MASTER_PORT NODE_RANK RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE; do
  unset "$v"
done

# Force local rendezvous for rank0 if anything reads these on driver side
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29549}

# NCCL diagnostics and safer defaults for single-node TCP
export NCCL_DEBUG=WARN
# Prefer excluding loopback/docker to let NCCL pick the active NIC automatically
export NCCL_SOCKET_IFNAME='^lo,docker0'
export NCCL_SOCKET_DISABLE_IPV6=1
# Disable IB/RDMA on single-node to avoid cross-host handshakes
export NCCL_IB_DISABLE=1
# Better surfacing of async errors and blocking waits for clear stack traces
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# Optional: richer torch.distributed logs when debugging hangs
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Quick echo for visibility in logs
echo "[Env] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME='^lo,docker0' NCCL_SOCKET_DISABLE_IPV6=1 

nvidia-smi -i 4,5 -c EXCLUSIVE_PROCESS
export HYDRA_FULL_ERROR=1
LOG_TO_FILE=${LOG_TO_FILE:-0}
LOG_DIR=${LOG_DIR:-logs}
if [ "$LOG_TO_FILE" = "1" ]; then
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S).log"
  # Prune logs older than 7 days to control disk usage
  find "$LOG_DIR" -type f -name "${EXPERIMENT_NAME}-*.log*" -mtime +7 -delete 2>/dev/null || true
  echo "[INFO] Logging to $LOG_FILE (toggle with LOG_TO_FILE=0/1)"
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train_transformed.parquet \
    data.val_files=$TEST_DATA_DIR/test_transformed.parquet \
    reward_model.reward_manager=simple_dense_feedback \
    data.reward_fn_key=data_source \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=3584 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=6 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=6 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=search_r1.async_runtime.naive_chat_scheduler.NaiveChatCompletionScheduler \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.external_executor=false \
    actor_rollout_ref.rollout.cuda_visible_devices=[4] \
    +actor_rollout_ref.rollout.dp_size_override=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    env.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=400 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=enable \
    trainer.total_epochs=15 \
    trainer.total_training_steps=402 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    max_turns=6 \
    final_turn_do_search=true \
    +retriever.url="http://10.201.8.114:8000/retrieve" \
    retriever.num_workers=5 \
    retriever.rate_limit=120 \
    retriever.timeout=30 \
    retriever.enable_global_rate_limit=true \
    retriever.topk=3 \
    +data.train_data_num=3072 \
    +data.val_data_num=256 \
    data.max_start_length=512 \
    data.max_obs_length=1900 \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    +data.shuffle_train_dataloader=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=1 \
    2>&1 | { if [ "$LOG_TO_FILE" = "1" ]; then tee -a "$LOG_FILE"; else cat; fi; }
