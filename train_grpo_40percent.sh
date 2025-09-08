export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/datapool/data/deepresearcher'
WAND_PROJECT='Search-R1'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-em-40percent-fixed
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em
export TRAIN_DATA_DIR='/datapool/data/deepresearcher'
export TEST_DATA_DIR='/datapool/data/deepresearcher'
# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Not compatible with vLLM memory pool
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_MODELSCOPE=false

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Store checkpoints on /datapool via symlink so path remains stable
TARGET_CKPT_ROOT="/datapool/verl_checkpoints"
SRC_CKPT_ROOT="verl_checkpoints"
mkdir -p "$TARGET_CKPT_ROOT" "$SRC_CKPT_ROOT"

if [ -n "$EXPERIMENT_NAME" ]; then
  SRC_DIR="$SRC_CKPT_ROOT/$EXPERIMENT_NAME"
  DST_DIR="$TARGET_CKPT_ROOT/$EXPERIMENT_NAME"
  # If a real directory exists at source, move its content to /datapool once
  if [ -d "$SRC_DIR" ] && [ ! -L "$SRC_DIR" ]; then
    mkdir -p "$DST_DIR"
    echo "Relocating existing checkpoints to $DST_DIR ..."
    mv "$SRC_DIR"/* "$DST_DIR" 2>/dev/null || true
    rmdir "$SRC_DIR" 2>/dev/null || true
  fi
  # Point source path to /datapool
  ln -sfn "$DST_DIR" "$SRC_DIR"
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train_transformed.parquet \
    data.val_files=$TEST_DATA_DIR/test_transformed.parquet \
    reward_model.reward_manager=simple_dense_feedback \
    data.reward_fn_key=data_source \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    env.rollout.n=5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=1 \
    +actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    ++trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=402 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +max_turns=6 \
    +final_turn_do_search=true \
    +retriever.url="http://10.201.8.114:8000/retrieve" \
    +retriever.num_workers=5 \
    +retriever.rate_limit=120 \
    +retriever.timeout=30 \
    +retriever.enable_global_rate_limit=true \
    +retriever.topk=3 \
    +data.train_data_num=3072 \
    +data.val_data_num=256 \
    +data.max_start_length=1024 \
    +data.max_obs_length=1420 \
    +data.shuffle_train_dataloader=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    +algorithm.no_think_rl=false \
    +actor_rollout_ref.rollout.n_agent=1 \
    2>&1 | tee $EXPERIMENT_NAME.log
