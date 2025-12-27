# FoldAct: Stable Training for Long-Horizon RL with Context Folding

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**FoldAct** is a framework for training long-horizon reinforcement learning agents with context folding, addressing the fundamental non-stationary observation problem that arises when summary actions modify the agent's future observation space.


## Installation


### Setup

1. **Create and activate conda environment**:
```bash
conda create -n verl-agent python=3.10
conda activate verl-agent
```

2. **Install PyTorch** (adjust CUDA version as needed):
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

3. **Install flash-attention**
```bash
pip install flash-attn --no-build-isolation
```

4. **Install dependencies**:
```bash
pip install -e .
```


## Quick Start

### Training with FoldAct

The main training script is `train_grpo_asearcher_consistency.sh`. Key configuration options:

```bash
# Enable FoldAct features
# +actor_rollout_ref.actor.use_separated_loss=true  # (Auto-enabled) Separated loss computation
+actor_rollout_ref.actor.use_full_context_supervision=true  # Consistency loss

# Loss weights
+actor_rollout_ref.actor.summary_loss_weight=1      # for summary loss
+actor_rollout_ref.actor.action_loss_weight=1       # for action loss
+actor_rollout_ref.actor.consistency_loss_weight=1  # for consistency loss

# Context window management
+use_summary=true                              # Enable compression (auto-enables separated loss)
+per_turn_dropout_prob=0.5                     # Selective segment training
```

### Example Training Command

```bash
bash train_grpo_asearcher_consistency.sh
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

