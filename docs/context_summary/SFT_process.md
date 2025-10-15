# Agent SFT Data Processing & Training Pipeline

This directory contains the complete pipeline for processing Agent data and training models with summary-enhanced supervised fine-tuning (SFT).

## Overview

The pipeline transforms raw Agent traces into two specialized training datasets that enable models to both generate summaries and perform efficient reasoning.

## 📁 Data Processing Pipeline

### Original Data
- **File**: `filtered_results_sample_200.jsonl`
- **Content**: 200 Agent reasoning traces with multi-step search and reasoning processes

## Data Processing Pipeline

### Process Raw Data into Multi-Turn Format

```bash
python3 examples/data_preprocess/process_search_agent_multiturn.py \
    --input_file data/sft_compress/filtered_results_sample_200.jsonl \
    --output_file data/sft_compress/sft_train_multiturn.jsonl \
    --output_format jsonl
```


### Dataset with Summary

1. Split long reasoning traces into multiple training samples at each `search_result` step.

**Script**: `examples/data_preprocess/process_search_agent_sft.py`


**Logic**:
- For a trace with N `search_result` steps, generates N training samples
- Each sample includes:
  - **Prompt**: Question + reasoning steps + search queries + search results up to step i
  - **Answer**: Next reasoning/search/answer steps until step i+1 (or end)

**Example**:
```
Original trace: Question → reasoning₁ → search₁ → result₁ → reasoning₂ → search₂ → result₂ → answer

Generated samples:
Sample 1: [Question → reasoning₁ → search₁ → result₁] → [reasoning₂ → search₂]
Sample 2: [Question → reasoning₁ → search₁ → result₁ → reasoning₂ → search₂ → result₂] → [answer]
```

**Output**: `sft_train_sample.jsonl` (~1,600 training samples from 200 original traces)

**Usage**:
```bash
python3 examples/data_preprocess/process_search_agent_sft.py \
    --input_file data/sft_compress/filtered_results_sample_200.jsonl \
    --output_file data/sft_compress/sft_train_sample.jsonl
```

2. Summary Generation: Add concise summaries to each training sample using OpenAI API (GPT-4o-mini)

**Script**: `examples/data_preprocess/add_summary_openai.py`

**Summary Components**:
1. **Question**: Clearly state what needs to be answered
2. **Reasoning Summary**: Brief description of logical thinking process
3. **Search Results Summary**: Key information from search results (verbatim quotes)

**Output**: `sft_train_with_summary.jsonl` (each sample now includes a `summary` field)

**Usage**:
```bash
python3 examples/data_preprocess/add_summary_openai.py \
    --input_file data/sft_compress/sft_train_sample.jsonl \
    --output_file data/sft_compress/sft_train_with_summary.jsonl \
    --max_concurrent 100
```

3. Dataset Splitting: Create two specialized datasets for different training objectives.

**Script**: `examples/data_preprocess/split_summary_datasets.py`


**Dataset 1 (Summary Prefix)**: `sft_train_summary_prefix.jsonl`
- **Prompt**: Original full context (question + reasoning + search results)
- **Answer**: **Summary + Original answer**
- **Purpose**: Teach model to generate summaries and perform complete reasoning
- **Size**: 1.6MB → Parquet: 2.9MB (train) + 166KB (val)

**Dataset 2 (Summary Only)**: `sft_train_summary_only.jsonl`
- **Prompt**: **Summary only** (discard original context)
- **Answer**: Original answer (no summary prefix)
- **Purpose**: Teach model to reason efficiently from concise context
- **Size**: 442KB → Parquet: 657KB (train) + 53KB (val)

**Comparison**:
```
┌─────────────────────┬─────────────────────────┬──────────────────────────┐
│ Feature             │ Dataset 1 (Prefix)      │ Dataset 2 (Only)         │
├─────────────────────┼─────────────────────────┼──────────────────────────┤
│ Prompt Source       │ Full original context   │ Summary only             │
│ Answer Source       │ Summary + Original      │ Original answer          │
│ Avg Token Length    │ ~3000 tokens            │ ~1500 tokens             │
│ Learning Objective  │ Summarize + Reason      │ Efficient reasoning      │
│ Training Speed      │ Slower (longer seq)     │ Faster (shorter seq)     │
└─────────────────────┴─────────────────────────┴──────────────────────────┘
```

**Usage**:
```bash
python3 examples/data_preprocess/split_summary_datasets.py \
    --input_file data/sft_compress/sft_train_with_summary.jsonl \
    --output_dataset1 data/sft_compress/sft_train_summary_prefix.jsonl \
    --output_dataset2 data/sft_compress/sft_train_summary_only.jsonl
```

**Split Train/Val and Convert to Parquet**
`examples/data_preprocess/prepare_train_val_datasets.sh` 

**Purpose**: 
1. Split both datasets into train (90%) and validation (10%) sets
2. Convert all splits to Parquet format (required by VERL trainer)

**Quick Usage**:
```bash
# One command to do everything (split + convert)
bash examples/data_preprocess/prepare_train_val_datasets.sh

# Customize validation ratio (e.g., 15%)
VAL_RATIO=0.15 bash examples/data_preprocess/prepare_train_val_datasets.sh
```


## 🚀 Training Scripts


### Strategy 1: Dataset 1 Only (Summary Prefix)

**Script**: `examples/sft/context_summary/train_summary_prefix.sh`

**When to use**: 
- Need model to generate summaries
- Have sufficient GPU memory and training time
- Want single-stage training

**Capabilities**:
- ✅ Generate summaries from full context
- ✅ Perform reasoning with summarization
- ⚠️  Less optimized for concise input reasoning

**Usage**:
```bash
# Basic training (8 GPUs)
bash examples/sft/context_summary/train_summary_prefix.sh 8 /tmp/sft_prefix

# With custom model
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash examples/sft/context_summary/train_summary_prefix.sh 8 /tmp/sft_prefix

# With Hydra overrides
bash examples/sft/context_summary/train_summary_prefix.sh 8 /tmp/sft_prefix \
    trainer.total_epochs=10 \
    optim.lr=1e-5 \
    trainer.logger=['console','wandb']
```

### Strategy 2: Dataset 2 Only (Summary Only)

**Script**: `examples/sft/context_summary/train_summary_only.sh`

**When to use**:
- Quick prototyping or validation
- Limited GPU resources
- Only need reasoning from summaries (no summary generation)

**Capabilities**:
- ✅ Efficient reasoning from concise summaries
- ✅ Faster training (shorter sequences)
- ❌ Cannot generate summaries

**Usage**:
```bash
# Basic training (8 GPUs)
bash examples/sft/context_summary/train_summary_only.sh 8 /tmp/sft_only

# Quick test with fewer epochs
bash examples/sft/context_summary/train_summary_only.sh 8 /tmp/sft_only \
    trainer.total_epochs=3
```

### Strategy 3: Progressive Training (Three-Phase)

**Script**: `examples/sft/context_summary/train_progressive.sh`

**Three-Phase Training**:

```
Phase 1: Multi-turn SFT (Complete Reasoning Chains)
├─ Dataset: sft_train_multiturn.parquet
├─ Duration: 10 epochs
├─ Learning Rate: 1e-5
├─ Goal: Learn complete multi-turn reasoning chains
└─ Output: phase1_multiturn/global_step_*

         ↓ (Use Phase 1 checkpoint as initialization)

Phase 2: Summary Only (Efficient Reasoning)
├─ Dataset: sft_train_summary_only.parquet
├─ Duration: 10 epochs
├─ Learning Rate: 1e-5
├─ Goal: Learn efficient reasoning from summaries
└─ Output: phase2_summary_only/global_step_*

         ↓ (Use Phase 2 checkpoint as initialization)

Phase 3: Summary Prefix (Summary Generation)
├─ Dataset: sft_train_summary_prefix.parquet
├─ Duration: 5 epochs
├─ Learning Rate: 5e-6 (lower to preserve previous learning)
├─ Goal: Add summary generation capability
└─ Output: phase3_summary_prefix/global_step_* (final model)
```

**Final Model Capabilities**:
- ✅ Generate summaries from full context
- ✅ Efficient reasoning from summaries
- ✅ Complete reasoning chains
- ✅ Better generalization (progressive learning)

**Usage**:
```bash
# Basic progressive training (8 GPUs)
bash examples/sft/context_summary/train_progressive.sh 8 /tmp/sft_progressive

# With custom model
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash examples/sft/context_summary/train_progressive.sh 8 /tmp/sft_progressive

# With additional Hydra overrides (apply to all phases)
bash examples/sft/context_summary/train_progressive.sh 8 /tmp/sft_progressive \
    trainer.logger=['console','wandb'] \
    data.max_length=4096

# Dry run (see commands without executing)
DRY_RUN=1 bash examples/sft/context_summary/train_progressive.sh 8 /tmp/test
```

**Training Flow**:
1. Phase 1 trains on `sft_train_multiturn.parquet` (complete reasoning chains)
2. Script automatically finds latest Phase 1 checkpoint
3. Phase 2 uses Phase 1 checkpoint as `model.partial_pretrain` and trains on `sft_train_summary_only.parquet`
4. Script automatically finds latest Phase 2 checkpoint
5. Phase 3 uses Phase 2 checkpoint as `model.partial_pretrain` and trains on `sft_train_summary_prefix.parquet`
6. Final model saved in `phase3_summary_prefix/global_step_*`

**Recommended for evaluation**: Use Phase 3 final checkpoint for inference and downstream tasks.

## 📊 Training Parameters

## 🔍 Evaluation

**Script**: `examples/sft/context_summary/evaluate_model.py`

Evaluate model on two tasks:
1. **Summary-Reasoning**: Full context → Generate summary + reasoning
2. **Reasoning from Summary**: Summary → Generate reasoning/answer

**Quick Evaluation**:
```bash
bash examples/sft/context_summary/quick_evaluate.sh \
    /tmp/sft_mixed/phase2_summary_prefix/global_step_50 \
    20  # number of test samples
```

**Manual Evaluation**:
```bash
python3 examples/sft/context_summary/evaluate_model.py \
    --model_path /tmp/sft_mixed/phase2_summary_prefix/global_step_50 \
    --test_file data/sft_compress/sft_train_with_summary.jsonl \
    --output results/eval_results.json \
    --num_samples 50 \
    --max_new_tokens 1024
```

**Analyze Results**:
```bash
python3 examples/sft/context_summary/analyze_results.py \
    --results_file results/eval_results.json
```

## 📝 File Structure

```
data/sft_compress/
├── filtered_results_sample_200.jsonl          # Original data (200 traces)
├── sft_train_sample.jsonl                     # After Step 1 (~1,600 samples)
├── sft_train_with_summary.jsonl              # After Step 2 (with summaries)
│
├── sft_train_summary_prefix.jsonl            # Dataset 1 (JSONL, 733 samples)
├── sft_train_summary_prefix_train.jsonl      # Dataset 1 train (660 samples)
├── sft_train_summary_prefix_val.jsonl        # Dataset 1 val (73 samples)
├── sft_train_summary_prefix_train.parquet    # Dataset 1 train (Parquet, 2.7MB)
├── sft_train_summary_prefix_val.parquet      # Dataset 1 val (Parquet, 339KB)
│
├── sft_train_summary_only.jsonl              # Dataset 2 (JSONL, 733 samples)
├── sft_train_summary_only_train.jsonl        # Dataset 2 train (660 samples)
├── sft_train_summary_only_val.jsonl          # Dataset 2 val (73 samples)
├── sft_train_summary_only_train.parquet      # Dataset 2 train (Parquet, 613KB)
└── sft_train_summary_only_val.parquet        # Dataset 2 val (Parquet, 90KB)
```

## 🎯 Quick Start

**Complete pipeline from scratch**:

```bash
# 1. Process traces into training samples
python3 examples/data_preprocess/process_search_agent_sft.py \
    --input_file data/sft_compress/filtered_results_sample_200.jsonl \
    --output_file data/sft_compress/sft_train_sample.jsonl

# 2. Add summaries with OpenAI API
python3 examples/data_preprocess/add_summary_openai.py \
    --input_file data/sft_compress/sft_train_sample.jsonl \
    --output_file data/sft_compress/sft_train_with_summary.jsonl \
    --max_concurrent 100

# 3. Split into two datasets
python3 examples/data_preprocess/split_summary_datasets.py \
    --input_file data/sft_compress/sft_train_with_summary.jsonl \
    --output_dataset1 data/sft_compress/sft_train_summary_prefix.jsonl \
    --output_dataset2 data/sft_compress/sft_train_summary_only.jsonl

# 4. Split train/val and convert to Parquet
bash examples/data_preprocess/prepare_train_val_datasets.sh

# 5. Train with progressive strategy (recommended)
bash examples/sft/context_summary/train_progressive.sh 8 verl_checkpoints/sft_progressive

# 6. Evaluate the trained model
bash examples/sft/context_summary/quick_evaluate.sh \
    verl_checkpoints/sft_progressive/phase3_summary_prefix/global_step_50 \
    20
```

## 💡 Tips & Best Practices

1. **Use Progressive Training**: Best balance of efficiency and capability with complete reasoning chains
2. **Phase 3 for Production**: Always use Phase 3 checkpoint for inference
3. **Monitor Training**: Use `trainer.logger=['console','wandb']` for tracking
4. **GPU Memory**: Reduce `data.max_length` if OOM occurs
5. **Reproducibility**: Set `SEED=42` in environment for deterministic training
6. **Evaluation**: Use held-out test set (not training data) for fair evaluation

