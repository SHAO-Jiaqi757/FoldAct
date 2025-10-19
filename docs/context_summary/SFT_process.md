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
    --output_format jsonl \
    --think_drop_prob 0.3
```


### Dataset with Summary


**Script**: `examples/data_preprocess/add_summary_multiturn.py`

**Usage**:
```bash
# 使用OpenAI API添加summary（需要API key）
python3 examples/data_preprocess/add_summary_multiturn.py \
    --input_file data/sft_compress/sft_train_multiturn.jsonl \
    --output_file data/sft_compress/sft_train_multiturn_with_summary.jsonl \
    --max_concurrent 100
```


**Summary Components**:
1. **Question**: Clearly state what needs to be answered
2. **Reasoning Summary**: Brief description of logical thinking process  
3. **Search Results Summary**: Key information from search results (verbatim quotes)



3. Dataset Splitting: Create two specialized datasets for different training objectives.

```
┌─────────────────────┬──────────────────────────────────────────────┬──────────────────────────┐
│ Feature             │ Dataset 1 (Prev Turn)                        │ Dataset 2 (Summary Only) │
├─────────────────────┼──────────────────────────────────────────────┼──────────────────────────┤
│ Prompt Source       │ Question + previous summary/action + latest obs │ Summary only           │
│ Answer Source       │ Summary + Original answer                     │ Original answer          │
│ Avg Token Length    │ ~1700 tokens                                  │ ~1500 tokens             │
│ Learning Objective  │ Summarize & plan the next turn                │ Efficient reasoning      │
│ Training Speed      │ Moderate                                      │ Faster (shorter seq)     │
│ Dataset Format      │ MultiTurnSFTDataset (complete conversations)  │ MultiTurnSFTDataset      │
└─────────────────────┴──────────────────────────────────────────────┴──────────────────────────┘
```

**Usage**:
```bash
bash examples/data_preprocess/run_multiturn_pipeline.sh
```


## 🚀 Training Scripts

### Progressive Training (Three-Phase)

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

Phase 3: Prev-Turn Context (Summary Generation)
├─ Dataset: sft_train_summary_prefix.parquet
├─ Duration: 5 epochs
├─ Learning Rate: 5e-6 (lower to preserve previous learning)
├─ Goal: Add summary generation capability
└─ Output: phase3_summary_prefix/global_step_* (final model)
```


**Usage**:
```bash
# Basic progressive training (8 GPUs)
bash examples/sft/context_summary/train_progressive.sh 8 /tmp/sft_progressive
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

**Manual Evaluation**:
```bash
python3 examples/sft/context_summary/evaluate_model.py \
    --model_path /tmp/sft_mixed/phase2_summary_prefix/global_step_50 \
    --test_file data/sft_compress/sft_train_with_summary.jsonl \
    --output results/eval_results.json \
    --num_samples 50 \
    --max_new_tokens 1024
```

## 🎯 Quick Start

**Complete pipeline from scratch**:

```bash
# 1. Process traces into multiturn format
python3 examples/data_preprocess/process_search_agent_multiturn.py \
    --input_file data/sft_compress/filtered_results_sample_200.jsonl \
    --output_file data/sft_compress/sft_train_multiturn.jsonl \
    --output_format jsonl \
    --think_drop_prob 0.3

# 2. Add summaries to multiturn data (only turn_index >= 1)
python3 examples/data_preprocess/add_summary_multiturn.py \
    --input_file data/sft_compress/sft_train_multiturn.jsonl \
    --output_file data/sft_compress/sft_train_multiturn_with_summary.jsonl \
    --max_concurrent 100

# 3. Create Dataset 1 & 2, Split train/val and convert to Parquet
bash examples/data_preprocess/run_multiturn_pipeline.sh

# 4. Train with progressive strategy (recommended)
bash examples/sft/context_summary/train_progressive.sh 8 verl_checkpoints/sft_progressive

```
