#!/bin/bash

# Script to run inference on HotpotQA and PopQA datasets
# Model: verl_checkpoints/consistency-loss-agent-drop0.2-qwen2.5-3b-it-context-monitoring/global_step_390/actor/huggingface
# GPU Configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5

MODEL_PATH="verl_checkpoints/consistency-loss-agent-dropout-0.8-qwen2.5-3b-it-context-monitoring/global_step_120/actor/huggingface"

# Run inference for HotpotQA and PopQA using a for loop

declare -A DATASETS
DATASETS["hotpotqa"]="/datapool/data/hotpotqa/process_test.parquet"
DATASETS["popqa"]="/datapool/data/popqa/process_test.parquet"

mkdir -p results
for dataset in "${!DATASETS[@]}"; do
    data_path="${DATASETS[$dataset]}"
    output_path="results/p0.5_${dataset}_inference_results.jsonl"

    echo "=========================================="
    echo "Running inference on ${dataset^} dataset"
    echo "=========================================="

    python -m search_r1.llm_agent.inference \
        --model_path "$MODEL_PATH" \
        --data_path "$data_path" \
        --output_path "$output_path" \
        --max_turns 8 \
        --max_new_tokens 2048 \
        --val_batch_size 32 \
        --retriever_url "http://10.200.14.82:5000/retrieve" \
        --top_k 3 \
        --do_search \
        --use_sliding_window \
        --prompt_template "zeroshot"

    echo ""
    echo "=========================================="
    echo "${dataset^} inference completed"
    echo "=========================================="
    echo ""
done

echo "All inference tasks completed!"
