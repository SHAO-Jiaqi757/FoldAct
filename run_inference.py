#!/usr/bin/env python3
"""
Script to run inference on HotpotQA and PopQA datasets
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
MODEL_PATH = "verl_checkpoints/consistency-loss-agent-drop0.2-qwen2.5-3b-it-context-monitoring/global_step_390/actor/huggingface"

DATASETS = [
    {
        "name": "HotpotQA",
        "data_path": "/datapool/data/hotpotqa/process_test_clean.parquet",
        "output_path": "results/hotpotqa_inference_results.jsonl"
    },
    {
        "name": "PopQA",
        "data_path": "/datapool/data/popqa/process_test_clean.parquet",
        "output_path": "results/popqa_inference_results.jsonl"
    }
]

# Inference parameters
INFERENCE_ARGS = {
    "max_turns": 5,
    "max_new_tokens": 500,
    "val_batch_size": 1,
    "retriever_url": "http://10.201.8.114:8000/retrieve",
    "top_k": 3,
    "do_search": True,
    "use_sliding_window": True,
    "prompt_template": "zeroshot",
    "enable_debug_logs": False,
    "use_summary": False
}


def run_inference(dataset_config):
    """Run inference for a single dataset"""
    name = dataset_config["name"]
    data_path = dataset_config["data_path"]
    output_path = dataset_config["output_path"]
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running inference on {name} dataset")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Model path: {MODEL_PATH}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable, "-m", "search_r1.llm_agent.inference",
        "--model_path", MODEL_PATH,
        "--data_path", data_path,
        "--output_path", output_path,
        "--max_turns", str(INFERENCE_ARGS["max_turns"]),
        "--max_new_tokens", str(INFERENCE_ARGS["max_new_tokens"]),
        "--val_batch_size", str(INFERENCE_ARGS["val_batch_size"]),
        "--retriever_url", INFERENCE_ARGS["retriever_url"],
        "--top_k", str(INFERENCE_ARGS["top_k"]),
        "--prompt_template", INFERENCE_ARGS["prompt_template"]
    ]
    
    # Add optional flags
    if INFERENCE_ARGS["do_search"]:
        cmd.append("--do_search")
    if INFERENCE_ARGS["use_sliding_window"]:
        cmd.append("--use_sliding_window")
    if INFERENCE_ARGS["enable_debug_logs"]:
        cmd.append("--enable_debug_logs")
    if INFERENCE_ARGS["use_summary"]:
        cmd.append("--use_summary")
    
    # Run inference
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*60}")
        print(f"{name} inference completed successfully!")
        print(f"{'='*60}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"Error running inference on {name}: {e}")
        print(f"{'='*60}\n")
        return False


def main():
    """Main function to run inference on all datasets"""
    print("="*60)
    print("Starting inference pipeline")
    print("="*60)
    
    results = []
    for dataset in DATASETS:
        success = run_inference(dataset)
        results.append((dataset["name"], success))
    
    # Summary
    print("\n" + "="*60)
    print("Inference Pipeline Summary")
    print("="*60)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name}: {status}")
    print("="*60)
    
    # Exit with error if any failed
    if not all(success for _, success in results):
        sys.exit(1)


if __name__ == "__main__":
    main()






