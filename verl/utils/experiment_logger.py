"""
Experiment Logger for paper experiments.

This module provides centralized logging for experiment metrics including:
- Summary distribution statistics
- Stability metrics (consistency loss, KL divergence, etc.)
- Task performance metrics (success rate, rewards, etc.)
- Distribution shift metrics
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Centralized logger for experiment metrics.
    
    Logs metrics to JSON files organized by experiment name and metric type.
    """
    
    def __init__(self, log_dir: str = "logs/paper_experiments", experiment_name: str = "default"):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Base directory for experiment logs
            experiment_name: Name of the experiment (used for subdirectory)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.stability_metrics = []
        self.task_metrics = []
        self.summary_distributions = []
        self.shift_metrics = []
        
        logger.info(f"ExperimentLogger initialized: {self.experiment_dir}")
    
    def log_stability_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log stability metrics (consistency loss, KL divergence, etc.).
        
        Args:
            step: Training step
            metrics: Dictionary of metric names to values
        """
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.stability_metrics.append(entry)
        
        # Save to file periodically (every 10 entries)
        if len(self.stability_metrics) % 10 == 0:
            self._save_metrics("stability_metrics.json", self.stability_metrics)
    
    def log_task_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """
        Log task performance metrics (success rate, rewards, etc.).
        
        Args:
            epoch: Training epoch
            step: Training step
            metrics: Dictionary of metric names to values
        """
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.task_metrics.append(entry)
        
        # Save to file immediately for task metrics
        self._save_metrics("task_metrics.json", self.task_metrics)
    
    def log_summary_distribution(
        self, 
        step: int, 
        summaries: List[str], 
        tokenizer: Any,
        save_full_data: bool = False
    ):
        """
        Log summary distribution statistics.
        
        Args:
            step: Training step
            summaries: List of summary strings
            tokenizer: Tokenizer for computing token-level statistics
            save_full_data: Whether to save full summary texts
        """
        if not summaries:
            return
        
        # Compute statistics
        summary_lengths = []
        summary_token_counts = []
        
        for summary in summaries:
            if summary:
                summary_lengths.append(len(summary))
                if tokenizer is not None:
                    try:
                        tokens = tokenizer.encode(summary, add_special_tokens=False)
                        summary_token_counts.append(len(tokens))
                    except Exception as e:
                        logger.warning(f"Failed to tokenize summary: {e}")
        
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_summaries": len(summaries),
            "avg_length": float(sum(summary_lengths) / len(summary_lengths)) if summary_lengths else 0.0,
            "avg_token_count": float(sum(summary_token_counts) / len(summary_token_counts)) if summary_token_counts else 0.0,
            "min_length": float(min(summary_lengths)) if summary_lengths else 0.0,
            "max_length": float(max(summary_lengths)) if summary_lengths else 0.0,
        }
        
        if save_full_data:
            entry["summaries"] = summaries
        
        self.summary_distributions.append(entry)
        
        # Save to file periodically
        if len(self.summary_distributions) % 10 == 0:
            self._save_metrics("summary_distributions.json", self.summary_distributions)
    
    def log_shift_metrics(
        self, 
        step: int, 
        current_summaries: List[str], 
        tokenizer: Any,
        previous_summaries: Optional[List[str]] = None,
        compute_info_preservation: bool = False,
        full_contexts: Optional[List[str]] = None
    ):
        """
        Log distribution shift metrics.
        
        Args:
            step: Training step
            current_summaries: Current batch of summaries
            tokenizer: Tokenizer for computing statistics
            previous_summaries: Previous batch of summaries (optional, for comparison)
        """
        if not current_summaries:
            return
        
        # Compute current statistics
        current_lengths = [len(s) for s in current_summaries if s]
        current_token_counts = []
        for summary in current_summaries:
            if summary and tokenizer is not None:
                try:
                    tokens = tokenizer.encode(summary, add_special_tokens=False)
                    current_token_counts.append(len(tokens))
                except Exception:
                    pass
        
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "current_avg_length": float(sum(current_lengths) / len(current_lengths)) if current_lengths else 0.0,
            "current_avg_tokens": float(sum(current_token_counts) / len(current_token_counts)) if current_token_counts else 0.0,
        }
        
        # Compare with previous if available
        if previous_summaries:
            prev_lengths = [len(s) for s in previous_summaries if s]
            if prev_lengths and current_lengths:
                entry["length_shift"] = float(
                    (sum(current_lengths) / len(current_lengths)) - 
                    (sum(prev_lengths) / len(prev_lengths))
                )
        
        self.shift_metrics.append(entry)
        
        # Save to file periodically
        if len(self.shift_metrics) % 10 == 0:
            self._save_metrics("shift_metrics.json", self.shift_metrics)
    
    def _save_metrics(self, filename: str, data: List[Dict]):
        """Save metrics to JSON file."""
        filepath = self.experiment_dir / filename
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics to {filepath}: {e}")
    
    def close(self):
        """Close the logger and save all remaining metrics."""
        # Save all metrics
        self._save_metrics("stability_metrics.json", self.stability_metrics)
        self._save_metrics("task_metrics.json", self.task_metrics)
        self._save_metrics("summary_distributions.json", self.summary_distributions)
        self._save_metrics("shift_metrics.json", self.shift_metrics)
        
        logger.info(f"ExperimentLogger closed: {self.experiment_dir}")

