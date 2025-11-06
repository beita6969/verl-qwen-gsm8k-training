#!/usr/bin/env python3
"""
Training Monitor
Real-time monitoring of PPO training progress
"""

import argparse
import time
import re
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta


def parse_training_log(log_file: Path, tail_lines: int = 100):
    """Parse training log and extract metrics"""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Get last N lines
    recent_lines = lines[-tail_lines:] if len(lines) > tail_lines else lines

    metrics = {
        'current_step': None,
        'total_steps': None,
        'validation_accuracy': None,
        'average_reward': None,
        'loss': None,
        'learning_rate': None,
    }

    for line in reversed(recent_lines):
        # Parse validation accuracy
        if 'val-core' in line and metrics['validation_accuracy'] is None:
            match = re.search(r'accuracy[:\s]+([0-9.]+)', line)
            if match:
                metrics['validation_accuracy'] = float(match.group(1))

        # Parse reward
        if 'reward' in line.lower() and metrics['average_reward'] is None:
            match = re.search(r'reward[:\s]+([0-9.]+)', line)
            if match:
                metrics['average_reward'] = float(match.group(1))

        # Parse step info
        if 'step' in line.lower() and metrics['current_step'] is None:
            match = re.search(r'step[:\s]+(\d+)/(\d+)', line)
            if match:
                metrics['current_step'] = int(match.group(1))
                metrics['total_steps'] = int(match.group(2))

    return metrics


def estimate_time_remaining(current_step: int, total_steps: int,
                           time_per_step: float) -> str:
    """Estimate remaining training time"""
    if current_step == 0 or time_per_step == 0:
        return "Unknown"

    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps * time_per_step

    return str(timedelta(seconds=int(remaining_seconds)))


def display_metrics(metrics: dict, start_time: datetime):
    """Display training metrics in a formatted way"""
    print("\n" + "=" * 70)
    print(f"Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if metrics is None:
        print("⚠️  No training data found yet...")
        return

    # Progress
    if metrics['current_step'] and metrics['total_steps']:
        progress = (metrics['current_step'] / metrics['total_steps']) * 100
        print(f"Progress: {metrics['current_step']}/{metrics['total_steps']} "
              f"({progress:.1f}%)")

    # Metrics
    if metrics['validation_accuracy'] is not None:
        print(f"Validation Accuracy: {metrics['validation_accuracy']:.2%}")

    if metrics['average_reward'] is not None:
        print(f"Average Reward: {metrics['average_reward']:.4f}")

    # Time info
    elapsed = datetime.now() - start_time
    print(f"Elapsed Time: {str(elapsed).split('.')[0]}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor PPO training")
    parser.add_argument(
        "--log-file",
        type=str,
        default="/root/training_1.5b.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=10,
        help="Refresh interval in seconds"
    )

    args = parser.parse_args()
    log_file = Path(args.log_file)

    print(f"Monitoring training log: {log_file}")
    print(f"Refresh interval: {args.refresh_interval}s")
    print("Press Ctrl+C to stop monitoring\n")

    start_time = datetime.now()

    try:
        while True:
            metrics = parse_training_log(log_file)
            display_metrics(metrics, start_time)
            time.sleep(args.refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
