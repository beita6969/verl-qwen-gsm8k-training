#!/usr/bin/env python3
"""
PPO Training Script for Qwen2.5-1.5B on GSM8K
Uses verl-agent framework for reinforcement learning
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf


def setup_environment(config: Dict[str, Any]):
    """Setup environment variables for training"""
    env_config = config.get('environment', {})

    # CUDA configuration
    if 'cuda_visible_devices' in env_config:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(env_config['cuda_visible_devices'])

    if 'cuda_home' in env_config:
        os.environ['CUDA_HOME'] = env_config['cuda_home']

    if 'ld_library_path' in env_config:
        os.environ['LD_LIBRARY_PATH'] = env_config['ld_library_path']

    # Unset conflicting environment variables
    if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
        del os.environ['PYTORCH_CUDA_ALLOC_CONF']

    # Set random seed
    seed = env_config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"✓ Environment configured")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - PyTorch version: {torch.__version__}")


def build_verl_config(config: Dict[str, Any]) -> OmegaConf:
    """Build verl-agent configuration from YAML config"""

    # Convert to OmegaConf for compatibility with verl-agent
    verl_config = OmegaConf.create({
        'data': {
            'train_files': [config['data']['train_files']],
            'val_files': [config['data']['val_files']],
            'train_batch_size': config['data']['train_batch_size'],
            'max_prompt_length': config['data']['max_prompt_length'],
            'max_response_length': config['data']['max_response_length'],
            'filter_overlong_prompts': config['data']['filter_overlong_prompts'],
        },
        'actor_rollout_ref': {
            'model': {
                'path': config['model']['name'],
            },
            'actor': {
                'optim': {
                    'lr': config['ppo']['actor']['learning_rate']
                },
                'ppo_mini_batch_size': config['ppo']['actor']['ppo_mini_batch_size'],
                'ppo_micro_batch_size_per_gpu': config['ppo']['actor']['ppo_micro_batch_size_per_gpu'],
            },
            'rollout': {
                'name': config['ppo']['rollout']['name'],
                'tensor_model_parallel_size': config['ppo']['rollout']['tensor_model_parallel_size'],
                'gpu_memory_utilization': config['ppo']['rollout']['gpu_memory_utilization'],
                'max_model_len': config['ppo']['rollout']['max_model_len'],
                'log_prob_micro_batch_size_per_gpu': config['ppo']['rollout']['log_prob_micro_batch_size_per_gpu'],
            }
        },
        'critic': {
            'model': {
                'path': config['model']['name'],
            },
            'optim': {
                'lr': config['ppo']['critic']['learning_rate']
            },
            'ppo_micro_batch_size_per_gpu': config['ppo']['critic']['ppo_micro_batch_size_per_gpu'],
        },
        'trainer': {
            'logger': config['trainer']['logger'],
            'project_name': config['trainer']['project_name'],
            'experiment_name': config['trainer']['experiment_name'],
            'n_gpus_per_node': config['trainer']['n_gpus_per_node'],
            'nnodes': config['trainer']['nnodes'],
            'save_freq': config['trainer']['save_freq'],
            'test_freq': config['trainer']['test_freq'],
            'total_epochs': config['trainer']['total_epochs'],
            'default_local_dir': config['trainer']['default_local_dir'],
        }
    })

    return verl_config


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-1.5B with PPO on GSM8K")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("Qwen2.5-1.5B PPO Training on GSM8K")
    print("=" * 70)
    print(f"\nLoading configuration from: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    setup_environment(config)

    # Build verl configuration
    verl_config = build_verl_config(config)

    print("\n" + "=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: GSM8K")
    print(f"Training samples: {config['data']['train_files']}")
    print(f"Validation samples: {config['data']['val_files']}")
    print(f"Batch size: {config['data']['train_batch_size']}")
    print(f"Total epochs: {config['trainer']['total_epochs']}")
    print(f"Actor LR: {config['ppo']['actor']['learning_rate']}")
    print(f"Critic LR: {config['ppo']['critic']['learning_rate']}")
    print(f"Checkpoint dir: {config['trainer']['default_local_dir']}")
    print("=" * 70)

    # Import and run verl-agent training
    try:
        from verl.trainer import main_ppo

        print("\nStarting PPO training...")
        print("This will take several hours depending on your hardware.\n")

        # Run training with verl-agent
        main_ppo.main(verl_config)

    except ImportError as e:
        print(f"\n❌ Error: verl-agent not found!")
        print(f"Please install verl-agent: pip install verl-agent==0.7.0.dev0")
        print(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
