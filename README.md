# Qwen2.5-1.5B PPO Training on GSM8K

Production-ready PPO (Proximal Policy Optimization) training scripts for Qwen2.5-1.5B model on GSM8K math reasoning dataset using verl-agent framework.

## 🎯 Project Overview

This repository contains optimized training scripts that successfully train Qwen2.5-1.5B on GSM8K dataset, achieving **75%+ validation accuracy** within the first 41 training steps.

### Key Features

- ✅ **Torch Version Fix**: Resolves vLLM compatibility issues (Segmentation fault fix)
- ✅ **Disk Space Management**: Prevents checkpoint bloat (saves 171GB disk space)
- ✅ **Production Tested**: Successfully running on NVIDIA A100 80GB
- ✅ **Fast Convergence**: 6.75% → 75.44% accuracy in 47 minutes

## 📊 Training Results

| Metric | Initial | After 41 Steps | Improvement |
|--------|---------|----------------|-------------|
| Validation Accuracy | 6.75% | 75.44% | +68.7% |
| Average Reward | 0.17 | 0.84 | +394% |
| Training Time | - | 47 min | - |

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with 40GB+ VRAM
- CUDA 12.5+
- Python 3.8+
- verl-agent 0.7.0.dev0
- vLLM 0.11.0 (requires torch==2.8.0)

### Installation & Training

```bash
# 1. Upload the training script to your server
scp fix_torch_train_no_checkpoints.sh root@your-server:/root/

# 2. SSH into your server
ssh root@your-server

# 3. Run the training script
chmod +x /root/fix_torch_train_no_checkpoints.sh
bash /root/fix_torch_train_no_checkpoints.sh
```

The script will automatically:
1. Fix torch version compatibility (downgrade to 2.8.0)
2. Install missing dependencies
3. Configure environment variables
4. Start PPO training in background

### Monitor Training

```bash
# View real-time training logs
tail -f /root/training_1.5b.log

# Check validation accuracy
tail -100 /root/training_1.5b.log | grep "val-core"

# Check disk usage
df -h /

# Check GPU utilization
nvidia-smi
```

## 📁 Repository Structure

```
.
├── fix_torch_train_no_checkpoints.sh   # Main training script (recommended)
├── fix_torch_and_train.sh              # Alternative version with checkpoints
├── README.md                            # This file
├── README_CN.md                         # Chinese documentation
├── TRAINING_GUIDE.md                    # Detailed training guide
└── .gitignore                           # Git ignore file
```

## 🔧 Key Configuration

### Model & Dataset

- **Model**: Qwen/Qwen2.5-1.5B-Instruct (1.54B parameters)
- **Dataset**: GSM8K (7,473 train + 1,319 test samples)
- **Algorithm**: PPO with 3-model architecture (Actor, Reference, Critic)

### Training Parameters

```bash
data.train_batch_size=64
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.max_model_len=1024
trainer.total_epochs=10
trainer.save_freq=10000  # Prevents checkpoint bloat
actor_rollout_ref.actor.optim.lr=1e-6
critic.optim.lr=1e-5
```

## 🐛 Troubleshooting

### Issue 1: Segmentation Fault

**Symptom**: Training crashes immediately with "Segmentation fault (core dumped)"

**Cause**: torch 2.9.0 incompatible with vLLM 0.11.0

**Solution**: The script automatically downgrades to torch 2.8.0

### Issue 2: Disk Space Exhausted

**Symptom**: Training crashes with "OSError(28, 'No space left on device')"

**Cause**: verl-agent saves abnormally large checkpoints (29-36GB each)

**Solution**: Script uses `trainer.save_freq=10000` to disable intermediate checkpoints

### Issue 3: CUDA Memory Error

**Symptom**: "CUDA out of memory"

**Solution**: Reduce GPU memory utilization:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

## 📈 Performance Metrics

- **Throughput**: 448-673 tokens/second
- **Time per step**: ~37-69 seconds
- **GPU Memory Usage**: 72.25 GB peak
- **CPU Memory Usage**: 10.3 GB
- **Expected completion time**: ~21 hours for 1,160 steps

## 🔍 Technical Details

### Torch Version Compatibility

The script ensures compatibility between:
- **torch**: 2.8.0+cu126
- **vLLM**: 0.11.0 (strictly requires torch==2.8.0)
- **xformers**: 0.0.32.post1
- **torchvision**: 0.23.0

### Environment Configuration

```bash
unset PYTORCH_CUDA_ALLOC_CONF  # Prevents vLLM memory pool conflict
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.5
export CUDA_VISIBLE_DEVICES=0
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{verl-qwen-gsm8k,
  title={Qwen2.5-1.5B PPO Training on GSM8K},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/YOUR_USERNAME/verl-qwen-gsm8k-training}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- [verl-agent](https://github.com/volcengine/verl) - RL training framework
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - Base language model
- [GSM8K](https://github.com/openai/grade-school-math) - Math reasoning dataset

## 📧 Contact

For questions and issues, please open an issue on GitHub.

---

**Status**: ✅ Production Ready | Last Updated: 2025-11-06
