#!/bin/bash
# 修复Torch版本兼容性并启动训练（无checkpoint版本，防止磁盘填满）
set -e

echo "=========================================="
echo "修复Torch版本兼容性并启动训练"
echo "配置: 禁用训练过程checkpoint保存"
echo "=========================================="
echo ""

# 环境配置
unset PYTORCH_CUDA_ALLOC_CONF
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.5
export CUDA_VISIBLE_DEVICES=0

echo "步骤1: 检查当前版本..."
python3 -c "import torch; print(f'Current torch: {torch.__version__}')" || echo "Torch import failed"
python3 -c "import vllm; print(f'Current vLLM: {vllm.__version__}')" || echo "vLLM import failed"
echo ""

echo "步骤2: 降级Torch到2.8.0 (兼容vLLM)..."
echo "这可能需要2-3分钟..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall -q
echo "✓ Torch 2.8.0 安装完成"
echo ""

echo "步骤3: 验证版本..."
python3 -c "import torch; print(f'✓ Torch: {torch.__version__}')"
python3 -c "import vllm; print(f'✓ vLLM: {vllm.__version__}')"
python3 -c "import xformers; print(f'✓ xformers: {xformers.__version__}')"
echo ""

echo "步骤4: 安装其他依赖..."
pip install tensordict codetiming hydra-core pybind11 pylatexenc trl 'numpy<2.0.0' -q
echo "✓ 依赖安装完成"
echo ""

echo "步骤5: 清理并启动训练..."
pkill -9 -f 'python.*main_ppo' 2>/dev/null || true
ray stop --force 2>&1 || true
sleep 2

cd /root/verl-agent

# 数据路径
TRAIN_DATA="/root/data/gsm8k/train.parquet"
TEST_DATA="/root/data/gsm8k/test.parquet"

echo "启动训练（Qwen2.5-1.5B，无checkpoint）..."
echo "⚠️  训练过程不保存checkpoint，避免磁盘填满"
echo ""

nohup python3 -m verl.trainer.main_ppo \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$TEST_DATA']" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    trainer.logger="['console']" \
    trainer.project_name=verl_gsm8k \
    trainer.experiment_name=qwen2.5_1.5b_no_ckpt \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=1 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=/root/checkpoints_1.5b \
    > /root/training_1.5b.log 2>&1 &

TRAINING_PID=$!
echo "✓ 训练已启动，PID: $TRAINING_PID"
echo "  日志: tail -f /root/training_1.5b.log"
echo ""

echo "等待30秒验证启动..."
sleep 30

if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "✅ 训练进程运行正常!"
    echo ""
    echo "最新日志:"
    tail -30 /root/training_1.5b.log
else
    echo "❌ 训练进程已退出"
    echo "完整日志:"
    cat /root/training_1.5b.log
    exit 1
fi

echo ""
echo "=========================================="
echo "启动完成！"
echo "=========================================="
echo ""
echo "配置说明:"
echo "  - trainer.save_freq=10000 (远大于总步数1160)"
echo "  - 训练过程不会保存checkpoint"
echo "  - 训练结束后会自动保存最终模型"
echo "  - 避免磁盘空间再次填满"
echo ""
echo "监控命令:"
echo "  - 实时日志: tail -f /root/training_1.5b.log"
echo "  - 磁盘使用: watch -n 10 'df -h /'"
echo "  - GPU监控: nvidia-smi"
echo ""
