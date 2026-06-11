#!/bin/bash

# 设置 Hugging Face 镜像（避免网络问题）
export HF_ENDPOINT=https://hf-mirror.com

# 固定其他参数
# 允许通过环境变量覆盖 TASK，默认 SST-2
export TASK=${TASK:-SST-2}
export K=512
export SEED=42
export BS=64
export LR=5e-6         
export EPS=1e-3
export WD=0
export STEP=10000
export EVAL_STEP=10000
export MODEL=roberta-large
export DPZERO_PRIVACY_EPS=6.0
export DPZERO_PRIVACY_DELTA=5e-6

# R2T 自适应裁剪：不再需要手动指定 C 值，trainer.py 内部会并行尝试多个候选阈值
# 此处保留一个占位值（不影响实际训练）
export DPZERO_THRESHOLD=100

# 设置实验标签，包含任务名和时间戳，避免输出目录覆盖
export EXTRA_TAG="R2T"
export TAG="${EXTRA_TAG}-${TASK}-$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Running R2T adaptive clipping experiment on TASK=${TASK}"
echo "Output TAG = ${TAG}"
echo "========================================="

# 调用官方训练脚本
bash examples/dpzero.sh

echo "All experiments completed."