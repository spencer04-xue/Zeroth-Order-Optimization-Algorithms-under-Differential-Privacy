#!/bin/bash

# 设置 Hugging Face 镜像（避免网络问题）
export HF_ENDPOINT=https://hf-mirror.com

# 固定其他参数（与成功训练时一致）
export TASK=SST-2
export K=512
export SEED=42
export BS=64
export LR=5e-6         
export EPS=1e-3
export WD=0
export STEP=800
export EVAL_STEP=10000
export MODEL=roberta-large
export DPZERO_PRIVACY_EPS=6.0
export DPZERO_PRIVACY_DELTA=5e-6

# 定义要测试的 C 值列表
C_VALUES=(100)

for C in "${C_VALUES[@]}"; do
    echo "========================================="
    echo "Running experiment with C = ${C}"
    echo "========================================="
    
    # 设置裁剪阈值
    export DPZERO_THRESHOLD=${C}
    
    # 为每个实验创建不同的输出目录（避免覆盖）
    export TAG="k${K}-${MODEL}-dpzero-ft-c${C}-gradrecord"
    
    # 调用官方训练脚本
    bash examples/dpzero.sh
done

echo "All experiments completed."