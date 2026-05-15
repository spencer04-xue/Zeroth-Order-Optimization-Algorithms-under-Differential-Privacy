#!/bin/bash

# 基础设置
TASK="SST-2"           # 你可以换成 SST-5, SNLI 等 [cite: 345, 872]
EPSILON=8.0            # 隐私预算 [cite: 41, 347]
LEARNING_RATE=1e-6     # 论文中微调 RoBERTa 的常用学习率 [cite: 888, 889]

# 定义你想要测试的 C 值
CLIP_VALUES=(0.5 1 2 4)

for C in "${CLIP_VALUES[@]}"
do
    echo "===================================================="
    echo "正在测试截断阈值 C = $C"
    echo "===================================================="

    # 为每个 C 值创建唯一的输出路径
    OUTPUT_DIR="./outputs/${TASK}_epsilon${EPSILON}_C${C}"

    python run_classification.py \
      --model_name_or_path roberta-large \
      --task_name $TASK \
      --do_train \
      --do_eval \
      --max_grad_norm $C \
      --non_private False \
      --epsilon $EPSILON \
      --learning_rate $LEARNING_RATE \
      --output_dir $OUTPUT_DIR \
      --overwrite_output_dir \
      --num_train_epochs 10 \
      --per_device_train_batch_size 64 \
      --seed 42
done