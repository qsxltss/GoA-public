#!/bin/bash

# 设置 Hugging Face 镜像地址
export HF_ENDPOINT="https://hf-mirror.com"

# 解析输入参数
GPUS="0,1"
DATASET="cola"
OUTPUT_DIR="evaluate_results"
OBFUS="none"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --obfus) OBFUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 执行 Python 脚本
python evaluation/evaluate_model.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --obfus "$OBFUS"