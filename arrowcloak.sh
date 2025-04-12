#!/bin/bash

# 设置 Hugging Face 镜像地址
export HF_ENDPOINT="https://hf-mirror.com"

# 解析输入参数
GPUS="0,1"
DATASET="mnli"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --weight_dir) WEIGHT_DIR="$2"; shift ;;
        --restore_dir) RESTORE_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 执行 Python 脚本
python arrowcloak/arrowcloak.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --weight_dir "$WEIGHT_DIR" \
    --restore_dir "$RESTORE_DIR" \