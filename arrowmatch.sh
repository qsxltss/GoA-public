#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
DATASET="mnli"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowmatch_results"
OBFUS="translinkguard"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --weight_dir) WEIGHT_DIR="$2"; shift ;;
        --restore_dir) RESTORE_DIR="$2"; shift ;;
        --obfus) OBFUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


python code/arrowmatch.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \\
    --weight_dir "$WEIGHT_DIR" \
    --restore_dir "$RESTORE_DIR" \
    --obfus "$OBFUS"