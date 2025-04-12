#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
DATASET="mnli"
BLACKBOX_DIR="results/blackbox_results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --blackbox_dir) BLACKBOX_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python code/blackbox_test.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --blackbox_dir "$BLACKBOX_DIR" \
    --obfus "$OBFUS"