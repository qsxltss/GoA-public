#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

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

python code/evaluate_model.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --obfus "$OBFUS"