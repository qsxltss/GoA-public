#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

python -m code.run --mode "gpu+enclave" \
    --protect_method "arrowcloak" \
    --model "gpt2-base" \
