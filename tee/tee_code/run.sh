#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

python -m tee_code.run --mode "gpu+enclave" \
    --protect_method "arrowcloak" \
    --model "gpt2-base" \
