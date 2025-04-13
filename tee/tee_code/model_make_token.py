import os 
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json
import time
from pathlib import Path
from typing import Optional

from pdb import set_trace as st

import torch

    
from .model import ModelArgs, Transformer, Tokenizer

def make_token():

    tokenizer_path = os.path.join(current_script_dir, "tokenizer.model")
    
    tokenizer = Tokenizer(model_path=tokenizer_path)


    prompts = [
        "I believe the meaning of life is",
    ]

    # prompts = [
    #     "What do you think of life? I am an optimist. I believe the meaning of life is " +
    #     "What do you think of life? I am an optimist. I believe the meaning of life is " +
    #     "What do you think of life? I am an optimist. I believe the meaning of life is " +
    #     "What do you think of life? I am an optimist. I believe the meaning of life is "
    # ]

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    bsz = len(prompt_tokens)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    # assert max_prompt_len <= model.params.max_seq_len
    assert max_prompt_len <= 128
    max_gen_len = 64
    # total_len = min(model.params.max_seq_len, max_gen_len + max_prompt_len)
    total_len = min(128, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")


    prev_pos, cur_pos = 0, 8
    cur_tokens = tokens[:, prev_pos:cur_pos]
    return cur_tokens




@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


def load_llama(ckpt_dir, tokenizer_path):
    start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

    # ckpt_path = checkpoints[0]
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=128,
        max_batch_size=4,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    # model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    
    return model, tokenizer


if __name__ == "__main__":

    use_lora = False

    tokenizer_path = os.path.join(current_script_dir, "tokenizer.model")
    ckpt_dir = os.path.join(current_script_dir, "tiny-llama-2-7b")
    model, tokenizer = load_llama(ckpt_dir, tokenizer_path)


    if use_lora:
        from peft import inject_adapter_in_model, LoraConfig

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["w1", "w2", "w3"],
        )

        st()
        model = inject_adapter_in_model(lora_config, model)



    prompts = [
        "I believe the meaning of life is",
    ]

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    bsz = len(prompt_tokens)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model.params.max_seq_len
    max_gen_len = 64
    total_len = min(model.params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        
    # st()

    # prev_pos, cur_pos = 0, 8
    # logits = model(tokens[:, prev_pos:cur_pos], start_pos=0)

    cur_tokens = make_token()
    logits = model(cur_tokens, start_pos=0)