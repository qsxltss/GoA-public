import os
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
import sys
from pdb import set_trace as st
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import argparse

import torch
from torch import optim, nn
import torch.distributed as dist


from .sgx_transformer import *
from .model import ModelArgs, Tokenizer

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)


def make_my_token(seq_len=100, bsz=1):

    tokenizer_path = os.path.join(current_script_dir, "tokenizer.model")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    prompt = "What do you think of life? I am an optimist. I believe the meaning of life is "
    len_prompt = len(prompt.split())
    repeat_time = seq_len // len_prompt
    
    prompt = prompt * repeat_time
    prompts = [prompt for _ in range(bsz)]

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    
    tokens = torch.tensor(prompt_tokens).to(torch.long).to("cuda")
    tokens = tokens[:, :seq_len]
    
    return tokens


def make_token():
    tokenizer_path = os.path.join(current_script_dir, "tokenizer.model")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompts = [
        "I believe the meaning of life is",
    ]
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    bsz = len(prompt_tokens)
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    # assert max_prompt_len <= model.params.max_seq_len
    assert max_prompt_len <= 128
    max_gen_len = 64
    total_len = min(128, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos, cur_pos = 0, 8
    cur_tokens = tokens[:, prev_pos:cur_pos]
    return cur_tokens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_repeat", default=3, type=int)
    parser.add_argument(
        "--mode", default="all_enclave",
        choices=["all_gpu", "all_cpu", "all_enclave", "gpu+enclave"],
    )
    parser.add_argument("--protect_method", default="arrowcloak", choices=["arrowcloak", "otp", "tsqp", "shadownet", "none"])
    parser.add_argument("--model", default="gpt2-base", choices=["gpt2-base", "gpt2-xl", "vit"])
    args = parser.parse_args()
    return args

if __name__=="__main__":
    ## 对应的conda环境是goten_py10
    args = parse_args()
    
    cpu_num = 1 
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    
    if args.model == "gpt2-base":
        seq_len = 512
        num_layer = 12
        n_heads = 12
        dim = 768
    elif args.model == "gpt2-xl":
        seq_len = 512
        num_layer = 12 #48=12*4
        n_heads = 25
        dim = 1600
    elif args.model == "vit":
        seq_len = 197
        num_layer = 12
        n_heads = 12
        dim = 768
    else:
        raise NotImplementedError
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=seq_len,
        max_batch_size=args.batch_size,
        dim=dim,
        multiple_of=256,
        n_heads=n_heads,
        n_layers=num_layer,
        norm_eps=1e-5,
        vocab_size=50257,
    )

    input_tokens = make_my_token(
        seq_len=seq_len, 
        bsz=1
    ).cpu()

    model_args.batch_size = input_tokens.shape[0]
    
    if args.mode == "all_gpu":
        linear_mode = ExecutionModeOptions.GPU
        nonlinear_mode = ExecutionModeOptions.GPU
        #lora_mode = ExecutionModeOptions.GPU
    elif args.mode == "all_enclave":
        linear_mode = ExecutionModeOptions.Enclave
        nonlinear_mode = ExecutionModeOptions.Enclave
        #lora_mode = ExecutionModeOptions.Enclave
    elif args.mode == "gpu+enclave":
        linear_mode = ExecutionModeOptions.GPU
        nonlinear_mode = ExecutionModeOptions.Enclave
        #lora_mode = ExecutionModeOptions.Enclave
    else:
        raise NotImplementedError

    recover_obfus, recover_otp, recover_tsqp, recover_shadownet = False, False, False, False
    if args.protect_method == "arrowcloak":
        print("Using Arrowcloak")
        recover_obfus = True
    elif args.protect_method == "otp":
        print("Using OTP")
        recover_otp = True
    elif args.protect_method == "tsqp":
        print("Using TSQP")
        recover_tsqp = True
    elif args.protect_method == "shadownet":
        print("Using Shadownet")
        recover_shadownet = True
        

    GlobalTensor.init()
    model_constructor = Transformer(
        model_args, 
        LinearEnclaveMode=linear_mode,
        NonLinearEnclaveMode=nonlinear_mode,
        token=input_tokens,
        ## 使用arrowcloak保护
        recover_obfus = recover_obfus,
        ## 使用otp保护
        recover_otp =  recover_otp,
        ## 使用tsqp保护
        recover_tsqp = recover_tsqp,
        ## 使用shadownet保护
        recover_shadownet = recover_shadownet,
    )
    layers = model_constructor.sgx_layers
    
    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    

    for layer in layers:
        layer.print_connection_info()
    layers[0].print_tensor_link_relation()
    
    
    print("======== After This Line is the Logged Time ======")

    layers[0].set_input(input_tokens)

    for i in range(args.num_repeat):
        secret_nn.forward()
        