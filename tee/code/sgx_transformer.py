import os
import sys
from pdb import set_trace as st
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import optim, nn
import torch.distributed as dist

from python.common_net import register_layer, register_weight_layer, get_layer_weight, get_layer_input, \
    get_layer_weight_grad, get_layer_output, get_layer_output_grad, get_layer_input_grad
from python.enclave_interfaces import GlobalTensor
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.sgx_net import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_LM import SGXLinearLM
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.identity import SecretIdentityLayer
from python.layers.add import SecretAddLayer
from python.layers.embedding import SecretEmbeddingLayer
from python.layers.rms_norm import SecretRMSNormLayer
from python.layers.attention_multiply import SecretAttentionMultiply
from python.layers.silu import SecretSiLULayer
from python.layers.multiply import SecretMultiplyLayer

from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions

from .model_make_token import make_token


device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class SecretBasicBlock():
    expansion = 1

    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, downsample_layers=None, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBasicBlock, self).__init__()
        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=planes, filter_hw=3, stride=stride, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu1 = SecretReLULayer(sid, f"{name_prefix}.relu1", EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=3, stride=1, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode, 
            manually_register_next=True, link_next=False,
        )
        self.relu2 = SecretReLULayer(sid, f"{name_prefix}.relu2", EnclaveMode=EnclaveMode)

        self.add = SecretAddLayer(
            sid=sid, LayerName=f"{name_prefix}.add", EnclaveMode=EnclaveMode,
            manually_register_prev=True
        )
        self.add.register_prev_layer(self.bn2)
        self.bn2.register_next_layer(self.add)

        layers = [
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2,
        ]
        self.downsample_layers = downsample_layers
        if downsample_layers is not None:
            layers += self.downsample_layers
            self.add.register_prev_layer(self.downsample_layers[-1])
            self.downsample_layers[-1].register_next_layer(self.add)
        layers.append(self.add)
        layers.append(self.relu2)
        self.layers = layers


    def __str__(self):
        info = f"SecretBasicBlock\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu1.LayerName}: {self.relu1}"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        if self.downsample_layers is not None:
            if len(self.downsample_layers) == 1:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}\n"
            elif len(self.downsample_layers) == 2:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}"
                info += f"\t{self.downsample_layers[1].LayerName}: {self.downsample_layers[1]}\n"
        info += f"\t{self.add.LayerName}: {self.add}\n"
        info += f"\t{self.relu2.LayerName}: {self.relu2}"
        return info
    
    def __repr__(self):
        return self.__str__()


class FeedForwardGPT2():
    def __init__(
        self, sid, name_prefix, args: ModelArgs,
        LinearEnclaveMode=ExecutionModeOptions.Enclave,
        NonLinearEnclaveMode=ExecutionModeOptions.Enclave,
        recover_obfus = False, recover_otp = False,
        recover_tsqp = False, recover_shadownet = False,
    ):
        super().__init__()
        
        dim = args.dim
        hidden_dim = 4*args.dim
        
        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        
        self.layers = []
        
        self.w1 = SGXLinearLM(
            sid, f"{name_prefix}.w1", LinearEnclaveMode, 
            self.batch_size, self.max_seq_len,
            hidden_dim, 
            args.dim,
        )
        self.layers.append(self.w1)
        if recover_obfus or recover_otp or recover_tsqp or recover_shadownet:
            self.identity1 = SecretIdentityLayer(
                sid, f"{name_prefix}.recover_identity1",
                NonLinearEnclaveMode,
                recover=recover_obfus, recover_otp=recover_otp,
                recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
            )
            self.layers.append(self.identity1)
        self.relu = SecretReLULayer(
            sid, f"{name_prefix}.relu", NonLinearEnclaveMode, 
            #dim=hidden_dim,
            manually_register_next=False, link_next=False,
        )
        self.layers.append(self.relu)
        
        self.w2 = SGXLinearLM(
            sid, f"{name_prefix}.w2", LinearEnclaveMode, 
            self.batch_size, self.max_seq_len,
            args.dim,
            hidden_dim,
        )
        self.layers.append(self.w2)


class Attention():
    def __init__(
        self, sid, name_prefix, args: ModelArgs,
        freqs_cis_cpu, freqs_cis_gpu,
        prev_layer,
        LinearEnclaveMode=ExecutionModeOptions.Enclave,
        NonLinearEnclaveMode=ExecutionModeOptions.Enclave,
        recover_obfus = False,recover_otp = False,
        recover_tsqp = False, recover_shadownet = False,
    ):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.name = name_prefix
        self.prev_layer = prev_layer

        self.wq = SGXLinearLM(
            sid, f"{name_prefix}.wq", LinearEnclaveMode, 
            self.batch_size, self.max_seq_len,
            args.n_heads * self.head_dim, 
            args.dim,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.wq.register_prev_layer(self.prev_layer)
        self.prev_layer.register_next_layer(self.wq)

        self.wk = SGXLinearLM(
            sid, f"{name_prefix}.wk", LinearEnclaveMode, 
            self.batch_size, self.max_seq_len,
            args.n_heads * self.head_dim, 
            args.dim,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.wk.register_prev_layer(self.prev_layer)
        self.prev_layer.register_next_layer(self.wk)
        
        self.wv = SGXLinearLM(
            sid, f"{name_prefix}.wv", LinearEnclaveMode, 
            self.batch_size, self.max_seq_len,
            args.n_heads * self.head_dim, 
            args.dim,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.wv.register_prev_layer(self.prev_layer)
        self.prev_layer.register_next_layer(self.wv)
        if recover_obfus or recover_otp or recover_tsqp or recover_shadownet:
            self.attention_multiply = SecretAttentionMultiply(
                sid, f"{name_prefix}.multiply", LinearEnclaveMode,
                freqs_cis_cpu, freqs_cis_gpu,
                args, 
                manually_register_prev=True,
                recover=recover_obfus, recover_otp=recover_otp,
                recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
            )
            self.identity1 = SecretIdentityLayer(
                sid, f"{name_prefix}.recover_identity1",
                NonLinearEnclaveMode,
                manually_register_prev=True,
                manually_register_next=True,
                recover=recover_obfus, recover_otp=recover_otp,
                recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
            )
            self.identity2 = SecretIdentityLayer(
                sid, f"{name_prefix}.recover_identity2",
                NonLinearEnclaveMode,
                manually_register_prev=True,
                manually_register_next=True,
                recover=recover_obfus, recover_otp=recover_otp,
                recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
            )
            self.identity3 = SecretIdentityLayer(
                sid, f"{name_prefix}.recover_identity3",
                NonLinearEnclaveMode,
                manually_register_prev=True,
                manually_register_next=True,
                recover=recover_obfus, recover_otp=recover_otp,
                recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
            )
            self.wq.register_next_layer(self.identity1)
            self.wk.register_next_layer(self.identity2)
            self.wv.register_next_layer(self.identity3)
            self.identity1.register_prev_layer(self.wq)
            self.identity2.register_prev_layer(self.wk)
            self.identity3.register_prev_layer(self.wv)
            self.identity1.register_next_layer(self.attention_multiply)
            self.identity2.register_next_layer(self.attention_multiply)
            self.identity3.register_next_layer(self.attention_multiply)
            self.attention_multiply.register_prev_layer(self.identity1)
            self.attention_multiply.register_prev_layer(self.identity2)
            self.attention_multiply.register_prev_layer(self.identity3)
            self.wo = SGXLinearLM(
                sid, f"{name_prefix}.wo", LinearEnclaveMode, 
                self.batch_size, self.max_seq_len,
                args.n_heads * self.head_dim, 
                args.dim
            )
            self.layers = [
                #self.attention_norm, 
                self.wq, self.wk, self.wv,
                self.identity1, self.identity2, self.identity3,
                self.attention_multiply,
                self.wo
            ]
        else:
            self.attention_multiply = SecretAttentionMultiply(
                sid, f"{name_prefix}.multiply", LinearEnclaveMode,
                freqs_cis_cpu, freqs_cis_gpu,
                args, 
                manually_register_prev=True
            )
            self.wq.register_next_layer(self.attention_multiply)
            self.wk.register_next_layer(self.attention_multiply)
            self.wv.register_next_layer(self.attention_multiply)
            self.attention_multiply.register_prev_layer(self.wq)
            self.attention_multiply.register_prev_layer(self.wk)
            self.attention_multiply.register_prev_layer(self.wv)
        
            self.wo = SGXLinearLM(
                sid, f"{name_prefix}.wo", LinearEnclaveMode, 
                self.batch_size, self.max_seq_len,
                args.n_heads * self.head_dim, 
                args.dim
            )
            
            self.layers = [
                #self.attention_norm, 
                self.wq, self.wk, self.wv,
                self.attention_multiply,
                self.wo
            ]
        

class TransformerBlock():
    def __init__(
        self, sid, layer_id: int, args: ModelArgs,
        freqs_cis_cpu, freqs_cis_gpu,
        LinearEnclaveMode=ExecutionModeOptions.Enclave,
        NonLinearEnclaveMode=ExecutionModeOptions.Enclave,
        recover_obfus = False,
        recover_otp = False,
        recover_tsqp = False,
        recover_shadownet = False,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        
        self.name = f"Block-{layer_id}"
        
        self.layers = []
        
        # 12.17: add layerNorm
        self.pre_attention_identity = SecretIdentityLayer(
            sid, f"{self.name}.pre_atten_identity",
            NonLinearEnclaveMode,
        )
        self.layers.append(self.pre_attention_identity)
        
        self.attention = Attention(
            sid, f"{self.name}.attention", args,
            freqs_cis_cpu, freqs_cis_gpu,
            prev_layer=self.pre_attention_identity,
            LinearEnclaveMode=LinearEnclaveMode,
            NonLinearEnclaveMode=NonLinearEnclaveMode,
            recover_obfus=recover_obfus,
            recover_otp=recover_otp,
            recover_tsqp=recover_tsqp,
            recover_shadownet=recover_shadownet,
        )
        self.layers += self.attention.layers

        self.post_attention_identity = SecretIdentityLayer(
            sid, f"{self.name}.post_atten_identity",
            NonLinearEnclaveMode, recover=recover_obfus, recover_otp=recover_otp,
            recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
        )
        self.layers.append(self.post_attention_identity)
        
        self.attention_add = SecretAddLayer(
            sid, f"{self.name}.attention_add", 
            NonLinearEnclaveMode,
            manually_register_prev=True,
        )
        self.attention_add.register_prev_layer(self.pre_attention_identity)
        self.attention_add.register_prev_layer(self.post_attention_identity)
        self.layers.append(self.attention_add)

        
        self.pre_ffn_identity = SecretIdentityLayer(
            sid=sid, LayerName=f"{self.name}.pre_ffn_identity", 
            EnclaveMode=NonLinearEnclaveMode, recover=recover_obfus, recover_otp=recover_otp,
            recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
        )
        self.layers.append(self.pre_ffn_identity)
        
        self.ffn = FeedForwardGPT2(
            sid, f"{self.name}.FFN", args,
            LinearEnclaveMode=LinearEnclaveMode,
            NonLinearEnclaveMode=NonLinearEnclaveMode,
            recover_obfus=recover_obfus,
            recover_otp=recover_otp,
            recover_tsqp=recover_tsqp,
            recover_shadownet=recover_shadownet,
        )
        self.layers += self.ffn.layers
        
        self.post_ffn_identity = SecretIdentityLayer(
            sid=sid, LayerName=f"{self.name}.post_ffn_identity", EnclaveMode=NonLinearEnclaveMode, 
            recover=recover_obfus, recover_otp=recover_otp,
            recover_tsqp=recover_tsqp, recover_shadownet=recover_shadownet,
        )
        self.layers.append(self.post_ffn_identity)

        # 12.17 加入dropout
        self.ffn_add = SecretAddLayer(
            sid, f"{self.name}.ffn_add", 
            NonLinearEnclaveMode,
            manually_register_prev=True,
        )
        self.ffn_add.register_prev_layer(self.pre_ffn_identity)
        self.ffn_add.register_prev_layer(self.post_ffn_identity)
        self.layers.append(self.ffn_add)
        


class Transformer(nn.Module):
    def __init__(self, 
            params: ModelArgs, 
            token, 
            sid=0, 
            LinearEnclaveMode=ExecutionModeOptions.Enclave,
            NonLinearEnclaveMode=ExecutionModeOptions.Enclave,
            recover_obfus = False,
            recover_otp = False,
            recover_tsqp = False,
            recover_shadownet = False,
        ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.recover_obfus = recover_obfus
        self.recover_otp = recover_otp
        self.recover_tsqp = recover_tsqp
        self.recover_shadownet = recover_shadownet
        
        self.freqs_cis_cpu = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.freqs_cis_gpu = self.freqs_cis_cpu.cuda()

        self.input_shape = list(token.size())

        params.max_seq_len = self.input_shape[1]
        
        
        self.input_layer = SecretInputLayer(
            sid, "InputLayer", self.input_shape, ExecutionModeOptions.CPU
        )
        self.input_layer.StoreInEnclave = False
        
        self.tok_embeddings = SecretEmbeddingLayer(
            sid, "TokenEmbedding", ExecutionModeOptions.GPU,
            vocab_size=params.vocab_size, dim=params.dim
        )
        
        self.sgx_layers = [self.input_layer, self.tok_embeddings]
        
        self.blocks, self.block_layers = [], []
        
        for block_id in range(params.n_layers):
            self.blocks.append(
                TransformerBlock(
                    sid, block_id, params,
                    self.freqs_cis_cpu, self.freqs_cis_gpu,
                    LinearEnclaveMode,
                    NonLinearEnclaveMode,
                    recover_obfus=self.recover_obfus,
                    recover_otp=self.recover_otp,
                    recover_tsqp=self.recover_tsqp,
                    recover_shadownet=self.recover_shadownet,
                )
            )
            self.sgx_layers += self.blocks[block_id].layers
            

        self.output_layer = SecretOutputLayer(
            sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True
        )
        self.sgx_layers.append(self.output_layer)

        


if __name__=="__main__":    

    model_args: ModelArgs = ModelArgs(
        max_seq_len=128,
        max_batch_size=4,
        dim=4096,
        multiple_of=256,
        n_heads=32,
        n_layers=2,
        norm_eps=1e-5,
        vocab_size=32000,
    )
    
    input_tokens = make_token().cpu()
    model_args.batch_size = input_tokens.shape[0]

    GlobalTensor.init()
    resnet_constructor = Transformer(
        model_args,
        LinearEnclaveMode=ExecutionModeOptions.GPU,
        NonLinearEnclaveMode=ExecutionModeOptions.Enclave,
        token=input_tokens,
    )
    layers = resnet_constructor.sgx_layers

    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    

    for layer in layers:
        layer.print_connection_info()
    layers[0].print_tensor_link_relation()

    layers[0].set_input(input_tokens)

    secret_nn.forward()
