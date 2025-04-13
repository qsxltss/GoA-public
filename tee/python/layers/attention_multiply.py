import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple
import math

from python.layers.base import SecretLayerBase
from python.sgx_net import TensorLoader
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from pdb import set_trace as st
from python.recover import *
import numpy as np


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SecretAttentionMultiply(SecretLayerBase):
    def __init__(
        self, sid, LayerName, EnclaveMode, 
        freqs_cis_cpu, freqs_cis_gpu,
        model_args, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        recover = False, recover_otp = False,
        recover_tsqp = False, recover_shadownet = False,
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, 
            link_prev, link_next, 
            manually_register_prev, manually_register_next
        )
        self.ForwardFuncName = "AttentionMultiply"
        self.BackwardFuncName = "AttentionMultiply"
        self.model_args = model_args

        self.head_dim = model_args.dim // model_args.n_heads
        self.batch_size = model_args.batch_size
        self.n_local_heads = model_args.n_heads
        
        self.freqs_cis_cpu = freqs_cis_cpu
        self.freqs_cis_gpu = freqs_cis_gpu
        
        self.PrevLayer = []
        self.recover = recover
        self.recover_otp = recover_otp
        self.recover_tsqp = recover_tsqp
        self.recover_shadownet = recover_shadownet
        
        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            self.EnclaveMode = ExecutionModeOptions.CPU

    def init_shape(self):
        assert len(self.PrevLayer) == 3
        output_shape1 = self.PrevLayer[0].get_output_shape()
        output_shape2 = self.PrevLayer[1].get_output_shape()
        output_shape3 = self.PrevLayer[2].get_output_shape()
        if not output_shape1 == output_shape2 == output_shape3:
            print(f"{self.LayerName} input shapes not consistent: ")
            print(self.PrevLayer[0].LayerName, output_shape1)
            print(self.PrevLayer[1].LayerName, output_shape2)
            print(self.PrevLayer[2].LayerName, output_shape3)
        assert output_shape1 == output_shape2 == output_shape3
        self.InputShape = output_shape1
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
        m, n = self.InputShape[1], self.InputShape[2]
        self.X = torch.randn(m, n)
        self.mask_vector1, self.mask_vector2, self.mask_vector3 = torch.randn(n), torch.randn(n), torch.randn(n)
        self.deshuffle1, self.deshuffle2, self.deshuffle3 = np.random.permutation(n), np.random.permutation(n), np.random.permutation(n)

        self.otp1, self.otp2, self.otp3 = torch.randn(1, m, n), torch.randn(1, m, n), torch.randn(1, m, n)
        self.scale1_1, self.scale1_2, self.scale1_3 = torch.randn(1, n), torch.randn(1, n), torch.randn(1, n)
        self.scale2_1, self.scale2_2, self.scale2_3 = torch.randn(1, n), torch.randn(1, n), torch.randn(1, n)
        
    def register_prev_layer(self, layer):
        if layer not in self.PrevLayer:
            self.PrevLayer.append(layer)
        
    def get_output_shape(self):
        return self.OutputShape
    
    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.OutputShape, None),
                            ("input_query", self.InputShape, None),
                            ("input_key", self.InputShape, None),
                            ("input_value", self.InputShape, None)
                            ]
        self.tensor_name_list = NeededTensorNames
        
    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(
                self.get_tag("input_query", remap=False), self.PrevLayer[0].get_tag("output", remap=False)
            )
            gt.link_tags(
                self.get_tag("input_key", remap=False), self.PrevLayer[1].get_tag("output", remap=False)
            )
            gt.link_tags(
                self.get_tag("input_value", remap=False), self.PrevLayer[2].get_tag("output", remap=False)
            )
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(
                self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False)
            )

    def init(self, start_enclave=True):

        #     TensorLoader.init(self, start_enclave)
        # self.ForwardFunc = self.ForwardFunc(
        #     self.vocab_size, self.dim
        # )
        # self.PlainFunc = self.PlainFunc(
        #     self.vocab_size, self.dim
        # )

        TensorLoader.init(self, start_enclave)

    def forward(self):
        
        with NamedTimerInstance(
            f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER
        ):
            #with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Forward Tensor Transfer", verbose_level=VerboseLevel.LAYER):
            self.forward_tensor_transfer()
                
            if self.EnclaveMode == ExecutionModeOptions.CPU:
                xq = self.get_cpu("input_query")
                xk = self.get_cpu("input_key")
                xv = self.get_cpu("input_value")
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                xq = self.get_gpu("input_query")
                xk = self.get_gpu("input_key")
                xv = self.get_gpu("input_value")
            else:
                raise RuntimeError("Attention multiply not implemented for Enclave mode")
            if self.EnclaveMode == ExecutionModeOptions.CPU:
                #with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Recover Process", verbose_level=VerboseLevel.LAYER):
                if self.recover:
                    xq = recover(xq, self.X, self.deshuffle1, self.mask_vector1,  self.otp1, self.scale1_1, self.scale2_1)
                    xk = recover(xk, self.X, self.deshuffle2, self.mask_vector2, self.otp2, self.scale1_2, self.scale2_2)
                    xv = recover(xv, self.X, self.deshuffle3, self.mask_vector3, self.otp3, self.scale1_3, self.scale2_3)
                elif self.recover_otp:
                    xq = recover_otp(xq, self.otp1)
                    xk = recover_otp(xk, self.otp2)
                    xv = recover_otp(xv, self.otp3)
                elif self.recover_tsqp:
                    xq = recover_tsqp(xq, self.otp1, self.scale1_1)
                    xk = recover_tsqp(xk, self.otp2, self.scale1_2)
                    xv = recover_tsqp(xv, self.otp3, self.scale1_3)
                elif self.recover_shadownet:
                    xq = recover_shadownet(xq, self.deshuffle1, self.mask_vector1, self.otp1, self.scale1_1)
                    xk = recover_shadownet(xk, self.deshuffle2, self.mask_vector2, self.otp2, self.scale1_2)
                    xv = recover_shadownet(xv, self.deshuffle3, self.mask_vector3, self.otp3, self.scale1_3)
            bsz, seqlen, _ = xq.shape
            start_pos = 0
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            

            if self.EnclaveMode == ExecutionModeOptions.CPU:
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis_cpu[start_pos : start_pos + seqlen])
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis_gpu[start_pos : start_pos + seqlen])
            else:
                raise RuntimeError("apply_rotary_emb not implemented for Enclave mode")
                
            keys, values = xk, xv
            
            
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            # if mask is not None:
            #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            # st()
            
            
            # output = query
            if self.EnclaveMode == ExecutionModeOptions.CPU:
                self.set_cpu("output", output)
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.set_gpu("output", output)
            else:
                raise RuntimeError("Attention multiply not implemented for Enclave mode")

            # # self.forward_tensor_transfer will make tensors to float type, but embedding needs int/long
            # # manually change tensor type
            # self.set_gpu("input", self.get_gpu("input").type(torch.long))
            # self.set_cpu("input", self.get_cpu("input").type(torch.long))
            
            # # self.requires_grad_on_cpu("input")
            # if self.EnclaveMode == ExecutionModeOptions.Enclave:
            #     raise RuntimeError(
            #         "Embedding in SGX is not checked, recommend to use GPU for embedding"
            #     )
            #     if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.Enclave:
            #         self.transfer_enclave_to_cpu("input")
            #         if torch.sum(self.get_cpu("input").abs()) == 0:
            #             raise RuntimeError(f"{self.LayerName}: SGX input not load")
            #         self.transfer_cpu_to_enclave("input")
            #     self.transfer_enclave_to_cpu("input")
            #     self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            #     self.transfer_cpu_to_enclave("output")
            # elif self.EnclaveMode == ExecutionModeOptions.CPU:
            #     if (
            #         self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and 
            #         torch.sum(self.get_cpu("input").abs()) == 0
            #     ):
            #         raise RuntimeError(f"{self.LayerName}: SGX input not load")
            #     self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            # elif self.EnclaveMode == ExecutionModeOptions.GPU:
            #     if (
            #         self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and 
            #         torch.sum(self.get_gpu("input").abs()) == 0
            #     ):
            #         raise RuntimeError(f"{self.LayerName}: SGX input not load")
            #     self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
            # else:
            #     raise RuntimeError

    def print_connection_info(self):
        input_info = ""
        for input_layer in self.PrevLayer:
            input_info += f"{input_layer.LayerName},"
        print(
            f"{self.LayerName:20} shape{self.InputShape}{' ':30} mode{self.EnclaveMode}{' ':20} input {input_info}      output {self.NextLayer.LayerName}"
        )
        
    def forward_tensor_transfer(self):
        """
        input_query -> 0
        input_key -> 1
        input_value -> 2
        
        Current don't consider attention multiply is in Enclave
        """
        def transfer_one_tensor(name, tensor_id):
            # Enclave to CPU
            if (
                self.PrevLayer[tensor_id] is not None and 
                self.PrevLayer[tensor_id].EnclaveMode is ExecutionModeOptions.Enclave and
                self.EnclaveMode is ExecutionModeOptions.CPU
            ):
                self.transfer_enclave_to_cpu(name)
            
            # Enclave to GPU
            if (
                self.PrevLayer[tensor_id] is not None and 
                self.PrevLayer[tensor_id].EnclaveMode is ExecutionModeOptions.Enclave and
                self.EnclaveMode is ExecutionModeOptions.GPU
            ):
                self.transfer_enclave_to_cpu(name)
                self.transfer_cpu_to_gpu(name)
                
            if (
                self.PrevLayer[tensor_id] is not None and 
                self.PrevLayer[tensor_id].EnclaveMode is ExecutionModeOptions.CPU and
                self.EnclaveMode is ExecutionModeOptions.GPU
            ):
                self.transfer_cpu_to_gpu(name)
       
        transfer_one_tensor("input_query", 0)
        transfer_one_tensor("input_key", 1)
        transfer_one_tensor("input_value", 2)