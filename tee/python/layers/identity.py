from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from pdb import set_trace as st
from python.utils.basic_utils import ExecutionModeOptions
import torch.nn as nn
from tee_code.recover import *
import numpy as np
import torch

class SecretIdentityLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        recover=False, recover_otp=False,
        recover_tsqp=False, recover_shadownet=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.Shapefortranspose = None
        self.PrevLayer = []
        self.NextLayer = []
        self.recover = recover
        self.recover_otp = recover_otp
        self.recover_tsqp = recover_tsqp
        self.recover_shadownet = recover_shadownet
    def init_shape(self):
        output_shape1 = self.PrevLayer[0].get_output_shape()
        for layer in self.PrevLayer[1:]:
            output_shape = layer.get_output_shape()
            if not output_shape1 != output_shape:
                print(self.PrevLayer[0].LayerName, output_shape1)
                print(layer.LayerName, output_shape1)
            assert output_shape1 == output_shape
        self.InputShape = output_shape1
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
        m, n = self.InputShape[1], self.InputShape[2]
        self.X = torch.randn(m, n)
        self.mask_vector = torch.randn(n)
        self.deshuffle = np.random.permutation(n)
        self.otp = torch.randn(1, m, n)
        self.scale1 = torch.randn(1, n) 
        self.scale2 = torch.randn(1, n)

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = []
        # for nextlayer in self.NextLayer:
        #     NeededTensorNames.append(
        #         (f"output-{nextlayer.LayerName}", self.OutputShape, None)
        #     )
        # for prevlayer in self.PrevLayer:
        #     NeededTensorNames.append(
        #         (f"input-{prevlayer.LayerName}", self.InputShape, None)
        #     )

        NeededTensorNames = [("output", self.OutputShape, None),
                            ("input", self.InputShape, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            for prevlayer in self.PrevLayer:
                gt.link_tags(
                    self.get_tag("input", remap=False), 
                    prevlayer.get_tag("output", remap=False)
                )
        if self.link_next and self.NextLayer is not None:
            for nextlayer in self.NextLayer:
                gt.link_tags(
                    self.get_tag("output", remap=False), 
                    nextlayer.get_tag("input", remap=False)
                )

    def transfer_to_cpu(self, transfer_tensor="input"):
        if self.PrevLayer is not None and self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_enclave_to_cpu(transfer_tensor)

        if self.PrevLayer is not None and self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_gpu_to_cpu(transfer_tensor)

            
    def transfer_from_cpu(self, transfer_tensor="input"):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_cpu_to_enclave(transfer_tensor)

        if self.EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_cpu_to_gpu(transfer_tensor)

            
    def register_prev_layer(self, layer):
        self.PrevLayer.append(layer)
    
    
    def register_next_layer(self, layer):
        self.NextLayer.append(layer)

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Transfer phase1", verbose_level=VerboseLevel.LAYER):
                self.transfer_to_cpu("input")
                input = self.get_cpu("input")
            if self.recover == True and ("recover" in self.LayerName or "post" in self.LayerName):
                with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Recover", verbose_level=VerboseLevel.LAYER):    
                    output = input.clone()  
                    output = recover(output, self.X, self.deshuffle, self.mask_vector, self.otp, self.scale1, self.scale2)
            elif self.recover_otp == True and ("recover" in self.LayerName or "post" in self.LayerName):
                with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Recover", verbose_level=VerboseLevel.LAYER):    
                    output = input.clone()
                    output = recover_otp(output, self.otp)
            elif self.recover_tsqp == True and ("recover" in self.LayerName or "post" in self.LayerName):
                with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Recover", verbose_level=VerboseLevel.LAYER):    
                    output = input.clone()
                    output = recover_tsqp(output, self.otp, self.scale1)
            elif self.recover_shadownet == True and ("recover" in self.LayerName or "post" in self.LayerName):
                with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Recover", verbose_level=VerboseLevel.LAYER):    
                    output = input.clone()
                    output = recover_shadownet(output, self.deshuffle, self.mask_vector, self.otp, self.scale1)
            elif "pre_ffn" in self.LayerName or "pre_atten" in self.LayerName:
                output = input.clone()
                output = nn.LayerNorm(self.InputShape[-1])(output)
            else:
                output = input.clone()
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Transfer phase2", verbose_level=VerboseLevel.LAYER):
                self.set_cpu("output", output)
                # print("Identity: ", input[0,0,0,:10])
                self.transfer_from_cpu("output")


    def backward(self):
        return super().backward()

    def print_connection_info(self):
        prev_info = ""
        for layer in self.PrevLayer:
            prev_info += f"{layer.LayerName}, "
        next_info = ""
        for layer in self.NextLayer:
            next_info += f"{layer.LayerName}, "
        print(f"{self.LayerName:20} shape{self.InputShape}{' ':30} input {prev_info:20} output {next_info:20}")


