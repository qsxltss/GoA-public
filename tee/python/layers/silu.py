import torch
import torch.nn as nn
import torch.nn.functional as F

from python.layers.base import SecretLayerBase
from python.enclave_interfaces import GlobalTensor as gt
from python.sgx_net import TensorLoader
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel

from pdb import set_trace as st

class SecretSiLULayer(SecretLayerBase):
    def __init__(
        self, sid, LayerName, EnclaveMode, 
        dim, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, 
            link_prev, link_next, 
            manually_register_prev, manually_register_next
        )
        self.ForwardFuncName = "SiLU"
        self.BackwardFuncName = "SiLU"
        self.dim = dim
        
        self.ForwardFunc = F.silu
        self.PlainFunc = F.silu

        self.NextLayer = []

        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            self.EnclaveMode = ExecutionModeOptions.CPU

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.WeightShape = [self.dim]
        self.HandleShape = self.InputShape
        
    def register_next_layer(self, layer):
        if layer not in self.NextLayer:
            self.NextLayer.append(layer)
        
    def get_output_shape(self):
        return self.OutputShape
    
    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.OutputShape, None),
                            ("input", self.InputShape, None),
                            ("weight", self.WeightShape, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(
                self.get_tag("input", remap=False), 
                self.PrevLayer.get_tag("output", remap=False)
            )
        if self.link_next and self.NextLayer is not None:
            for next_layer in self.NextLayer:
                gt.link_tags(
                    self.get_tag("output", remap=False), 
                    next_layer.get_tag("input", remap=False)
                )

    def init(self, start_enclave=True):
        # self.ForwardFunc = self.ForwardFunc(
        #     self.dim, 
        # )
        # self.PlainFunc = self.PlainFunc(
        #     self.dim, 
        # )

        TensorLoader.init(self, start_enclave)
        
        # if self.EnclaveMode == ExecutionModeOptions.GPU:
        #     self.ForwardFunc.cuda()
        #     self.PlainFunc.cuda()

    def forward(self):
        with NamedTimerInstance(
            f"S{self.sid}: {self.LayerName} {self.EnclaveMode} Forward", verbose_level=VerboseLevel.LAYER
        ):
            self.forward_tensor_transfer()
            
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                raise RuntimeError(
                    "SiLU in SGX is not checked, recommend to use GPU for SiLU"
                )

            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                if (
                    self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and 
                    torch.sum(self.get_cpu("input").abs()) == 0
                ):
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                if (
                    self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and 
                    torch.sum(self.get_gpu("input").abs()) == 0
                ):
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
            else:
                raise RuntimeError

    def print_connection_info(self):
        output_info = ""
        for output_layer in self.NextLayer:
            output_info += f"{output_layer.LayerName},"
        print(
            f"{self.LayerName:20} shape{self.InputShape}{' ':30} mode{self.EnclaveMode}{' ':20} input {self.PrevLayer.LayerName}      output {output_info}"
        )