'''
    TODO: 实现华为 om 的推理，基于 ais_bench 模块
'''
from .amodel_impl import AModelBackend
from typing import List, Any, Tuple
from ais_bench.infer.inferface import InferSession

class OMBackend(AModelBackend):
    def setup(self, model_path: str, **kwargs):
        self.model_path = model_path
        device_id = kwargs.get("device_id", 0)
        self.infer_mode = kwargs.get("infer_mode", "dymbatch")
        self.sess = InferSession(device_id, model_path)
        self.input_desc = self.sess.get_inputs()
        self.output_desc = self.sess.get_outputs()
    
    def teardown(self):
        self.sess.free_resource()
    
    def get_input_num(self) -> int:
        return len(self.input_desc)
    
    def get_input_shape(self, idx:int) -> tuple[int, ...]:
        return self.input_desc[idx].shape
    
    def get_input_dtype(self, idx:int) -> str:
        return str(self.input_desc[idx].dtype)
    
    def get_output_num(self):
        return len(self.output_desc)
    
    def get_output_shape(self, idx: int) -> Tuple[int, ...]:
        return self.output_desc[idx].shape
    
    def get_output_dtype(self, idx: int) -> str:
        return str(self.output_desc[idx].dtype)
    
    def infer(self, inputs: Tuple[Any]) -> List[Any]:
        return self.sess.infer(
            feeds=list(inputs),
            mode=self.infer_mode,
            out_array=True,
        )