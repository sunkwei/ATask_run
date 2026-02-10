'''
    实现 bm1688 的后端 ...
'''
from .amodel_impl import AModelBackend
from typing import List, Any, Tuple
import ctypes as C

class BM1688Backend(AModelBackend):
    '''
    TODO: bm1688 backend，使用 ctypes 直接访问 libinfer_bm1688.so 吧
    '''
    def setup(self, model_path: str, **kwargs):
        return super().setup(model_path, **kwargs)
    
    def teardown(self):
        return super().teardown()
    
    def get_input_num(self) -> int:
        return super().get_input_num()
    
    def get_input_shape(self, idx:int) -> tuple[int, ...]:
        return super().get_input_shape(idx)
    
    def get_input_dtype(self, idx:int) -> str:
        return super().get_input_dtype(idx)
    
    def get_output_num(self):
        return super().get_output_num()
    
    def get_output_shape(self, idx:int) -> tuple[int, ...]:
        return super().get_output_shape(idx)
    
    def get_output_dtype(self, idx:int) -> str:
        return super().get_output_dtype(idx)
    
    def infer(self, inputs: Tuple[Any]) -> List[Any]:
        return super().infer(inputs)