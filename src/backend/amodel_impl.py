'''
backend.amodel_impl 的 Docstring
'''

from typing import List, Tuple, Any

class AModelBackend:
    def setup(self, model_path:str, **kwargs):
        pass

    def teardown(self):
        pass

    def get_input_num(self) -> int:
        ## 返回模型的 input tensor 个数
        return -1
    
    def get_input_shape(self, idx:int) -> tuple[int, ...]:
        ## 返回第 idx 个 input tensor 的 shape
        return (-1,)
    
    def get_input_dtype(self, idx:int) -> str:
        ## 返回第 idx 个 input tensor 的 dtype
        return ""

    def get_output_num(self) -> int:
        ## 返回模型的 output tensor 个数
        return -1
    
    def get_output_shape(self, idx:int) -> tuple[int, ...]:
        ## 返回第 idx 个 output tensor 的 shape
        return (-1,)
    
    def get_output_dtype(self, idx:int) -> str:
        ## 返回第 idx 个 output tensor 的 dtype
        return ""    

    def infer(self, inputs:Tuple[Any]) -> List[Any]:
        ## 执行推理，输入 inputs 为一个列表，每个元素都是对应的 input tensor
        ## 输出 outputs 也是一个列表，每个元素都是对应的 output tensor
        raise NotImplementedError("infer not implemented")
