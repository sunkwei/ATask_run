''' AModel 提供基类，封装各种不同的推理卡， 
    提供三个方法对应每个模型数据的:
        预处理阶段
        推理阶段
        后处理阶段

    注意每个方法都有可能在不同的工作线程中调用

    一般来说，每个模型的预处理，后处理逻辑都是相同的，但需要
    根据 backend 转化数据类型，启动的 backend_impl 负责

    根据硬件，决定启动模型数
'''

from .atask import ATask
from .backend.amodel_impl import AModelBackend
from .backend.ort import OnnxruntimeBackend
import logging

logger = logging.getLogger("model")

class AModel:
    def __init__(
        self, 
        name:str,
        model_path:str,
        mid:int,            ## DO_ACT, DO_FACEDET ...
        backend:str="onnxruntime",
        **kwargs,
    ):
        '''
            根据 config/{name}.yaml 配置初始化
        '''
        assert "input" in kwargs and "backend_cfg" in kwargs
        self.__name = name
        self.__mid = mid
        self.__model_path = model_path
        self.__backend = backend
        if backend == "onnxruntime":
            self.__backend_impl = OnnxruntimeBackend()
        else:
            ## TODO: 支持额外的实现
            raise NotImplementedError
        self.__backend_impl.setup(model_path, **kwargs["backend_cfg"])

    def __del__(self):
        if hasattr(self, '__backend_impl'):
            self.__backend_impl.teardown()

    def __repr__(self):
        return f"AModel: {self.__name}, mid:{self.__mid:08x}, path:{self.__model_path}, backend:{self.__backend}\n{self.__backend_impl}"

    def mid(self) -> int:
        return self.__mid
    
    def name(self) -> str:
        return self.__name
    
    def model_path(self) -> str:
        return self.__model_path
    
    def backend(self) -> str:
        return self.__backend
    
    def impl(self) -> AModelBackend:
        return self.__backend_impl

    def preprocess(self, task:ATask):
        pass

    def infer(self, task:ATask):
        pass

    def postprocess(self, task:ATask):
        pass
