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
from .backend.ort import OnnxruntimeBackend
import logging
import numpy as np

logger = logging.getLogger("model")

class AModel:
    def __init__(
        self, 
        name:str,
        model_path:str,
        mid:int,            ## DO_ACT, DO_FACEDET ...
        **kwargs,
    ):
        '''
            根据 config/{name}.yaml 配置初始化
        '''
        assert "input" in kwargs
        self.__name = name
        self.__mid = mid
        self.__model_path = model_path

        backends = kwargs.get("backend", [
            {
                "type": "onnxruntime", 
                "backend_cfg": {
                    "force_cpu": True,
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "providers": ["CPUExecutionProvider"],
                    "providers_cfg": [
                        {}
                    ]
                }
            }
        ])

        ## 实例化所有配置的 backend
        self.__balance_scores = np.zeros((len(backends),), dtype=np.float32)
        self.__balance_ratios = np.ones((len(backends),), dtype=np.float32)

        self.__backend_impls = []
        for i, backend in enumerate(backends):
            if backend["type"] == "onnxruntime":
                impl = OnnxruntimeBackend()
                impl.setup(model_path, **backend["backend_cfg"])

                ratio = backend.get("balance_ratio", 1.0)
                self.__balance_ratios[i] = ratio
            else:
                # TODO: 实现其它的 backend
                raise NotImplementedError
            
            self.__backend_impls.append(impl)

    def __del__(self):
        logger.info(f"del {self}")
        if hasattr(self, '__backend_impls'):
            for impl in self.__backend_impls:
                impl.teardown()

    def __repr__(self):
        info = f"AModel: {self.__name}, mid:{self.__mid:08x}, path:{self.__model_path}, with {len(self.__backend_impls)} backend impls\n"
        for i, impl in enumerate(self.__backend_impls):
            info += f"#{i}: {impl}, balance {self.__balance_scores[i]}, {self.__balance_ratios[i]}\n"
        return info

    def mid(self) -> int:
        return self.__mid
    
    def name(self) -> str:
        return self.__name
    
    def model_path(self) -> str:
        return self.__model_path
    
    def _preprocess(self, task:ATask):
        pass

    def _infer(self, task:ATask):
        pass

    def _postprocess(self, task:ATask):
        pass

    def __get_impl_id(self):
        '''
           根据 balance_score + balance_ratios 平均分配策略
           返回 balance_score 中最小的 backend_impl
        '''
        idx = np.argmin(self.__balance_scores)
        logger.debug("AModel: {}, idx:{}, balance:{}".format(self.__name, idx, self.__balance_scores))
        self.__balance_scores[idx] += self.__balance_ratios[idx]
        return idx

    def __call__(self, *args):
        idx = self.__get_impl_id()
        return self.__backend_impls[idx].infer(*args)