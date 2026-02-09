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

logger = logging.getLogger("asr_runner")

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
        self.debug_path = ""
        self.debug = kwargs.get("debug", False)
        if self.debug:
            import sys
            if sys.platform == "win32":
                self.debug_path = "p:/tmp/debug"
            else:
                self.debug_path = "/media/pub/tmp/debug"

        # logger.info(f"AModel: init: name:{name}, mid:{mid}, model_path:{model_path}, kwargs:{kwargs}")

        backends = kwargs.get("backend", [
            {
                "type": "onnxruntime", 
                "backend_cfg": {
                    "force_cpu": True,
                    "balance_ratio": 1.0,
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
            if not "type" in backend:
                raise ValueError("backend type is required")

            if not "model_path" in backend:
                backend["model_path"] = model_path  ## 默认填充

            if backend["type"] == "onnxruntime":
                if not "backend_cfg" in backend:
                    backend["backend_cfg"] = {
                        "force_cpu": True,
                        "balance_ratio": 1.0,
                        "intra_op_num_threads": 1,
                        "inter_op_num_threads": 1,
                        "providers": ["CPUExecutionProvider"],
                        "providers_cfg": [
                            {}
                        ]
                    }

                impl = OnnxruntimeBackend()
                impl.setup(backend["model_path"], **backend["backend_cfg"])
                ratio = float(backend["backend_cfg"].get("balance_ratio", 1.0))
                self.__balance_ratios[i] = ratio

            else:
                # TODO: 实现其它的 backend
                raise NotImplementedError
            
            self.__backend_impls.append(impl)

    def __del__(self):
        if hasattr(self, '__backend_impls'):
            for impl in self.__backend_impls:
                impl.teardown()
            self.__backend_impls.clear()

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
    
    def get_input_count(self) -> int:
        return self.__backend_impls[0].get_input_num()
    
    def get_input_shape(self, i:int) -> tuple[int, ...]:
        return self.__backend_impls[0].get_input_shape(i)
    
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
        if self.debug:
            logger.debug("AModel: {}, idx:{}, balance:{}".format(self.__name, idx, self.__balance_scores))
        self.__balance_scores[idx] += self.__balance_ratios[idx]
        return idx

    def __call__(self, *args):
        idx = self.__get_impl_id()
        return self.__backend_impls[idx].infer(*args)

    def softmax(self, x):
        """稳定的softmax实现"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _hlp_batch_infer(self, B:int, inp:np.ndarray, default_out=np.empty((0,), np.float32)) -> np.ndarray:
        ## B 必须 > 0 且为 2 的整数幂，根据 args 数据分解为 B, B/2, B/4, ... 1 推理
        ## 然后合并
        ## FIXME: 目前仅仅支持单输入，且输入为 np.ndarray，形状为 (B, xx, xx, ... )
        ## default_out: 当没有输入时，返回该值
        assert B > 0 and (B & (B-1)) == 0
        assert self.get_input_count() == 1

        def next_batch(inp0: np.ndarray, batch_size: int):
            assert batch_size > 0 and (batch_size & (batch_size - 1)) == 0  # 确保是2的幂
            assert inp0.ndim >= 2
            S = 0
            while len(inp0[S:]) >= 1:
                if len(inp0[S:]) >= batch_size:
                    yield inp0[S:S+batch_size]
                    S += batch_size
                else:
                    if batch_size == 1:
                        yield inp0[S:S+1]
                        break
                    else:
                        batch_size //= 2  # 减半batch_size继续尝试

        ret = []
        for batch in next_batch(inp, B):
            out = self((batch,))[0]
            ret.append(out)

        if not ret:
            return default_out
        return np.vstack(ret)
