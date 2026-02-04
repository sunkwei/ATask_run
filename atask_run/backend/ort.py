''' 基于 onnxruntime 的后端

'''

import onnxruntime as rt
from .amodel_impl import AModelBackend
from typing import List, Tuple, Any
import logging

logger = logging.getLogger("backend_onnxruntime")

class OnnxruntimeBackend(AModelBackend):
    def setup(self, model_path: str, **kwargs):
        self.model_path = model_path
        sess_opt = rt.SessionOptions()
        sess_opt.intra_op_num_threads = kwargs.get("intra_op_num_threads", 1)
        sess_opt.inter_op_num_threads = kwargs.get("inter_op_num_threads", 1)

        if kwargs.get("force_cpu", False):
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        else:
            providers = kwargs.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
            provider_options = kwargs.get("providers_cfg", [{"device_id": 0}, {}])

            if len(providers) != len(provider_options):
                raise ValueError(
                    "The length of providers must be equal to the length of provider_options"
                )

        self.sess = rt.InferenceSession(
            model_path, 
            sess_opt, 
            providers=[ (p, po) for p,po in zip(providers, provider_options) ],
        )

        ## 实际使用的 privoider
        self.__curr_p = self.sess.get_providers()[0]

        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def __repr__(self) -> str:
        info = f"OrtBackend: model:{self.model_path} P:{self.__curr_p}, with inputs({len(self.input_names)}) outputs({len(self.output_names)})"
        for i, name in enumerate(self.input_names):
            info += f"\n\tinput[{i}] {name}: shape={self.get_input_shape(i)}"
        for i, name in enumerate(self.output_names):
            info += f"\n\toutput[{i}] {name}: shape={self.get_output_shape(i)}"
        return info

    def teardown(self):
        del self.sess

    def get_input_num(self) -> int:
        return len(self.input_names)

    def get_input_shape(self, idx: int) -> tuple[int, ...]:
        return self.sess.get_inputs()[idx].shape

    def get_input_dtype(self, idx: int) -> str:
        return self.sess.get_inputs()[idx].type

    def get_output_num(self) -> int:
        return len(self.output_names)

    def get_output_shape(self, idx: int) -> tuple[int, ...]:
        return self.sess.get_outputs()[idx].shape

    def get_output_dtype(self, idx: int) -> str:
        return self.sess.get_outputs()[idx].type

    def infer(self, inputs: Tuple[Any]) -> List[Any]:
        out = self.sess.run(
            None,
            input_feed=dict(zip(self.input_names, inputs)),
        )
        return list(out)