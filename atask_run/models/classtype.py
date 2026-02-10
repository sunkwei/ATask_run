'''
  课堂类型判断模型

  input:
        f32 (1,3,360,640)

    output:
        f32 (1,2)   ( 小组课概率， 传统课堂概率)
'''

from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2

logger = logging.getLogger("classtype")

class Model_classtype(AModel):
    def _preprocess(self, task: ATask):
        ''' bgr -> rgb -> float -116.28, *0.0175 -> transpose -> (1,3,360,640)
        '''
        def do_pre(bgr: np.ndarray) -> np.ndarray:
            rgb = bgr[..., ::-1].astype(np.float32)
            rgb = rgb - np.array([-116.28, -116.28, -116.28])
            rgb = rgb * 0.0175
            return rgb.transpose((2, 0, 1))[None]

        if isinstance(task.inpdata, np.ndarray):
            task.data["classtype_inp"] = do_pre(task.inpdata)
        elif isinstance(task.inpdata, (tuple, list)):
            task.data["classtype_inp"] = np.vstack([
                do_pre(bgr)
                for bgr in task.inpdata
            ])
        else:
            raise TypeError("task.inpdata must be a single image (np.ndarray) or multiple images (tuple | list).")

    def _infer(self, task: ATask):
        task.data["classtype_infer"] = self._hlp_batch_infer(
            1,
            task.data["classtype_inp"],
            default_out=np.zeros((1, 2), dtype=np.float32)
        )

    def _postprocess(self, task: ATask):
        task.data["classtype_result"] = [
            v for v in task.data["classtype_infer"]
        ]
