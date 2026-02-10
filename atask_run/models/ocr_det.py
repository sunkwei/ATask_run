# TODO: ocr 检测
## 封装百度 ppocr 的 ocr 检测模块

from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2

logger = logging.getLogger("ocr_det")

class Model_ocr_det(AModel):
    def _preprocess(self, task: ATask):
        pass
    
    def _postprocess(self, task: ATask):
        pass

    def _infer(self, task: ATask):
        pass