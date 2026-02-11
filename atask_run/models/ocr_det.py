# TODO: ocr 检测
## 封装百度 ppocr 的 ocr 检测模块

from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2
from .ppocr_postprocess.db_postprocess import DBPostProcess

logger = logging.getLogger("ocr_det")

class Model_ocr_det(AModel):
    def _preprocess(self, task: ATask):
        '''
        ppOCRv5 的 text detect 预处理：
            bgr -> rgb -> float32
            img /= 255.0
            img - mean (0.485, 0.456, 0.406)
            img / std (0.229, 0.224, 0.225)
            transpose (hwc -> chw)
            img = np.expand_dims(img, axis=0)
        '''
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)

        def prepare_img_align(image0: np.ndarray, align: int = 32) -> Tuple[np.ndarray, float]:
            """
            将输入图像按比例缩放到 `align` 的整数倍，保持宽高比，并返回缩放后的图像及缩放比例。
            
            Args:
                image0: 输入图像 (H, W, C) 或 (H, W)，格式为 np.uint8 或 np.float32。
                align:  目标尺寸对齐的倍数（默认32）。
            
            Returns:
                Tuple[np.ndarray, float]: (缩放后的图像, 缩放比例)
            """
            h, w = image0.shape[:2]
            
            # 计算缩放比例（长边不超过 align 的倍数，同时保持宽高比）
            target_max_size = max(h, w)
            ratio = align * (target_max_size // align + (1 if target_max_size % align != 0 else 0)) / target_max_size
            new_h, new_w = int(h * ratio), int(w * ratio)
            
            # 调整尺寸到 align 的倍数（向上取整）
            new_h = (new_h + align - 1) // align * align
            new_w = (new_w + align - 1) // align * align
            
            # 实际缩放比例（可能因向上取整略有变化）
            actual_ratio_h = new_h / h
            actual_ratio_w = new_w / w
            actual_ratio = (actual_ratio_h + actual_ratio_w) / 2  # 取平均值
            
            # 使用 INTER_AREA 缩小时抗锯齿
            resized_img = cv2.resize(image0, (new_w, new_h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)    
            return resized_img, actual_ratio  # 返回缩放后的图像和还原比例（ratio）
        
        def do_pre(bgr:np.ndarray) -> Tuple[np.ndarray, float]:
            bgr, ratio = prepare_img_align(bgr)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = img.transpose((2, 0, 1))
            return img[None, ...], ratio       # (1, 3, h, w)

        inps = []  ## [((1,3,h,w), ratio), ... ]
        if isinstance(task.inpdata, np.ndarray):
            inps.append(do_pre(task.inpdata))
        else:
            for bgr in task.inpdata:
                inps.append(do_pre(bgr))
        task.data["ocr_det_inps"] = inps    ## 注意：这里输入是 list
    
    def _infer(self, task: ATask):
        '''
        ppOCRv5 的 text detect 推理：
        '''
        out = [ self((x[0],))[0] for x in task.data["ocr_det_inps"] ]
        task.data["ocr_det_infer"] = out
    
    def _postprocess(self, task: ATask):
        if not hasattr(self, "db_post_impl"):
            self.db_post_impl = DBPostProcess(
                thresh=task.userdata.get("ocr_det_thresh", 0.3),
                box_thresh=task.userdata.get("ocr_det_box_thresh", 0.7),
                max_candidates=task.userdata.get("ocr_det_max_candidates", 1000),
                unclip_ratio=task.userdata.get("ocr_det_unclip_ratio", 2.0),
                use_dilation=task.userdata.get("ocr_det_use_dilation", False),
                score_mode=task.userdata.get("ocr_det_score_mode", "fast"),
                box_type=task.userdata.get("ocr_det_box_type", "quad"),
            )
        shape_list = [
            (x[0].shape[-2], x[0].shape[-1], -1, -1)  ## (src_h, src_w, ratio_h, ratio_w)
            for x in task.data["ocr_det_inps"]
        ]
        results = []
        for b,pred in enumerate(task.data["ocr_det_infer"]):
            pred_dict = {"maps": pred}
            boxes = self.db_post_impl(pred_dict, shape_list)[0]["points"]
            ratio = task.data["ocr_det_inps"][b][1]
            boxes = boxes / ratio       ## 还原到原图坐标
            results.append(boxes.astype(np.int32))
        task.data["ocr_det_result"] = results
