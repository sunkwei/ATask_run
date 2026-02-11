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
        TODO: 需要支持较固定几个大小的输入，如 960x544, 1280x720, 640x360 等


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
        
        def prepare_image(
            image: np.ndarray, 
            sizes=[(640,384), (960,544), (1280,736)]
        ) -> Tuple[np.ndarray, float]:
            
            ## FIXME: sizes 总是从小到大
            ## 模型输入需要 32 整除
            assert len(sizes) > 0
            h, w = image.shape[:2]

            ## 反序找到比 (w, h) 小的目标尺寸
            target_w, target_h = sizes[0]

            if w < sizes[0][0] and h <= sizes[0][1]:
                pass
            else:
                for tw, th in reversed(sizes):
                    if w < tw and h < th:
                        continue

                    if w <= tw or h <= th:
                        target_w = tw; target_h = th
                        break

            # 计算缩放比例（保持比例，且不放大）
            scale = min(target_w / w, target_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)

            # 缩放图像
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 左上角填充
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas[0:new_h, 0:new_w] = resized

            return canvas, scale

        def do_pre(bgr:np.ndarray) -> Tuple[np.ndarray, float]:
            # bgr, ratio = prepare_img_align(bgr)
            bgr, ratio = prepare_image(bgr)
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
