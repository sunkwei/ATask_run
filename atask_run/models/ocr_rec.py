# TODO: OCR rec 模型
## 封装百度 ppocr 的 ocr 识别模块

from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2
from .ppocr_postprocess.rec_postprocess import CTCLabelDecode

logger = logging.getLogger("ocr_rec")

class Model_ocr_rec(AModel):
    width_segs = [64, 128, 192, 256, 384, 512, 768, 1024, 1280, 1600]

    def _preprocess(self, task: ATask):
        ''' 
        ppOCRv5 的 text rec 预处理：
            与 text detect 相同，直接将检测框内的像素提取出来
            输入尺寸为 (B, 3, 48, w)

            1. 根据 task.data["ocr_det_result"] 抠图，需要处理旋转情况
            2. 每个图，保持比例，高缩放到 48 像素
            3. 将图像根据最长，做padding, 构造批次？如果长短差别太大，是不是应该使用“桶”？

            为了支持 tensorRT，长度最好分几个段
        '''
        assert "ocr_det_result" in task.data
        task.data["ocr_rec_inps"] = []
        for b, boxes in enumerate(task.data["ocr_det_result"]):
            inps = []
            img0 = task.inpdata if isinstance(task.inpdata, np.ndarray) else task.inpdata[b]
            max_fw = 0
            for points in boxes:
                fixed_img, rot90 = self.__get_rotate_crop_image(img0, points)
                h, w = fixed_img.shape[:2]
                fw = int(w / (h / 48))
                max_fw = max(max_fw, fw)
                fixed_img = cv2.resize(fixed_img, (fw, 48))

                inp = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                inp -= 0.5
                inp /= 0.5
                inp = np.transpose(inp, (2, 0, 1))[None, ...]
                inps.append(inp)

            ## 将 inps 中所有 inp 扩展到 max_fw 宽度，不足部分填充 
            ## 从 self.width_segs 中选择一个合适的宽度
            target_width = self.width_segs[-1]
            for w in self.width_segs[:-1]:
                if w >= max_fw:
                    target_width = w
                    break

            for i in range(len(inps)):
                ## inp: (1, 3, H, W), 扩展 W 到 fw
                if inps[i].shape[-1] < target_width:
                    pad_w = target_width - inps[i].shape[-1]
                    inps[i] = np.pad(inps[i], ((0, 0), (0, 0), (0, 0), (0, pad_w)), mode="constant", constant_values=0)

            task.data["ocr_rec_inps"].append(np.vstack(inps))


    def _infer(self, task: ATask):
        task.data["ocr_rec_infer"] = []
        for b, inps in enumerate(task.data["ocr_rec_inps"]):
            out = self._hlp_batch_infer(
                16,
                inps,
                default_out=np.empty((0, 1, 18385), np.float32)
            )
            task.data["ocr_rec_infer"].append(out)
    
    def _postprocess(self, task: ATask):
        if not hasattr(self, "decoder"):
            import os.path as osp
            model_path = osp.dirname(self.model_path())
            dict_path = osp.join(model_path, "ppocrv5")
            self.decoder = CTCLabelDecode(
                character_dict_path=osp.join(dict_path, "ppocrv5_dict.txt")
            )
        
        task.data["ocr_rec_result"] = []
        for b, logit in enumerate(task.data["ocr_rec_infer"]):
            texts = self.decoder(logit)
            task.data["ocr_rec_result"].append(texts)

    def __get_rotate_crop_image(self, img, points) -> Tuple[np.ndarray, bool]:
        roat_90 = False
        img_crop_width  = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

        pts_std = np.array([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)

        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )

        dst_img_height, dst_img_width = dst_img.shape[:2]
        if (dst_img_height * 1.0 / dst_img_width) >= 1.5:
            dst_img = np.rot90(dst_img)
            roat_90 = True
        return dst_img, roat_90    