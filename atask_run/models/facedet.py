'''
    人脸检测，模型为 yolov5s-face.onnx
'''
from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2
from ..uty_preprocess import prepare_image
from ..uty_postprocess import yolo_facedet_post

logger = logging.getLogger("facedet")

''' 模型输入为 (B, 3, 544, 960)
    输出为 (B, 32130, 16)   其中 32130 的三个比例的锚框，16 为 (x,y,w,h,score, lm0, lm1, ... lm8, lm9, xx)
'''
class Model_facedet(AModel):
    def _preprocess(self, task: ATask):
        '''
            task.inpdata 可能是单张图像，或多张图象, 后者使用元组存储

            预处理：
                BGR24 -> RGB24
                resize 960 x 544
                HWC -> CHW
                cast uint8 -> float32 / 255.0

            如果存在 act_inp，则可以重用，
        '''

        if "act_inp" in task.data:
            task.data["facedet_inp"] = task.data["act_inp"]
        else:
            inp_shape = self.get_input_shape(0)         # (B, C, H, W)
            want_size = (inp_shape[3], inp_shape[2])    # 图像预处理大小
            if want_size[0] < 0 or want_size[1] < 0:
                want_size = (960, 544)

            def once(bgr24) -> Tuple[np.ndarray, float, float]:
                rgb = cv2.cvtColor(bgr24, cv2.COLOR_BGR2RGB)
                img, rx, ry = prepare_image(rgb, want_size, keep_aspect=True)
                img = img.astype(np.float32) / 255.0
                inp = np.transpose(img, (2, 0, 1))[None, ...]
                return inp, rx, ry  ## (1, C, H, W)
                
            if isinstance(task.inpdata, np.ndarray):
                task.data["facedet_inp"], xr, ry = once(task.inpdata)
                task.data["facedet_rx_ry"] = np.array([(xr, ry)])
            elif isinstance(task.inpdata, (tuple, list)):
                imgs = []; rx_ry = []
                for bgr in task.inpdata:
                    img, rx, ry = once(bgr)
                    imgs.append(img)
                    rx_ry.append((rx, ry))
                task.data["facedet_inp"] = np.vstack(imgs)
                task.data["facedet_rx_ry"] = np.array(rx_ry)
            else:
                raise TypeError("task.inpdata must be a single image (np.ndarray) or multiple images (tuple | list).")
            

    def _postprocess(self, task: ATask):
        '''
            后处理，返回 
                facedet_result_face: [ (n, 5), ... ]
                facedet_result_landmark: [ (n, 10), ... ]
        '''
        conf_thresh = task.userdata.get("facedet_conf_thresh", 0.2)
        iou_thresh = task.userdata.get("facedet_iou_thresh", 0.45)

        B = task.data["facedet_infer"].shape[0]
        task.data["facedet_result_face"], task.data["facedet_result_landmark"] = \
            yolo_facedet_post(task.data["facedet_infer"], conf_thresh, iou_thresh)

        for b in range(B):
            rx, ry = task.data["facedet_rx_ry"][b]
            task.data["facedet_result_face"][b][:, (0,2)] *= rx
            task.data["facedet_result_face"][b][:, (1,3)] *= ry
            task.data["facedet_result_landmark"][b][:, ::2] *= rx
            task.data["facedet_result_landmark"][b][:, 1::2] *= ry


    def _infer(self, task: ATask):
        '''
            推理，输入 facedet_inp: np.ndarray，输出 facedet_infer: np.ndarray
        '''
        task.data["facedet_infer"] = self._hlp_batch_infer(16, task.data["facedet_inp"])