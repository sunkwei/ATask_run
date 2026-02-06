'''
    行为检测，yolov5m
'''
from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2
from ..uty_preprocess import prepare_image
from ..uty_postprocess import yolo_act_post

logger = logging.getLogger("act")


'''
    使用 yolo_act16.onnx 模型

        输入： (B, 3, 544, 960)
        输出： (B, m, 6) m=行为数, 6=(x1,y1,x2,y2,score,cid), 其中 x1,y1,x2,y2 对应输入图像的坐标
'''
class Model_act(AModel):
    def _preprocess(self, task: ATask):
        '''
            task.inpdata 可能是单张图像，或多张图象, 后者使用元组存储

            预处理：
                BGR24 -> RGB24
                resize 960 x 544
                HWC -> CHW
                cast uint8 -> float32 / 255.0
        '''
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
            task.data["act_inp"], xr, ry = once(task.inpdata)
            task.data["acr_rx_ry"] = np.array([(xr, ry)])
        elif isinstance(task.inpdata, (tuple, list)):
            imgs = []; rx_ry = []
            for bgr in task.inpdata:
                img, rx, ry = once(bgr)
                imgs.append(img)
                rx_ry.append((rx, ry))
            task.data["act_inp"] = np.vstack(imgs)
            task.data["acr_rx_ry"] = np.array(rx_ry)
        else:
            raise TypeError("task.inpdata must be a single image (np.ndarray) or multiple images (tuple | list).")

    def _infer(self, task: ATask):
        task.data["act_infer"] = self._hlp_batch_infer(16, task.data["act_inp"])
        

    def _postprocess(self, task: ATask):
        assert "act_infer" in task.data
        ## 从 (B, 4 + 16, m) 解析，去重，....返回 (B, n, 6)  , 其中 (x1,y1,x2,y2,score,cid)

        conf_thresh = task.userdata.get("act_conf_thresh", 0.2)
        iou_thresh = task.userdata.get("act_iou_thresh", 0.45)

        B = task.data["act_infer"].shape[0]
        task.data["act_result"] = yolo_act_post(task.data["act_infer"], conf_thresh, iou_thresh)

        for b in range(B):
            rx, ry = task.data["acr_rx_ry"][b]
            task.data["act_result"][b][:, (0, 2)] *= rx
            task.data["act_result"][b][:, (1, 3)] *= ry

            ## 将 cx, cy, w, h 转为 x1, y1, x2, y2
            W = task.data["act_result"][b][:, 2] - 0.0
            H = task.data["act_result"][b][:, 3] - 0.0
            task.data["act_result"][b][:, 0] -= W / 2
            task.data["act_result"][b][:, 1] -= H / 2
            task.data["act_result"][b][:, 2] = task.data["act_result"][b][:, 0] + W
            task.data["act_result"][b][:, 3] = task.data["act_result"][b][:, 1] + H
           