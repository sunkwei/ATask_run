''' 人脸特征向量， facerec.onnx

    输入：经过校准的人脸图像， [ (n, 3, 112, 112), ... ]
    输出：人脸特征向量,       [ (n, 512), ... ]

模型输入：
        f32: (B, 3, 112, 112)
    输出:
        f32: (B, 512)

'''

from ..atask import ATask
from ..amodel import AModel
import numpy as np
import logging
import cv2
from typing import List
import os.path as osp

logger = logging.getLogger("facerec")

_std_face = np.array([
    (38.2946, 51.5387),
    (73.5317, 51.5387),
    (56.0253, 71.7366),
    (41.5493, 92.2041),
    (70.7299, 92.2041),
], np.float32)

def _crop_face_image(img0, faces, landmarks, face_score, debug_path:str="", debug:bool=False) -> List[np.ndarray]:
    ''' 对每个人脸，使用 _std_face 校准，然后将人脸图像调整为 (112, 112) 
        BGR24 -> RGB24 -> float -127.5 /127.5 -> CHW -> BCHW
    '''
    if debug:
        fname = osp.join(debug_path, "arun_facerec.jpg")
        cv2.imwrite(fname, img0)
        logger.warning(f"debug: save {fname}")

    inps = [ np.empty((0, 3, 112, 112), np.float32) ]
    for i in range(len(faces)):
        M, _ = cv2.estimateAffine2D(
            landmarks[i].reshape((5,2)),
            _std_face,
        )
        if M is None:
            x1,y1,x2,y2 = faces[i, :4].astype(np.int32)
            img = cv2.resize(img0[y1:y2, x1:x2, :], (112, 112))
        else:
            img = cv2.warpAffine(img0, M, (112, 112))

        if debug:
            fname = osp.join(debug_path, f"arun_facerec_{i}.jpg")
            logger.warning(f"debug: save {fname}")
            cv2.imwrite(fname, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = img.astype(np.float32).transpose(2, 0, 1)
        inp = (inp - 127.5) / 127.5
        inps.append(inp[None, ...])     # (1, 3, 112, 112)

    return inps

class Model_facerec(AModel):
    def _preprocess(self, task: ATask):
        batch_size = len(task.data["facedet_result_face"])

        assert "facedet_result_face" in task.data
        
        task.data["facerec_inp"] = []
        faces = task.data["facedet_result_face"]
        landmarks = task.data["facedet_result_landmark"]
        face_scores = task.data.get("face_score_result", [ [] for i in range(batch_size)])

        for b in range(batch_size):
            if isinstance(task.inpdata, np.ndarray):
                img = task.inpdata
            else:
                img = task.inpdata[b]
            task.data["facerec_inp"].append(
                np.vstack(_crop_face_image(img, faces[b], landmarks[b], face_scores[b], self.debug_path, self.debug))
            )

        if task.data["facerec_inp"]:
            task.data["facerec_inp"] = np.vstack(task.data["facerec_inp"])
        else:
            task.data["facerec_inp"] = np.zeros((0, 3, 112, 112), np.float32)
        
    def _infer(self, task: ATask):
        ## 将 facerec_infer 根据 batch 返回 [ (n, 512), ... ] 的格式
        task.data["facerec_infer"] = self._hlp_batch_infer(
            32, 
            task.data["facerec_inp"], 
            default_out=np.empty((0, 512), np.float32),
        )
