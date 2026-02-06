'''
人脸质量模型：face_score.onnx
'''

from ..atask import ATask
from ..amodel import AModel
import numpy as np
import logging
import cv2

logger = logging.getLogger("face_score")

''' 人脸质量打分
    输入 
        task.inpdata 可能是单张图像，或多张图象, 后者使用元组存储
        task.data["facedet_result_face"]: [(n, 5), ...] 人脸检测结果

    输出：
        task.data["face_score_result"]: [ (n, 10), ... ] n=人脸数
'''
class Model_face_score(AModel):
    def _preprocess(self, task: ATask):
        '''
            task.inpdata 可能是单张图像，或多张图象, 后者使用元组存储
            task.data["facedet_result_face"]: [(n, 5), ...]
            
            预处理：
                根据 facedet_result_face 的人脸检测结果，从 task.inpdata 中抠出人脸图像
                执行 BGR24 -> RGB24 -> resize 112 x 112 -> CHW -> float32 -127.5, /127.5 -> (B, C, H, W)
        '''
        assert "facedet_result_face" in task.data
        total_face_num = sum([len(r) for r in task.data["facedet_result_face"]])
        face_score_inp = np.empty((total_face_num, 3, 112, 112), np.float32)    ## 预分配

        face_no = 0
        for b in range(len(task.data["facedet_result_face"])):
            faces = task.data["facedet_result_face"][b]
            if isinstance(task.inpdata, np.ndarray):
                img = task.inpdata
            else:
                img = task.inpdata[b]

            for fid, face in enumerate(faces):
                x1, y1, x2, y2 = face[:4].astype(np.int32)
                face_img = img[y1:y2, x1:x2]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (112, 112))
                face_score_inp[face_no, ...] = np.transpose(face_img, (2, 0, 1)).astype(np.float32).reshape(3, 112, 112)
                face_no += 1

        face_score_inp = (face_score_inp - 127.5) / 127.5
        task.data["face_score_inp"] = face_score_inp

    def _infer(self, task: ATask):
        task.data["face_score_infer"] = self._hlp_batch_infer(
            32, 
            task.data["face_score_inp"],
            default_out=np.empty((0, 10), np.float32),
        )
    
    def _postprocess(self, task: ATask):
        assert "face_score_infer" in task.data
        assert "facedet_result_face" in task.data

        ## 根据每张图像中的人脸数，分割 face_score_infer 到 list
        face_score_list = []
        start = 0
        for b in range(len(task.data["facedet_result_face"])):
            faces = task.data["facedet_result_face"][b]
            landmarks = task.data["facedet_result_landmark"][b]

            end = start + len(faces)
            face_core = task.data["face_score_infer"][start:end]
            start = end
            
            ## 根据 face_score 前三维的和，对 facedet_result_face|landmark 进行排序
            criteria = face_core[:, :3]
            order = np.lexsort((-criteria[:,0], -criteria[:,1], -criteria[:,2]))
            task.data["facedet_result_face"][b] = faces[order]
            task.data["facedet_result_landmark"][b] = landmarks[order]
            face_score_list.append(face_core[order])

        task.data["face_score_result"] = face_score_list
