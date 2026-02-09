''' 人脸 106 特征点模型， faceori.onnx


    输入：
        经过校准的人脸图像， [ (n, 3, 192, 192), ... ]
    输出：
        68特征点，旋转向量，平移向量，角度等 ...
'''

from ..atask import ATask
from ..amodel import AModel
import numpy as np
import logging
import cv2
from typing import List, Tuple
import os.path as osp
import math

logger = logging.getLogger("faceori")

_idx_106_5 = np.array([
    38, 88, 86, 52, 61,  # 左眼，右眼，鼻子，左嘴角，右嘴角
])

_idx_106_68 = np.array([
    1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,   # 面部轮廓
    43,48,49,51,50,         # 左眉 5
    102,103,104,105,101,    # 右眉 5
    72,73,74,86,77,79,80,83,84, # 鼻子 9
    35,41,42,39,37,36,  # 左眼  6
    89,95,96,93,91,90,  # 右眼  6
    52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54, # 嘴巴 20
])

_3D_face_xyz = [
    # X
        -73.393523, -72.775014, -70.533638,
        -66.850058, -59.790187, -48.368973, 
        -34.121101, -17.875411, 0.098749,
        17.477031, 32.648966, 46.372358,
        57.343480, 64.388482, 68.212038,
        70.486405, 71.375822, -61.119406,
        -51.287588, -37.804800, -24.022754,
        -11.635713, 12.056636, 25.106256,
        38.338588, 51.191007, 60.053851 ,
        0.653940, 0.804809, 0.992204 ,
        1.226783, -14.772472, -7.180239, 
        0.555920, 8.272499, 15.214351 ,
        -46.047290, -37.674688, -27.883856, 
        -19.648268, -28.272965, -38.082418 ,
        19.265868, 27.894191, 37.437529 ,
        45.170805, 38.196454, 28.764989 ,
        -28.916267, -17.533194, -6.684590, 
        0.381001, 8.375443, 18.876618 ,
        28.794412, 19.057574, 8.956375 ,
        0.381549, -7.428895, -18.160634 ,
        -24.377490, -6.897633, 0.340663 ,
        8.444722, 24.474473, 8.449166 ,
        0.205322, -7.198266, 
    # Y  
        -29.801432, -10.949766, 7.929818, 
        26.074280, 42.564390, 56.481080,
        67.246992, 75.056892, 77.061286,
        74.758448, 66.929021, 56.311389, 
        42.419126, 25.455880, 6.990805,
        -11.666193, -30.365191, -49.361602, 
        -58.769795, -61.996155, -61.033399, 
        -56.686759 , -57.391033, -61.902186, 
        -62.777713 , -59.302347, -50.190255, 
        -42.193790 , -30.993721, -19.944596, 
        -8.414541 , 2.598255, 4.751589, 
        6.562900 , 4.661005, 2.643046, 
        -37.471411, -42.730510, -42.711517, 
        -36.754742, -35.134493, -34.919043, 
        -37.032306 ,-43.342445, -43.110822, 
        -38.086515 ,-35.532024, -35.484289, 
        28.612716 , 22.172187, 19.029051, 
        20.721118 , 19.035460, 22.394109, 
        28.079924 , 36.298248, 39.634575, 
        40.395647 , 39.836405, 36.677899, 
        28.677771 , 25.475976, 26.014269, 
        25.326198 , 28.323008, 30.596216, 
        31.408738 , 30.844876, 
    # Z    
        47.667532, 45.909403 , 44.842580, 
        43.141114, 38.635298 , 30.750622, 
        18.456453, 3.609035 , -0.881698, 
        5.181201, 19.176563 , 30.770570, 
        37.628629, 40.886309 , 42.281449, 
        44.142567, 47.140426 ,14.254422, 
        7.268147, 0.442051 , -6.606501, 
        -11.967398, -12.051204, -7.315098, 
        -1.022953, 5.349435 ,11.615746, 
        -13.380835, -21.150853, -29.284036,
        -36.948060, -20.132003, -23.536684,
        -25.944448, -23.695741 , -20.858157,
        7.037989, 3.021217 ,1.353629, 
        -0.111088, -0.147273 , 1.476612, 
        -0.665746, 0.247660 , 1.696435, 
        4.894163, 0.282961 , -1.172675,
        -2.240310, -15.934335, -22.611355, 
        -23.748437, -22.721995, -15.610679, 
        -3.217393, -14.987997 ,-22.554245, 
        -23.591626, -22.406106 ,-15.121907, 
        -4.785684, -20.893742 ,-22.220479, 
        -21.025520, -5.712776 , -20.671489,
        -21.903670, -20.328022 ,
]

_3D_face_pts = np.array(_3D_face_xyz).reshape((3, 68)).T
_3D_face_pts[:, 2] *= -1.0     ## Z 反转

class Model_faceori(AModel):
    def _preprocess(self, task: ATask):
        batch_size = len(task.data["facedet_result_face"])
        assert "facedet_result_face" in task.data

        inps = [ np.empty((0, 3, 192, 192), np.float32) ]
        Ms = [ [] for _ in range(batch_size) ]        
        for b in range(batch_size):
            if isinstance(task.inpdata, np.ndarray):
                img0 = task.inpdata
            else:
                img0 = task.inpdata[b]

            faces = task.data["facedet_result_face"][b]

            for x1,y1,x2,y2 in faces[:, :4].astype(np.float32):
                cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                scale = 192 * 2 / 3 / max(x2 - x1, y2 - y1)

                M = np.array([
                    [scale, 0, -cx * scale + 192 / 2],
                    [0, scale, -cy * scale + 192 / 2],
                ])
                Ms[b].append(M)

                img = cv2.warpAffine(img0, M, (192, 192))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                inp = np.transpose(img, (2, 0, 1)).astype(np.float32)
                inp -= 127.5
                inp /= 128.0
                inps.append(inp[None, ...])

        task.data["faceori_inp"] = np.vstack(inps)
        task.data["faceori_ms"] = Ms
    
    def _infer(self, task: ATask):
        task.data["faceori_infer"] = self._hlp_batch_infer(
            32,
            task.data["faceori_inp"],
            default_out=np.empty((0, 212), np.float32),
        )
               
    def _postprocess(self, task: ATask):
        B = len(task.data["facedet_result_face"])
        task.data["faceori_result_landmark"] = []
        task.data["faceori_camera_m"] = []
        task.data["faceori_camera_coeefs"] = []
        task.data["faceori_pts68"] = []     # [ (N, 68, 2), ... ] 人脸68点图像坐标
        task.data["faceori_angle"] = []         # [ (N, 3), ... ]
        task.data["faceori_rvec"] = []      # [ (N, 3), ... ]
        task.data["faceori_tvec"] = []      # [ (N, 3), ... ]

        ## 将 faceori_infer (mergedN, 212) 转换为每张图像的 (N, 212)
        infer_outs = []
        start = 0
        for b in range(B):
            num = len(task.data["facedet_result_face"][b])
            infer_outs.append(task.data["faceori_infer"][start:start+num])
            start += num

        ## 每张图像计算 68特征点，构造 faceori_result_landmark，....
        for b in range(B):
            ## XXX: 这些镜头参数，应该来自 userdata
            if isinstance(task.inpdata, np.ndarray):
                image0 = task.inpdata
            else:
                image0 = task.inpdata[b]

            focal_length = image0.shape[1] / 2.0    ## FIXME: 图像宽的一半
            camera_cx = image0.shape[1] / 2.0
            camera_cy = image0.shape[0] / 2.0
            coeefs = np.zeros((5, 1), np.float32)
            camera_m = np.array([
                (focal_length, 0, camera_cx),
                (0, focal_length, camera_cy),
                (0, 0, 1),
            ])
            task.data["faceori_camera_m"].append(camera_m)
            task.data["faceori_camera_coeefs"].append(coeefs)

            pts68s = [ np.empty((0, 68, 2), np.float32) ]
            tvecs = [ np.empty((0, 3), np.float32) ]
            rvecs = [ np.empty((0, 3), np.float32) ]
            angles = [ np.empty((0, 3), np.float32) ]
            landmarks = [ np.empty((0, 10), np.float32) ]

            Ms = task.data["faceori_ms"][b]
            for i, p212 in enumerate(infer_outs[b]):
                ## p212 为模型输出
                pts68 = p212.reshape((-1, 2))[_idx_106_68]
                pts68 += 1.0
                pts68 *= 96         ## 相对于 Ms 的坐标
                ## 逆变换为原图坐标
                ## 需要转换为齐次坐标
                pts68 = np.hstack((pts68, np.ones((pts68.shape[0], 1), dtype=pts68.dtype)))
                iM = cv2.invertAffineTransform(Ms[i])
                pts68 = np.dot(iM, pts68.T).T
                pts68 = pts68[:, :2]
                pts68s.append(pts68[None, ...])

                ## 生成 faceori_result_landmark，
                pts5 = p212.reshape((-1, 2))[_idx_106_5]
                pts5 += 1.0
                pts5 *= 96
                pts5 = np.hstack((pts5, np.ones((pts5.shape[0], 1), dtype=pts5.dtype)))
                pts5 = np.dot(iM, pts5.T).T
                pts5 = pts5[:, :2]
                landmarks.append(pts5.reshape((1, 10)))

                ## 使用鼻子为中心的坐标
                nose_dc = pts68[30] - (camera_cx, camera_cy) #鼻子相对中心的 x 偏移
                pts68_face_center = pts68 - nose_dc.reshape((-1, 2))  # 每张人脸到图像中心的偏移

                rc, r_vec, t_vec = cv2.solvePnP(_3D_face_pts, pts68_face_center, camera_m, coeefs)
                if rc:
                    rvecs.append(r_vec.astype(np.float32).reshape((1, 3)))
                    tvecs.append(t_vec.astype(np.float32).reshape((1, 3)))
                    angles.append(self.__rvec2euler(r_vec).reshape((1, 3)))

                else:
                    rvecs.append(np.zeros((1, 3,), np.float32))
                    tvecs.append(np.zeros((1, 3,), np.float32))
                    angles.append(np.array([[-90, -90, -90]], np.float32))
        
            task.data["faceori_pts68"].append(np.vstack(pts68s))
            task.data["faceori_rvec"].append(np.vstack(rvecs))
            task.data["faceori_tvec"].append(np.vstack(tvecs))
            task.data["faceori_angle"].append(np.vstack(angles))
            task.data["faceori_result_landmark"].append(np.vstack(landmarks))


    def __rvec2euler(self, rvec):
        ## 从旋转向量计算欧拉角
        r = cv2.Rodrigues(rvec)[0]     # (3,3)
        x = math.atan2(r[2,1], r[2,2])
        y = math.atan2(-r[2,0], math.sqrt(math.pow(r[2,1], 2) + math.pow(r[2,2], 2)))
        z = math.atan2(r[1,0], r[0,0])
        x = x * 180 / math.pi
        y = y * 180 / math.pi
        z = z * 180 / math.pi
        if (z > 90): z -= 180
        if (z < -90): z += 180
        return np.array([x, y, z], dtype=np.float32)