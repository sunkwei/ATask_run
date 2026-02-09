'''
uty_postprocess 的 Docstring

'''
import numpy as np
from typing import List, Tuple
import logging
import cv2

logger = logging.getLogger("uty_post")

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """标准 NMS，返回保留索引
        boxes: (N, 4)  [x,y,w,h]
        scores: (N, )
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        # IoU 计算
        xx1 = np.maximum(boxes[i, 0] - boxes[i, 2] / 2, boxes[idxs[1:], 0] - boxes[idxs[1:], 2] / 2)
        yy1 = np.maximum(boxes[i, 1] - boxes[i, 3] / 2, boxes[idxs[1:], 1] - boxes[idxs[1:], 3] / 2)
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2] / 2, boxes[idxs[1:], 0] + boxes[idxs[1:], 2] / 2)
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3] / 2, boxes[idxs[1:], 1] + boxes[idxs[1:], 3] / 2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = boxes[i, 2] * boxes[i, 3]
        area_rest = boxes[idxs[1:], 2] * boxes[idxs[1:], 3]
        iou = inter / (area_i + area_rest - inter + 1e-6)

        idxs = idxs[1:][iou <= iou_thresh]
    return keep

def yolo_act_post(infer_out: np.ndarray, conf_thresh=0.4, iou_thresh=0.45, cross_class_nms=False) -> List[np.ndarray]:
    """
    YOLO 后处理
    infer_out: (batch_size, 20, 17010)
    返回: list，每个元素为 (m, 6)，对应 (cx, cy, w, h, score, cid)
    """
    batch_size = infer_out.shape[0]
    results = []

    for b in range(batch_size):
        out = infer_out[b]  # (20, 17010)
        boxes = out[:4].T       # (17010, 4) -> (cx, cy, w, h)
        cls_scores = out[4:]    # (16, 17010)

        # 取最大类别
        cid = np.argmax(cls_scores, axis=0)
        score = cls_scores[cid, np.arange(cls_scores.shape[1])]

        # 阈值过滤
        mask = score >= conf_thresh
        boxes = boxes[mask]
        score = score[mask]
        cid = cid[mask]

        if boxes.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        if cross_class_nms:
            keep = nms(boxes, score, iou_thresh)
        else:
            keep = []
            for c in np.unique(cid):
                idxs = np.where(cid == c)[0]
                keep_c = nms(boxes[idxs], score[idxs], iou_thresh)
                keep.extend(idxs[keep_c])

        boxes = boxes[keep]
        score = score[keep]
        cid = cid[keep]

        det = np.concatenate([boxes, score[:, None], cid[:, None]], axis=1)
        results.append(det.astype(np.float32))

    return results

def yolo_facedet_post(infer_out: np.ndarray, conf_thresh=0.4, iou_thresh=0.45) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''
    yolov5 face 人脸检测后处理，输入 infer_out (np.ndarray) (B, 32130, 16)
    输出 
        list [ (n, 5), ... ]，人脸框，每个 (x1,y1,x2,y2,score)
        list [ (n, 10), ... ], 人脸特征点
    '''
    batch_size = infer_out.shape[0]
    result_face = []; result_landmark = []

    for b in range(batch_size):
        out = infer_out[b]  # (32130, 16)
        # 16: (x,y,w,h,score,landmark(10))

        ## 使用 score 过滤
        mask = out[:, 4] > conf_thresh
        out = out[mask]

        if out.shape[0] == 0:
            result_face.append(np.zeros((0, 5), dtype=np.float32))
            result_landmark.append(np.zeros((0, 10), dtype=np.float32))
            continue

        ## 使用 NMS 过滤
        keep = nms(out[:, :4], out[:, 4], iou_thresh)
        out = out[keep]

        ## xywh -> xyxy 格式
        w = out[:, 2]; h = out[:, 3]
        out[:, 0] = out[:, 0] - w/2; out[:, 1] = out[:, 1] - h/2
        out[:, 2] = out[:, 0] + w;   out[:, 3] = out[:, 1] + h

        result_face.append(out[:, :5])
        result_landmark.append(out[:, 5:15])

    return result_face, result_landmark

def clip_and_filter(boxes:np.ndarray, width:int, height:int) -> Tuple[np.ndarray, np.ndarray]:
    # clip 坐标
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)

    # 过滤掉 x1 >= x2 或 y1 >= y2 的行
    valid_mask = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
    boxes = boxes[valid_mask]
    return boxes, valid_mask

def debug_draw_faceori_box(image, data):
    # 要绘制朝向矩形，需要首先将 faceori_mask 对应的人脸移动到图像中心，才能使用 t_vec!!!
    cm = data['faceori_camera_matrix']
    coeefs = data['faceori_dist_coeefs']
    rear = np.array([(-50.0, -50.0, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    front = np.array([(-80.0, -80.0, 100), (80, -80, 100), (80, 80, 100), (-80, 80, 100)])
    pt_3d = np.vstack((rear, front)).reshape((-1, 3))

    image_center = np.array((image.shape[1] // 2, image.shape[0] // 2))

    for i in range(len(data['facedet_result_face'])):
        pt_2d, _ = cv2.projectPoints(pt_3d, data['faceori_rvec'][i], data['faceori_tvec'][i], cm, coeefs)
        pt_2d = pt_2d.reshape((-1, 2))

        ## 坐标需要减去图像中心，再移动到人脸中心，才能正确绘制
        ## XXX: 其实应该移动到鼻子
        x1,y1,x2,y2 = data["facedet_result_face"][i, :4].astype(int)
        face_center = np.array([(x1+x2)/2, (y1+y2)/2])
        pt_2d = pt_2d - image_center + face_center

        fp1,fp2,fp3,fp4 = pt_2d[:4, :].astype(int)
        rp1,rp2,rp3,rp4 = pt_2d[4:, :].astype(int)
        cv2.line(image, tuple(rp1), tuple(rp2), (0, 244, 244))
        cv2.line(image, tuple(rp2), tuple(rp3), (0, 244, 244))
        cv2.line(image, tuple(rp3), tuple(rp4), (0, 244, 244))
        cv2.line(image, tuple(rp4), tuple(rp1), (0, 244, 244))

        cv2.line(image, tuple(fp1), tuple(fp2), (255, 244, 0))
        cv2.line(image, tuple(fp2), tuple(fp3), (255, 244, 0))
        cv2.line(image, tuple(fp3), tuple(fp4), (255, 244, 0))
        cv2.line(image, tuple(fp4), tuple(fp1), (255, 244, 0))

        cv2.line(image, tuple(rp1), tuple(fp1), (0, 244, 0))
        cv2.line(image, tuple(rp2), tuple(fp2), (0, 244, 0))
        cv2.line(image, tuple(rp3), tuple(fp3), (0, 244, 0))
        cv2.line(image, tuple(rp4), tuple(fp4), (0, 244, 0))
