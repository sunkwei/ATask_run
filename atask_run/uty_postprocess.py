'''
uty_postprocess 的 Docstring

'''
import numpy as np
from typing import List
import logging

logger = logging.getLogger("uty_post")

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """标准 NMS，返回保留索引"""
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

        # sigmoid 激活
        # cls_scores = 1 / (1 + np.exp(-cls_logits))

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

