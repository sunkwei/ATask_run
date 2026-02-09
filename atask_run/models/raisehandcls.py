''' 举手二分模型 raisehandcls.onnx

    作为 act 的子任务，要求必须存在 act_result 结果

    输出为：
        [ (m, 2), ... ], 其中 2 对应 (id, score)， 
            id 包含 
                0: 举手
                1: 托腮，
                2: 其它，
                3: 端书阅读, 
                -1: 未执行检测
            m 对应 act_result 的行号
'''

from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import Tuple
import logging
import cv2

logger = logging.getLogger("raisehandcls")

class Model_raisehandcls(AModel):
    def _preprocess(self, task: ATask):
        '''
            task.inpdata 可能是单张图像，或多张图象, 后者使用元组存储
            task.data["act_result"]: [(m, 6), ... ]
            task.userdata["raisehandcls_cids"]: 需要关注的行为类别, 默认 (2,3,11)

            预处理：
                根据 act_result 的行为检测结果，从 task.inpdata 中抠出行为图像，
                resize 224 x 224
                HWC -> CHW
                float32 -116.28f, *0.0175f
        '''
        assert "act_result" in task.data
        cids = task.userdata.get("raisehandcls_cids", (2,3,11))

        raisehandcls_inp = []
        act_ids = []    # [ (b, aid), ... ]
        for b in range(len(task.data["act_result"])):
            acts = task.data["act_result"][b]
            if isinstance(task.inpdata, np.ndarray):
                img = task.inpdata
            else:
                img = task.inpdata[b]
            for aid, a in enumerate(acts):
                ## a: [x1,y1,x2,y2,score,cid]
                if int(a[5]) in cids:
                    x1, y1, x2, y2 = a[:4].astype(np.int32)
                    act_img = img[y1:y2, x1:x2, :]
                    if act_img.size == 0:
                        act_img = np.zeros((224, 224, 3), np.uint8)
                    act_img = cv2.resize(act_img, (224, 224))
                    inp = np.transpose(act_img, (2, 0, 1)).astype(np.float32)
                    inp = (inp - 116.28) * 0.0175
                    raisehandcls_inp.append(inp[None, ...])  # (1, 3, 224, 224)
                    act_ids.append((b, aid))

        if len(raisehandcls_inp):
            task.data["raisehandcls_inp"] = np.concatenate(raisehandcls_inp)
            task.data["raisehandcls_act_ids"] = act_ids
        else:
            task.data["raisehandcls_inp"] = np.empty((0, 3, 224, 224), np.float32)
            task.data["raisehandcls_act_ids"] = []

        ## 先准备好输出结果，默认都是 -1.0
        task.data["raisehandcls_result"] = [
            np.empty((len(acts), 2), np.float32) for acts in task.data["act_result"]
        ]
        for item in task.data["raisehandcls_result"]:
            item[...] = -1.0
 
    def _postprocess(self, task: ATask):
        '''
            后处理：
                从 raisehandcls_infer 提取结果，构造 raisehandcls_result [ (m, 2), ...]
        '''
        ## softmax
        def softmax(logits, axis=1):
            exp = np.exp(logits)
            return exp / np.sum(exp, axis=axis)[..., None]
        
        infer_out = softmax(task.data["raisehandcls_infer"])    # (n, 4), 需要在 axis=1 轴做 softmax

        for i in range(len(task.data["raisehandcls_act_ids"])):
            b, aid = task.data["raisehandcls_act_ids"][i]
            rid = np.argmax(infer_out[i]);
            score = infer_out[i][rid]
            task.data["raisehandcls_result"][b][aid] = [rid, score]

    def _infer(self, task: ATask):
        ## 好吧，模型不支持批次
        task.data["raisehandcls_infer"] = self._hlp_batch_infer(
            1, 
            task.data["raisehandcls_inp"], 
            np.empty((0, 3), np.float32)
        )