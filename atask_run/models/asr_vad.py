from ..amodel import AModel
from ..model_id import DO_ASR_VAD
import os.path as osp
import numpy as np
from .asr_vad_impl import Fsmn_vad
from typing import Tuple
import logging

logger = logging.getLogger("vad")

'''
    FIXME: 好吧，asr vad 任务应该是个独立的步骤，有上下文关系
        初始化 vad 后，循环调用 update() 更新 pcm，返回切割点
        
        每个切割点都是一个 Tuple，包含 (start_ms, end_ms)，时间相对 reset() update 的 pcm 时长
        一般来说为了 asr 方便，需要将切割点进行合并，并防止超长

        asr 使用最大 15秒输入，所以每个合并段不超过 15秒，若连续段间隔小于 1秒，则合并
'''
class Model_asr_vad:
    def __init__(self, model_path: str, **kwargs):
        # super().__init__("asr_vad", model_path, DO_ASR_VAD, backend, **kwargs)
        model_path = osp.dirname(model_path)
        self.__impl = Fsmn_vad(
            cmvn_file=osp.join(model_path, "am.mvn"),
            config_file=osp.join(model_path, "config.yaml"),
            model_file=osp.join(model_path, "asr_vad.onnx"),
        )
        self.__segs = []            ## reset() 后，所有原始切割段
        self.__asr_next_index = 0  ## asr 调用，下一片应该返回的片段索引，相对 self.__segs
    
    def reset(self):
        self.__impl.reset()
        self.__segs = []
        self.__asr_next_index = 0

    def get_all_segs(self):
        '''
        返回所有原始切割段
        '''
        return self.__segs
    
    def update(self, pcm:np.ndarray | None, last:bool=False) -> list:
        '''
            不断输入 pcm 数据，若 pcm 为 None，则表示结束
            返回一个 List[Tuple]，其中每个 Tuple 包含 (start_ms, end_ms)
        '''
        if not last:
            segs = self.__impl.update(pcm)
        else:
            segs = self.__impl.update(pcm, last=last)

        self.__segs.extend(segs)
        return segs
        
    def update_for_asr(
        self, pcm:np.ndarray | None, 
        max_duration_ms:int=15000, 
        max_merge_interval_ms:int=1200,
        last:bool=False
    ) -> Tuple[int, int]:
        '''
            将返回适合 asr 的 vad 片段。

            asr 模型输入为之多15秒的 pcm，因此尽量合并片段接近 15 秒效率最高
            如果 vad 片段之间间隔超过 max_merge_interval，则不能合并，否则 asr 会“幻听”

            原始 vad 每个片段都小于等于 8 秒
        '''
        self.update(pcm, last)
        if self.__asr_next_index >= len(self.__segs):
            return (-1, -1)
        
        head = self.__asr_next_index
        tail = head + 1

        trigger = False
        while tail < len(self.__segs):
            ## 检查累积时长是否超过 max_duration
            if self.__segs[tail][1] - self.__segs[head][0] >= max_duration_ms:
                trigger = True
                break

            ## 检查段间间隔是否大于 max_merge_interval
            if self.__segs[tail][0] - self.__segs[tail-1][1] > max_merge_interval_ms:
                trigger = True
                break

            tail += 1

        if trigger:
            ## 返回 [head, tail) 之间的片段
            self.__asr_next_index = tail
            return (self.__segs[head][0], self.__segs[tail-1][1])
        elif pcm is None:
            ## 最后返回剩余片段
            ret = self.__segs[head:]
            self.__asr_next_index = tail
            return (self.__segs[head][0], self.__segs[tail-1][1])
        else:
            return (-1, -1)
