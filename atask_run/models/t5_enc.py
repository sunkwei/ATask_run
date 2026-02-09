from ..atask import ATask
from ..amodel import AModel
import os.path as osp
import numpy as np
from typing import List

class BeamCandidate:
    """Beam Search候选序列"""
    def __init__(self, tokens: List[int], score: float, past_kvs: List[np.ndarray], finished: bool = False):
        self.tokens = tokens  # 当前序列的token IDs
        self.score = score    # 序列得分（对数概率和）
        self.past_kvs = past_kvs  # 当前的past_kvs状态
        self.finished = finished  # 是否已结束（遇到EOS）
    
    def __lt__(self, other):
        # 用于堆排序，得分高的优先
        return self.score > other.score
    
class Model_t5_enc(AModel):
    def _preprocess(self, task: ATask):
        if not hasattr(self, "Token"):
            from tokenizers import Tokenizer
            model_path = osp.dirname(self.model_path())
            token_fname = osp.join(model_path, "t5", "tokenizer.json")
            self.Token = Tokenizer.from_file(token_fname)

        text = f"translate to zh: {task.inpdata}"
        task.data["text"] = text
        tokens = self.Token.encode(text).ids
        task.data["tokens"] = tokens
        task.data["encoder_input"] = (np.array([tokens], dtype=np.int32), np.ones((1, len(tokens)), dtype=np.int32))

    def _infer(self, task: ATask):
        assert "encoder_input" in task.data, f"'encoder_input' not found in task.data, {task.data.keys()}"
        task.data["encoder_infer"] = self(task.data["encoder_input"])
    
    def _postprocess(self, task: ATask):
        pass