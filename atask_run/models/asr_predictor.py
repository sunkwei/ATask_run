from ..atask import ATask
from ..amodel import AModel
import os.path as osp
import yaml
import numpy as np

class Model_asr_predictor(AModel):
    def _preprocess(self, task: ATask):
        assert "asr_enc_infer" in task.data, f"'asr_enc_infer' not found in task.data, {task.data.keys()}"
        enc_out = task.data["asr_enc_infer"][0]
        enc_mask = np.ones((1, 1, enc_out.shape[1]), dtype=np.float32)
        task.data["asr_predictor_inp"] = (enc_out, enc_mask)
    
    def _infer(self, task: ATask):
        assert "asr_predictor_inp" in task.data, f"'asr_predictor_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_predictor_infer"] = self(task.data["asr_predictor_inp"])
    
    def _postprocess(self, task: ATask):
        assert len(task.data["asr_predictor_infer"]) == 4