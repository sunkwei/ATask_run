from ..atask import ATask
from ..amodel import AModel
import os.path as osp
import yaml
import numpy as np

class Model_asr_predictor(AModel):
    def _preprocess(self, task: ATask):
        assert "asr_enc_infer" in task.data, f"'asr_enc_infer' not found in task.data, {task.data.keys()}"
        assert "asr_enc_mask" in task.data, f"'asr_enc_mask' not found in task.data, {task.data.keys()}"
        enc_out = task.data["asr_enc_infer"][0]
        enc_mask = task.data["asr_enc_mask"]
        task.data["asr_predictor_inp"] = (enc_out, enc_mask)
    
    def _infer(self, task: ATask):
        assert "asr_predictor_inp" in task.data, f"'asr_predictor_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_predictor_infer"] = self(task.data["asr_predictor_inp"])
    
    def _postprocess(self, task: ATask):
        assert len(task.data["asr_predictor_infer"]) == 4 # (pre_acoustic_embeds, pre_token_length, alpha, pre_peak_index)