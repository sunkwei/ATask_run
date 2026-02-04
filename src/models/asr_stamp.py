from ..atask import ATask
from ..amodel import AModel
import numpy as np
from copy import copy
from .asr_post import sentence_postprocess, time_stamp_lfr6_onnx

class Model_asr_stamp(AModel):
    def preprocess(self, task: ATask):
        assert "asr_enc_infer" in task.data and "asr_predictor_infer" in task.data

        task.data["asr_stamp_inp"] = (
            task.data["asr_enc_infer"][0],      ## enc
            np.ones((1, 1, task.data["asr_enc_infer"][0].shape[1]), np.float32),    ## enc_mask
            task.data["asr_predictor_infer"][1],       ## pre_token_length
        )

    def infer(self, task: ATask):
        task.data["asr_stamp_infer"] = self.impl().infer(task.data["asr_stamp_inp"])

    def postprocess(self, task: ATask):
        assert "asr_dec_token" in task.data
        us_alphass, us_cif_peak = task.data["asr_stamp_infer"]
        token = task.data["asr_dec_token"]
        timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_cif_peak[0], copy(token))
        text_proc, timestamp_proc, _ = sentence_postprocess(token, timestamp_raw)
        task.data["asr_dec_stamp"] = timestamp_proc
        task.data["asr_dec_preds"] = text_proc
