from ..atask import ATask
from ..amodel import AModel
import numpy as np
from copy import copy
from .asr_post import sentence_postprocess, time_stamp_lfr6_onnx

class Model_asr_stamp(AModel):
    def _preprocess(self, task: ATask):
        assert "asr_enc_infer" in task.data and "asr_predictor_infer" in task.data
        assert "asr_enc_mask" in task.data

        task.data["asr_stamp_inp"] = (
            task.data["asr_enc_infer"][0],      ## enc
            task.data["asr_enc_mask"],          ## enc mask
            task.data["asr_predictor_infer"][1],  ## (pre_acoustic_embeds, pre_token_length, alphas, peek_index)
        )

    def _infer(self, task: ATask):
        task.data["asr_stamp_infer"] = self(task.data["asr_stamp_inp"])

    def _postprocess(self, task: ATask):
        assert "asr_dec_token" in task.data
        us_alphass, us_cif_peak = task.data["asr_stamp_infer"]
        token = task.data["asr_dec_token"]  ## List[List[char]]
        task.data["asr_dec_stamp"] = []
        task.data["asr_dec_preds"] = []

        for i in range(len(token)):
            timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_cif_peak[i], copy(token[i]))
            text_proc, timestamp_proc, _ = sentence_postprocess(token[i], timestamp_raw)
            task.data["asr_dec_stamp"].append(timestamp_proc)
            task.data["asr_dec_preds"].append(text_proc)
