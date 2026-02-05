from ..atask import ATask
from ..amodel import AModel
import numpy as np
import os.path as osp
from .asr_post import TokenIDConverter, Hypothesis
import json
import logging

logger = logging.getLogger("asr_dec")

class Model_asr_dec(AModel):
    
    def _preprocess(self, task: ATask):
        assert "asr_enc_infer" in task.data and "asr_predictor_infer" in task.data, \
            f"'asr_enc_infer' or 'asr_predictor_infer' not found in task.data, {task.data.keys()}"
        assert "asr_enc_mask" in task.data, f"'asr_enc_mask' not found in task.data, {task.data.keys()}"

        enc = task.data["asr_enc_infer"][0]
        # enc_mask = np.ones((1, 1, enc.shape[1]), dtype=np.float32)
        enc_mask = task.data["asr_enc_mask"]

        if self.debug:
            logger.info(f"enc_out: {enc.shape}, asr_enc_length: {np.squeeze(np.sum(enc_mask, axis=-1))}")

        pre_acoustic_embeds, pre_token_length = task.data["asr_predictor_infer"][:2]
        B = len(pre_token_length)
        max_length = int(np.max(pre_token_length))
        pre_token_mask = np.zeros((B, 1, max_length), np.float32)
        for i in range(B):
            length = int(pre_token_length[i])
            pre_token_mask[i, 0, :length] = 1.0
        task.data["asr_dec_inp"] = (enc, enc_mask, pre_acoustic_embeds, pre_token_mask)
        
    def _infer(self, task: ATask):
        assert "asr_dec_inp" in task.data, f"'asr_dec_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_dec_infer"] = self(task.data["asr_dec_inp"])
        
    def _postprocess(self, task: ATask):
        if not hasattr(self, "tokenizer"):
            model_path = osp.dirname(self.model_path())
            token_fname = osp.join(model_path, "asr_dec", "tokens.json")
            with open(token_fname, encoding="utf-8") as f:
                self.tokenizer = TokenIDConverter(json.load(f))
        logits = task.data["asr_dec_infer"][0]  # (B, T, vocab_size)
        
        task.data["asr_dec_token"] = []

        ## 批次
        B = len(logits)

        for i in range(B):
            token_len = int(np.squeeze(np.sum(task.data["asr_dec_inp"][3][i], axis=1)))

            # yseq = logits[i].argmax(axis=-1)    ## [T]
            # score = logits[i].max(axis=-1)
            # score = np.sum(score, axis=-1)

            yseq = np.argmax(logits[i, :token_len, :], axis=-1)
            score = np.max(logits[i, :token_len, :], axis=-1)
            score = np.sum(score, axis=-1)

            # pad with mask tokens to ensure compatibility with sos/eos tokens
            # asr_model.sos:1  asr_model.eos:2
            yseq = np.array([1] + yseq.tolist() + [2])
            hyp = Hypothesis(yseq=yseq, score=score)

            # remove sos/eos and get results
            last_pos = -1
            token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x not in (0, 2), token_int))

            # Change integer-ids to tokens
            token = self.tokenizer.ids2tokens(token_int)    ## ['有','句', ... ]

            task.data["asr_dec_token"].append(token)

