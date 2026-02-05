from ..atask import ATask
from ..amodel import AModel
import numpy as np

class Model_asr_predictor(AModel):
    def _preprocess(self, task: ATask):

        assert "asr_enc_inp" in task.data, f"'asr_enc_inp' not found in task.data, {task.data.keys()}"
        assert "asr_type" in task.data, f"'asr_type' not found in task.data, {task.data.keys()}"
        feats = task.data["asr_enc_inp"][1]
        feats_len = task.data["asr_enc_inp"][2]

        if task.data["asr_type"] == "en":
            # 纯英文
            task.data["asr_sensevoice_inp"] = (feats, feats_len, np.array([4], dtype=np.int32), np.array([14], dtype=np.int32))
     
        else:
            task.data["asr_sensevoice_inp"] = (feats, feats_len, np.array([3], dtype=np.int32), np.array([15], dtype=np.int32))

    def _infer(self, task: ATask):
        assert "asr_sensevoice_inp" in task.data, f"'asr_sensevoice_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_sensevoice_infer"] = self(task.data["asr_sensevoice_inp"])
    
    def _postprocess(self, task: ATask):
        assert "asr_sensevoice_infer" in task.data and "asr_type" in task.data
        ctc_logits, encoder_out_lens = task.data["asr_sensevoice_infer"][0], task.data["asr_sensevoice_infer"][1]
        us_peaks = task.data["asr_stamp_infer"][1]

        asr_res = []
        for b in range(ctc_logits.shape[0]):
            
            seq_len = encoder_out_lens[b]
            
            # 一步完成argmax和序列截取
            x = ctc_logits[b, :seq_len, :]
            yseq = np.argmax(x, axis=-1)

            # 使用更高效的unique_consecutive实现
            if len(yseq) > 0:
                # 找到非连续重复的位置
                mask = np.ones(len(yseq), dtype=bool)
                mask[1:] = yseq[1:] != yseq[:-1]
                yseq_unique = yseq[mask]
            else:
                yseq_unique = yseq
            
            # 移除blank token
            token_int = yseq_unique[yseq_unique != self.blank_id].tolist()
            asr_res.append(self.tokenizer.decode(token_int))

        pred = asr_res[0].split(">")[-1]
        raw_tokens = pred.split(" ")
        
        