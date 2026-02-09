from ..atask import ATask
from ..amodel import AModel
import numpy as np
import os.path as osp
import re, copy
from .asr_post import sentence_postprocess, time_stamp_lfr6_onnx
from ..syllables_cal import estimate 

class Model_asr_sensevoice(AModel):
    def _preprocess(self, task: ATask):

        assert "asr_enc_inp" in task.data, f"'asr_enc_inp' not found in task.data, {task.data.keys()}"
        assert "asr_type" in task.data, f"'asr_type' not found in task.data, {task.data.keys()}"
        feats = task.data["asr_enc_inp"][0]

        if task.data["asr_type"] == "en":
            # 纯英文
            task.data["asr_sensevoice_inp"] = (feats, np.array([feats.shape[1]], dtype=np.int32), np.array([4], dtype=np.int32), np.array([14], dtype=np.int32))
     
        else:
            task.data["asr_sensevoice_inp"] = (feats, np.array([feats.shape[1]], dtype=np.int32), np.array([3], dtype=np.int32), np.array([15], dtype=np.int32))

    def _infer(self, task: ATask):
        assert "asr_sensevoice_inp" in task.data, f"'asr_sensevoice_inp' not found in task.data, {task.data.keys()}"
        if len(task.data["asr_dec_raw_tokens"][0]) > 0:
            task.data["asr_sensevoice_infer"] = self(task.data["asr_sensevoice_inp"])
        else:
            task.data["asr_sensevoice_infer"] = ([], [])
    
    def _postprocess(self, task: ATask):
        if not hasattr(self, "sensevoice_tokenizer"):
            from ..sentencepiece_tokenizer import SentencepiecesTokenizer
            model_path = osp.dirname(self.model_path())
            token_fname = osp.join(model_path, "asr_sensevoice", "chn_jpn_yue_eng_ko_spectok.bpe.model")
            self.sensevoice_tokenizer = SentencepiecesTokenizer(bpemodel=token_fname)

        assert "asr_sensevoice_infer" in task.data and "asr_type" in task.data
        ctc_logits, encoder_out_lens = task.data["asr_sensevoice_infer"][0], task.data["asr_sensevoice_infer"][1]
        if len(ctc_logits) == 0:
            task.data["asr_sensevoice_result"] = [{"preds": "", "timestamp": [], "raw_tokens": []}]
            return
        
        us_peaks = task.data["asr_stamp_infer"][1][0]

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
            token_int = yseq_unique[yseq_unique != 0].tolist()
            asr_res.append(self.sensevoice_tokenizer.decode(token_int))

        pred = asr_res[0].split(">")[-1]
        raw_tokens = pred.split(" ")
        if len(raw_tokens) == 0:
            task.data["asr_sensevoice_result"] = [{"preds": "", "timestamp": [], "raw_tokens": []}]
            return
        
        if task.data["asr_type"][0] != "en":
            # 非纯英文，不需要做时间戳处理
            task.data["asr_sensevoice_result"] = [{"preds": "".join(raw_tokens), "timestamp": [], "raw_tokens": raw_tokens}]
            return
        
        raw_tokens_new0 = []
        punc = 0
        last_c = ""
        for ii in range(len(raw_tokens)):
            ch = raw_tokens[ii]
            if len(ch) == 0:
                continue

            if (ii == 0 or last_c in [".", "?"]) and not ("I'" in ch or ch == "I"):
                ch = ch.lower()
            
            # 去除标点符号
            merge_list = []
            merge = ""
            for jj, c in enumerate(ch):
                punc = 0
                code = ord(c)
                if 0x0020 <= code <= 0x007F:
                    if not (c == "'" or ('A' <= c <= 'Z') or ('a' <= c <= 'z') or ('0' <= c <= '9')):
                        punc = 1
                else:
                    if not('\u4e00' <= c <= '\u9fff' or '\u0030' <= c <= '\u0039'):
                        punc = 1
                
                if punc == 0:
                    merge += c
                else:
                    if len(merge) > 0:
                        merge_list.append(merge)
                        merge = ""

            if len(merge) > 0:
                merge_list.append(merge)

            raw_tokens_new0 = raw_tokens_new0 + merge_list
            last_c = c
        
        raw_tokens_new1 = []
        for s in raw_tokens_new0:
            # 使用正则表达式找到所有连续的数字和字母序列
            if "'" in s:
                raw_tokens_new1.append(s)
                continue

            parts = re.findall(r'\d+|[a-zA-Z]+', s)
            raw_tokens_new1.extend(parts)

        for idx, p in enumerate(raw_tokens_new1):
            if idx >= 1 and p == "to" and raw_tokens_new1[idx - 1] == "1":
                raw_tokens_new1[idx] = "2"
            elif p == "i":
                raw_tokens_new1[idx] = "I"
                
        __, timestamp_raw = time_stamp_lfr6_onnx(us_peaks, copy.copy(raw_tokens_new1), 1)
        text_proc, timestamp_proc, _ = sentence_postprocess(raw_tokens_new1, timestamp_raw)
        
        if len(timestamp_proc) == 0:
            task.data["asr_sensevoice_result"] = [{"preds": "", "timestamp": [], "raw_tokens": []}]
            return

        crop_time_list = [[timestamp_proc[0][0], timestamp_proc[-1][1]]]
            
        vad_lenth = crop_time_list[0][1] - crop_time_list[0][0]
        syllables_num = []
        last_t = crop_time_list[0][0]
        for t in text_proc.split(" "):
            syllables_num.append(estimate(t) + len(t) * 0.1)
        
        timestamp_proc = []
        all_syllables = np.sum(syllables_num)
        for n in syllables_num:
            during = n / all_syllables * vad_lenth
            timestamp_proc.append([int(last_t), int(last_t + during)])
            last_t = int(last_t + during)

        asr_res_stamp= [{'preds': text_proc, 'timestamp': timestamp_proc, "raw_tokens": raw_tokens_new1}]
        task.data["asr_sensevoice_result"] = asr_res_stamp
        return
        