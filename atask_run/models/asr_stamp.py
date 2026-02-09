from ..atask import ATask
from ..amodel import AModel
from copy import copy
from .asr_post import sentence_postprocess, time_stamp_lfr6_onnx, isAllAlpha1, check_stamp_txt

Interjections = ["哇", "呀", "呢", "嗯", "啊", "吧", "啦", "哦", "嘿", "哟", "嘛", "哈", "呵"
                              "喂", "了", "呗", "诶", "哎", "噢", "幺", "呃", "哼", "呦"]
single_str = ["谁", "好", "对", "来", "你", "坐", "停", "看", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

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
        task.data["asr_dec_raw_tokens"] = []
        task.data["asr_type"] = []

        for i in range(len(token)):
            if len(token[i]) == 0 or len(set(token[i]) & set(Interjections)) == len(set(token[i])) \
                or (len(token[i]) == 1 and len(set(token[i]) & set(single_str)) == 0):
                # 空、单个语气词、单字且不是有意义的单字，跳过
                task.data["asr_dec_stamp"].append([])
                task.data["asr_dec_preds"].append("")
                task.data["asr_dec_raw_tokens"].append([])
                task.data["asr_type"] = [""]
                continue

            timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_cif_peak[i], copy(token[i]))
            text_proc, timestamp_proc, _ = sentence_postprocess(token[i], timestamp_raw)
            pred = sentence_postprocess(token[i])

            l = isAllAlpha1(token[i])
            if l == 2:
                language = "en"
            elif l == 1:
                language = "cn"
            else:
                language = "mixed"

            task.data["asr_type"].append(language)

            tokens, timestamp_proc = check_stamp_txt(pred[1], timestamp_proc, language)
            task.data["asr_dec_stamp"].append(timestamp_proc)
            task.data["asr_dec_preds"].append(text_proc)
            task.data["asr_dec_raw_tokens"].append(pred[1])
