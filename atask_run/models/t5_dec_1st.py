from ..atask import ATask
from ..amodel import AModel
import numpy as np

class Model_t5_dec_1st(AModel):
    def _preprocess(self, task: ATask):
        assert "encoder_infer" in task.data, f"'encoder_infer' not found in task.data, {task.data.keys()}"

        input_id = np.array([[0]], np.int32)
        input_mask = np.array([[1]], np.int32)
        enc_hidden_state = task.data["encoder_infer"][0]
        enc_attention_mask = task.data["encoder_input"][1]

        task.data["decoder_1st_input"] = (input_id, input_mask, enc_hidden_state, enc_attention_mask)

    def _infer(self, task: ATask):
        assert "decoder_1st_input" in task.data, f"'decoder_1st_input' not found in task.data, {task.data.keys()}"
        task.data["decoder_1st_infer"] = self(task.data["decoder_1st_input"])
    
    def _postprocess(self, task: ATask):
        assert "decoder_1st_infer" in task.data, f"'decoder_1st_infer' not found in task.data, {task.data.keys()}"
        task.data["initial_logits"] = task.data["decoder_1st_infer"][0][0, 0, :]
        task.data["initial_past_kvs"] = task.data["decoder_1st_infer"][1:]

        # 计算初始token的概率分布
        log_probs = np.log(self.softmax(task.data["initial_logits"]))
        
        # 获取top-k个候选token
        beam_size = 2
        topk_indices = np.argpartition(log_probs, -beam_size)[-beam_size:]
        topk_scores = log_probs[topk_indices]

        task.data["initial_topk_indices"] = topk_indices
        task.data["initial_topk_scores"] = topk_scores