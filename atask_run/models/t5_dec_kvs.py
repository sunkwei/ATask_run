from ..atask import ATask
from ..amodel import AModel
import numpy as np
from typing import List
import os.path as osp
from ..zk_zhconv import convert

def build_input_kvs_feed(kvs):
    feeds = []
    for i in range(8):
        for j in range(4):
            feeds.append(kvs[i * 4 + j])
    return feeds

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

beam_size = 2
max_length = 100
length_penalty = 0.6

class Model_t5_dec_kvs(AModel):
    def _preprocess(self, task: ATask):

        topk_indices = task.data["initial_topk_indices"]
        topk_scores = task.data["initial_topk_scores"]
        # 创建初始候选
        beam_candidates = []
        for token_id, score in zip(topk_indices, topk_scores):
            candidate = BeamCandidate(
                tokens=[token_id],
                score=score,
                past_kvs=task.data["initial_past_kvs"],
                finished=(token_id == 1) # <eos>
            )
            beam_candidates.append(candidate)
        
        task.data["beam_candidates"] = beam_candidates

    def _infer(self, task: ATask):
        assert "beam_candidates" in task.data, f"'beam_candidates' not found in task.data, {task.data.keys()}"

        finished_candidates = []
        beam_candidates = task.data["beam_candidates"]
        tokens = task.data["tokens"]
        for step in range(max_length - 1):  # -1因为已经有一个初始token
            if len(beam_candidates) < beam_size:#not beam_candidates:
                break
                
            # 扩展所有活跃候选
            new_candidates = []
            
            for candidate in beam_candidates:
                if candidate.finished:
                    # 已完成序列，直接加入完成列表
                    finished_candidates.append(candidate)
                    continue

                dec_kvs_input = [np.array([[candidate.tokens[-1]]], np.int32), np.array([[1]], np.int32), \
                                 task.data["decoder_1st_input"][2], np.ones([1, len(tokens)], np.int32)]
                
                dec_kvs_input = dec_kvs_input + build_input_kvs_feed(candidate.past_kvs)
                # 解码下一个token
                dec_out = self(tuple(dec_kvs_input))
                logits = dec_out[0][0, 0, :]  # (vocab_size,)
                new_past_kvs = dec_out[1:]
                
                # 计算概率分布
                log_probs = np.log(self.softmax(logits))
                
                # 获取top-k个可能的下一token
                topk_indices = np.argpartition(log_probs, -beam_size)[-beam_size:]
                topk_scores = log_probs[topk_indices]            

                # 为当前候选生成新的候选
                for token_id, token_score in zip(topk_indices, topk_scores):
                    new_score = candidate.score + token_score
                    
                    # 应用长度惩罚
                    length = len(candidate.tokens) + 1
                    penalized_score = new_score / (length ** length_penalty)
                    
                    new_tokens = candidate.tokens + [token_id]

                    is_finished = (token_id == 1)
                    
                    # if is_finished:
                    #     continue
                    new_candidate = BeamCandidate(
                        tokens=new_tokens,
                        score=penalized_score,  # 使用惩罚后的分数
                        past_kvs=new_past_kvs,
                        finished=is_finished
                    )
                    
                    new_candidates.append(new_candidate)
                
                if topk_indices[-1] == 1:
                    continue

                if len(new_candidates) < beam_size:
                    break
            # 选择得分最高的beam_size个候选
            new_candidates.sort(key=lambda x: x.score, reverse=True)

            # if len(new_candidates) < beam_size:
            #     break
            beam_candidates = []
            
            for candidate in new_candidates[:beam_size]:
                if candidate.finished:
                    finished_candidates.append(candidate)
                else:
                    beam_candidates.append(candidate)

            # # 早停：如果所有候选都已结束
            if len(new_candidates) < beam_size:
                break
        
        # 处理剩余的候选
        finished_candidates.extend(beam_candidates)
        task.data["dec_kvs_output"] = finished_candidates

    def _postprocess(self, task: ATask):
        assert "dec_kvs_output" in task.data, f"'dec_kvs_output' not found in task.data, {task.data.keys()}"

        task.data["trans_post"] = ''
        if not task.data["dec_kvs_output"]:
            # 回退到原始贪心解码
            return 

        # 选择得分最高的候选
        best_candidate = max(task.data["dec_kvs_output"], key=lambda x: x.score)

        if not hasattr(self, "Token"):
            from tokenizers import Tokenizer
            model_path = osp.dirname(self.model_path())
            token_fname = osp.join(model_path, "t5", "tokenizer.json")
            self.Token = Tokenizer.from_file(token_fname)

        # 7. 解码为文本
        text_result = self.Token.decode(best_candidate.tokens, skip_special_tokens=True)
        translation = text_result.replace(",", "，").replace(".", "。").replace("?", "？")
        task.data["trans_post"] = convert(translation, 'zh-hans')