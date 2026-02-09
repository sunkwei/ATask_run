
import numpy as np
import Levenshtein, re, jieba, difflib, os
from .text_fusion import TextFusion
from .del_str import check_duplicate_words
fusion = TextFusion()
def max_same_info(s1, s2):
    """
    计算两个字符串的所有最长公共子串及其长度
    
    Args:
        s1, s2: 输入的字符串
        
    Returns:
        tuple: (公共子串列表, 最大长度)
    """
    if not s1 or not s2:
        return [], 0
    
    matcher = difflib.SequenceMatcher(None, s1, s2)
    matches = matcher.get_matching_blocks()
    
    # 找出最大长度
    max_size = 0
    for match in matches:
        if match.size > max_size:
            max_size = match.size
    
    if max_size == 0:
        return [], 0
    
    # 收集所有最大长度的公共子串
    common_substrings = []
    for match in matches:
        if match.size == max_size:
            common_sub = s1[match.a:match.a + match.size]
            common_substrings.append(common_sub)
    
    return common_substrings, max_size

def merge_asr(results_para, results_sensevoice):

    pred_txt = [[]] * len(results_para)         
    time_list = [[]] * len(results_para)
    crop_dex = []
    # 循环每个片段，ASR预测文本
    for i in range(len(results_para)):
        r0 = results_para[i]
        r1 = results_sensevoice[i]

        sub_time_list = r0["timestamp"]
        txt = r0["raw_tokens"]

        txt_sence = r1["raw_tokens"]
        if len(txt) == 0 or len(txt_sence) == 0:
            continue

        dex = r0["idx"]
        begin_ms = r0["begin_ms"]   

        language = r0["asr_type"]
        if language == "en":
            
            sub_time_list_sence = r1["timestamp"]
            new_time = np.array(sub_time_list_sence) + begin_ms

            pred_txt[dex] = txt_sence
            time_list[dex] = new_time.tolist()
        
        else:
            paraformer = r0["preds"].replace(" ", "")
            sencevoice = r1["preds"]
            if len(paraformer) == 0 or len(sencevoice) == 0:
                continue
            
            # 计算最小编辑距离
            dis = Levenshtein.distance(paraformer, sencevoice)
            similarity = 1 - dis / max(len(paraformer), len(sencevoice))
            common_sub_list, same_len = max_same_info(paraformer, sencevoice)

            if language == "mixed":
                # 中英混合
                if similarity <= 0.5:
                    continue

                new_time = np.array(sub_time_list) + begin_ms
                pred_txt[dex] = txt
                time_list[dex] = new_time.tolist()

            else:
                # 纯中文，两者比较，排除幻听
                if similarity <= 0.55 or (r0["audio_db"] < 0.005 and similarity <= 0.75):
                    f = 0
                    if similarity >= 0.35 and abs(len(paraformer) - len(sencevoice)) <= 1 and r0["audio_db"] >= 0.02 and min(len(paraformer), len(sencevoice)) >= 4:
                        # 如果音量较大，且相似度不是很小，字数差别比较小，且字数>=4个子，则保留
                        f = 1

                    if f == 0 and same_len < 4:
                        # 最大公共子串足够小
                        continue

                    if f == 0 and min(len(paraformer), len(sencevoice)) / max(len(paraformer), len(sencevoice)) >= 0.6:
                        # 纯中文时处理，在最大公共子串附近找一个最佳的文本
                        crop_txt = fusion(paraformer, sencevoice, common_sub_list)
                        begin_dex = paraformer.index(crop_txt)
                        end_dex = begin_dex + len(crop_txt)

                        new_time = np.array(sub_time_list)[begin_dex:end_dex] + begin_ms
                        pred_txt[dex] = list(crop_txt)
                        time_list[dex] = new_time.tolist()

                        # 这种情况的存一下，后面打标点时，这段做为单独的断句
                        crop_dex.append(dex)
                        continue
                    
                    else:
                        continue
            
                # 寻找叠字
                matches_aa0 = re.findall(r'(\w)\1', paraformer)
                matches_aa1 = re.findall(r'(\w)\1', sencevoice)
                
                if len(matches_aa0) == 0 or (matches_aa0 == matches_aa1):
                    # 没有叠字，或者叠字完全相同，不二次判断，直接采纳
                    new_time = np.array(sub_time_list) + begin_ms
                    pred_txt[dex] = txt
                    time_list[dex] = new_time.tolist()
                    continue
                
                del_dex = []
                # 有叠字的，二次判断
                for die in matches_aa0:
                    if die in matches_aa1:
                        # 两个asr都有这个叠字的，保留
                        continue
                    
                    dex0 = paraformer.index(die + die)   # 叠字位置
                    dex1_list = [i for i, x in enumerate(sencevoice) if x == die]   

                    if len(dex1_list) == 0:
                        # 如果sencevoice没有这个叠字，则跳过不处理
                        continue

                    find_bool = False
                    for d in dex1_list:
                        # 每个组选选出
                        # 每个组选出
                        condition0 = dex0 < len(paraformer) - 2 and d < len(sencevoice) - 1 \
                            and paraformer[dex0 + 2] == sencevoice[d + 1]
                        
                        # 现现的话
                        # 现在的话
                        condition1 = dex0 < len(paraformer) - 2 and d < len(sencevoice) - 2 \
                            and paraformer[dex0 + 2] == sencevoice[d + 2]
                        
                        # 特别喝喝
                        # 特别好喝
                        condition2 = dex0 > 0 and d > 1 and (paraformer[dex0 - 1] == sencevoice[d - 2])

                        if condition0:
                            del_dex.append(dex0 + 1)
                            find_bool = True
                            break

                        # 如果同时满足向前叠字、向后叠字，则根据分词情况判断
                        if condition1 and condition2:
                            tmp_txt = sencevoice[d-1:d+2]
                            split_r = jieba.lcut(tmp_txt, HMM=False)

                            if len(split_r) > 1:
                                if len(split_r[0]) > len(split_r[1]):
                                    txt[dex0] = sencevoice[d - 1]
                                    find_bool = True
                                    break

                                elif len(split_r[0]) < len(split_r[1]):
                                    txt[dex0 + 1] = sencevoice[d + 1]
                                    find_bool = True
                                    break

                        if condition1:
                            txt[dex0 + 1] = sencevoice[d + 1]
                            find_bool = True
                            break

                        if condition2:
                            txt[dex0] = sencevoice[d - 1]
                            find_bool = True
                            break

                    if not find_bool or not check_duplicate_words(die):
                        # 三种情况都不满足的，直接删掉其中一个字
                        del_dex.append(dex0 + 1)

                # 更新文本和时间戳列表
                new_time = np.array(sub_time_list) + begin_ms
                new_time = np.delete(new_time, del_dex, axis=0)
                txt_new = []
                for jj in range(len(txt)):
                    if jj in del_dex:
                        continue
                    txt_new.append(txt[jj])
                
                pred_txt[dex] = txt_new
                time_list[dex] = new_time.tolist()

    txt_list = []
    time_list_all = []
    crop_punc = []
    for i in range(len(pred_txt)):
        if i in crop_dex:
            crop_punc.append([len(txt_list) - 1, len(txt_list) + len(pred_txt[i]) - 1])
        txt_list = txt_list + pred_txt[i]
        time_list_all = time_list_all + time_list[i]
    
    return time_list_all, txt_list, crop_punc