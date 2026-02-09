import numpy as np
from .del_str import join_chinese_and_english, isEnglish, isAllAlpha, cal_char_num

from .chinese_itn import chinese_to_num
class ScentencePost:
    # 后处理，包括字句生成句子（断句、标点）、句子情绪、itn
    def __init__(self):

        self.punc_dict = {"1":"", "2":"，", "3":"。", "4":"？", "5":"、"}
        self.punc_dict_inv = {"，":2, "。":3, "？":4, "、":5, ".":6}
        self.up_punc = {".":0, "。":1, "？":2}
        self.first_rm = ["哇", "呀", "呢", "嗯", "啊", "吧", "啦", "哦", "嘿", "哟", "嘛", "哈", "呵"
                              "喂", "了", "呗", "诶", "哎", "噢", "幺", "呃", "哼", "呦", "吗"]
        
        self.single_str = ["好", "对", "来", "你", "坐", "停", "看", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        self.mantra = ["哇", "嗯", "啊", "哦", "嘿", "哟",  "喂", "诶", "哎", "噢",  "呃", "呦"] # 口头禅
        self.fix_str_single = ["好", "对", "来", "你", "坐", "停", "看",  "来"]
        self.fix_str_double = ["请坐", "孩子们", "同学们", "同学们好", "同学好", "谁来", "你来", "你说"]
        self.teacher_str = ['按照', '根据', '刚刚', '刚', '讲', '那么', '那', '说', '提示', '提醒', '你们', '你', '大家', '要求', '请', '希望', '强调', '再']

         
    def __call__(self, txt_list_origin, time_list_origin, punc, split_dex, txt_label, mantra_data):
        
        punc = np.array(punc)
        question_dex = np.where(punc == 4)[0]
        juhao_dex = np.where(punc == 3)[0]
        douhao_dex = np.where(punc == 2)[0]

        douhao = list(set(douhao_dex) | set(split_dex))

        punc[:] = 1
        punc[douhao] = 2      # 停顿处加逗号
        punc[-1] = 3             # 最后一个字后句号
        punc[question_dex] = 4   # 把问号赋值上
        punc[juhao_dex] = 3          # 把停顿处的句号赋值上

        new_txt_label = []
        for i in range(len(txt_label)):
            if i == 0:
                new_txt_label.append(txt_label[i])
                continue

            curr_label = txt_label[i]
            if curr_label != new_txt_label[-1] and punc[i-1] < 2 and punc[i] >= 2:
                curr_label = new_txt_label[-1]
            
            if i >=2 and curr_label != new_txt_label[-1] and punc[i-1] < 2 and punc[i-2] >= 2: 
                new_txt_label[-1] = curr_label

            new_txt_label.append(curr_label)
        
        txt_label = new_txt_label

        txt = []
        time_ramge = []
        scentence_txt = []
        scentence_time = []
        scentence_role = []
        # 统计字数、语速
        speak_time = 0
        speak_word = 0
        word_count = 0
        en_dex = []
        scentence_num = 0
        for i in range(len(time_list_origin)):
            
            curr_l = txt_label[i]
            curr_punc = punc[i]
            if i == len(time_list_origin) - 1:
                time_ramge.append(time_list_origin[i])
                txt.append(txt_list_origin[i])
                
                if "、" in txt:
                    txt_num = len(txt) - 1
                else:
                    txt_num = len(txt)

                word_count += txt_num
                if txt_num >= 10:
                    # 只统计10个字的语速
                    speak_word += txt_num
                    speak_time += (time_ramge[-1][1] - time_ramge[0][0])

                new_txt, en_num = join_chinese_and_english(txt)
                if en_num > 0:
                    # 存在英文，存下序号
                    en_dex.append(scentence_num)

                if isEnglish(txt[-1]) and str(curr_punc) == "3":
                    scentence_txt.append(new_txt + ".")
                elif str(curr_punc) != "5":
                    scentence_txt.append(new_txt + self.punc_dict[str(curr_punc)])
                else:
                    scentence_txt.append(new_txt)

                scentence_num += 1
                scentence_time.append([time_ramge[0][0], time_ramge[-1][1]])
                scentence_role.append(curr_l)
                continue

            next_l = txt_label[i + 1]

            time_ramge.append(time_list_origin[i])
            txt.append(txt_list_origin[i])
            if curr_punc == 5:
                txt.append("、")

            if (curr_punc > 1 and curr_punc < 5) or next_l != curr_l:
                
                if "、" in txt:
                    txt_num = len(txt) - 1
                else:
                    txt_num = len(txt)

                word_count += txt_num
                if txt_num >= 10:
                    speak_word += txt_num
                    speak_time += (time_ramge[-1][1] - time_ramge[0][0])

                new_txt, en_num = join_chinese_and_english(txt)
                
                if en_num > 0:
                    # 存在英文，存下序号
                    en_dex.append(scentence_num)

                # 英文时。改为.
                if isEnglish(txt[-1]) and str(curr_punc) == "3":
                    scentence_txt.append(new_txt + ".")
                elif str(curr_punc) != "5":
                    scentence_txt.append(new_txt + self.punc_dict[str(curr_punc)])
                else:
                    scentence_txt.append(new_txt)

                scentence_num += 1
                scentence_time.append([time_ramge[0][0], time_ramge[-1][1]])
                scentence_role.append(curr_l)

                txt = []
                time_ramge = []

        # 遍历找一句话角色分开的情况
        merge_list = []
        cache_list = []
        for i in range(len(scentence_txt)):
            part_txt = scentence_txt[i]
            if part_txt[-1] not in self.punc_dict_inv:
                cache_list.append(i)
                continue

            else:
                if len(cache_list) > 0:
                    cache_list.append(i)
                    merge_list.append(cache_list)
                    cache_list = []
                else:
                    merge_list.append([i])

        # 合并后，数据重构
        role_list = []
        time_list = []
        txt_list = []
        for i in range(len(merge_list)):
            if len(merge_list[i]) == 1:
                txt_list.append(scentence_txt[merge_list[i][0]])
                role_list.append(scentence_role[merge_list[i][0]])
                time_list.append(scentence_time[merge_list[i][0]])
                continue
            
            merge_dx = merge_list[i]
            txt = ""
            role_time_duration = [0, 0]
            role_set = []
            seg_time = []
            seg_txt = []
            for d in merge_dx:
                if (len(txt) > 0 and isEnglish(txt[-1])) or (len(txt) > 0 and isEnglish(scentence_txt[d][0])):
                    txt = txt + " " + scentence_txt[d]
                
                else:
                    txt = txt + scentence_txt[d]

                role_time_duration[scentence_role[d]] = role_time_duration[scentence_role[d]] + (scentence_time[d][1] - scentence_time[d][0])
                role_set.append(scentence_role[d])
                seg_time.append(scentence_time[d])
                seg_txt.append(scentence_txt[d])
                
            # 如果一个断句中，第一词是“你说”或“你来说”，且角色是老师，则第一个词老师，其余归学生
            if ("你说" in txt or "你来说" in txt) and len(set(role_set)) == 2 \
                and role_set[0] == 0 and (seg_txt[0][-2:] == "你说" or seg_txt[0][-3:] == "你来说"):
                txt_list.append(seg_txt[0] + "，")
                time_list.append(seg_time[0])
                role_list.append(0)
                
                txt_list.append("".join(seg_txt[1:]))
                time_list.append([seg_time[1][0], seg_time[-1][1]])
                role_list.append(1)        
                continue
            
            # 如果一个断句中既有老师和学生，且最后一个是问号，则问句的文字归老师，其余归学生
            if len(set(role_set)) == 2 and txt[-1] in ["?", "？"] and role_set[-1] == 0:
                
                if role_time_duration[0] > role_time_duration[1]:
                    # 如果老师声纹占比高，就整段当作老师
                    txt_list.append(txt)
                    time_list.append([scentence_time[merge_dx[0]][0], scentence_time[merge_dx[-1]][1]])
                    role_list.append(0)
                    continue

                for ii in range(len(role_set) - 1, -1, -1):
                    if role_set[ii] == 1:
                        break
                
                # 后半部分问句文本组合起来
                txt_back = ''
                for jj in range(ii + 1, len(role_set)):
                    if len(txt_back) > 0 and isEnglish(txt_back[-1]) or isEnglish(seg_txt[jj][0]):
                        txt_back = txt_back + " " + seg_txt[jj]
                    else:
                        txt_back = txt_back + seg_txt[jj]

                # print (txt_back, cal_char_num(txt_back))
                if cal_char_num(txt_back) < 3:
                    # 如果老师的问句少于3个字，则不处理
                    txt_list.append(txt)
                    time_list.append([scentence_time[merge_dx[0]][0], scentence_time[merge_dx[-1]][1]])
                    role_list.append(1)
                    continue

                # 前半部分学生文本组合起来
                txt_front = ''
                for jj in range(ii + 1):
                    if len(txt_front) > 0 and isEnglish(txt_front[-1]) or isEnglish(seg_txt[jj][0]):
                        txt_front = txt_front + " " + seg_txt[jj]
                    else:
                        txt_front = txt_front + seg_txt[jj]

                txt_list.append(txt_front + "，")
                time_list.append([seg_time[0][0], seg_time[ii][1]])
                role_list.append(1)

                txt_list.append(txt_back)
                time_list.append([seg_time[ii + 1][0], seg_time[-1][1]])
                role_list.append(0)

            else:
                txt_list.append(txt)
                time_list.append([scentence_time[merge_dx[0]][0], scentence_time[merge_dx[-1]][1]])
                role_list.append(int(np.argmax(role_time_duration)))

        data = []
        num = 0
        last_up = -1
        for i in range(len(txt_list)):
            
            part_txt = txt_list[i]
            part_time = time_list[i]
            part_role = role_list[i]
            
            try:
                part_txt = chinese_to_num(part_txt)
            except:
                pass

            if len(part_txt) == 2 and part_txt[0] in self.fix_str_single and part_role == 1:
                # 如果是老师单字专用词，强制老师角色
                part_role = 0

            elif any(word in part_txt for word in self.fix_str_double) and part_role == 1:
                part_role = 0
            
            if "老师" in part_txt and part_role == 1 and i > 0 and i < len(txt_list) - 1 and (role_list[i - 1] == 0 or role_list[i + 1] == 0):
                # 如果包含“老师”，且存在固定词语，且前后有老师角色，则强制老师角色
                mt = 0
                for t in self.teacher_str:
                    if t in part_txt:
                        mt += 1
                
                if mt > 0:
                    part_role = 0

            if len(part_txt) > 0 and i in en_dex and not isAllAlpha(part_txt):
                # 中英混合中，单英文强制大写
                tmp_list = part_txt.split(" ")
                part_txt_new = ""
                for tmp in tmp_list:
                    if len(tmp) == 1:
                        part_txt_new = part_txt_new + tmp.upper()
                    else:
                        part_txt_new = part_txt_new + tmp

                part_txt = part_txt_new
            
            trans_txt = ''

            if len(part_txt) > 0 and i in en_dex and isAllAlpha(part_txt):
                trans_txt = part_txt

            if len(part_txt) > 0 and i in en_dex and isEnglish(part_txt[0]) and (last_up == -1 or last_up == 1):
                # 首字母大写
                part_txt = part_txt[0].upper() + part_txt[1:]

            if len(part_txt) > 0 and part_txt[-1] in self.up_punc:
                last_up = 1
            else:
                last_up = 0
            
            scentence_data = {"begin":part_time[0], "end":part_time[1], "seg_num":num, "transcript":part_txt, 
                                "emotion":{}, "time_list":[], "role":part_role, "translation":trans_txt}
            num += 1
            data.append(scentence_data)
        
        word_count = sum([cal_char_num(item["transcript"]) for item in data])
        statistics = {"word_count":word_count, "keywords":{}, "mantra_count":mantra_data}
        if speak_time > 0:
            statistics["speed"] = speak_word // (speak_time / 60)  # 每分钟
        else:
            statistics["speed"] = 0
        R = {}
        R["data"] = data
        R["statistics"] = statistics

        return R
        # if len(trans_lenth) > 0:
        #     sort_dex = np.argsort(trans_lenth)
        #     for kk in range(len(sort_dex)):
        #         dd = sort_dex[kk]
        #         t = trans_task[dd]

        #         if kk == len(sort_dex) - 1:
        #             t.last = 1

        #         self.queue_0.put(t)

        #     print ("waiting for translation")
        #     for kk in range(len(sort_dex)):
        #         dd = sort_dex[kk]
        #         t = trans_task[dd]
        #         t.finished_env.wait()

        #         data[t.idx]["translation"] = t.trans_txt

        # if self.trans_model:
        #     self.trans_model.reset()

        # word_count = sum([cal_char_num(item["transcript"]) for item in data])
        # statistics = {"word_count":word_count, "keywords":{}, "mantra_count":mantra_data}
        # if speak_time > 0:
        #     statistics["speed"] = speak_word // (speak_time / 60)  # 每分钟
        # else:
        #     statistics["speed"] = 0
        # R = {}
        # R["data"] = data
        # R["statistics"] = statistics
        # # print (R)
        # return R

if __name__ == "__main__":
    SP = ScentencePost()

    txt = ["的", "吗一","般", "在", "引", "导", "词", "前","面", "停"]
    txt = ["how", "long", "is","your" ,"当", "我","们","说","身","高","的","时","候", "我","们","用","how","提","问","但","是","说","形","容","一","个","物","体","的","长","度","呢"]
    ss = " ".join(txt)
    s = SP.model_puc(ss)
    print (s)
    

