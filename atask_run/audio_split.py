import numpy as np
import jieba,re
from .del_str import join_chinese_and_english, cal_txt_len

class AudioSplit:
    # 声音切分类
    def __init__(self):
        self.short_time = 0.5
        # self.seg_model = pkuseg.pkuseg(model_name=model_dir)
    
    def cal_leaf_txt(self, txt_total, txt):
        num = 1
        while txt_total[:num] != txt:
            if num >= len(txt_total):
                break

            num += 1
        return txt_total[num:]
    
    def get_feat(self, cur_seg_time, txt_list):
        # 获取分割点以及提取特征
        time_during = cur_seg_time[-1, 1] - cur_seg_time[0, 0]
        split_point = []
        if time_during <= 1:
            # <=1秒的片段，不判断，直接成一段
            split_point.append([cur_seg_time[0, 0], cur_seg_time[-1, 1]])
            
        else:
            # 以1.5秒的窗口滑动，根据字词寻找切割点，划分片段
            start_point = cur_seg_time[0, 0]
            txt_use = []
            while True:

                end_point = start_point + 1.5

                wind_range = np.array([start_point, end_point])

                area_list = self.get_coincident_area(wind_range, cur_seg_time)
                dex = np.where(area_list > 0.001)[0]

                if len(dex) == 0:
                    # 区间内没字，跳过
                    continue
                
                elif len(dex) == 1:
                    # 区间内只有一个字，只包含这一个字
                    last_point = cur_seg_time[dex[0], 1]
                    split_point.append([start_point, last_point])
                    txt_use.append(txt_list[dex[0]])
                    start_point = last_point
                
                elif dex[-1] == len(cur_seg_time) - 1:

                    split_point.append([start_point, cur_seg_time[-1, 1]])
                    break

                else:
                    last_b = cur_seg_time[dex[-1], 0]
                    last_e = cur_seg_time[dex[-1], 1]

                    diff = last_b - cur_seg_time[dex[-1] - 1, 1] - (cur_seg_time[dex[-1] + 1, 0] - last_e)
                    before_txt = txt_list[dex[0]:dex[-1]]
                    only_one_t = 0
                    if len(before_txt) == 1:
                        t = before_txt[0]
                        if cal_txt_len(t) == 1:
                            only_one_t = 1

                    if diff <= 0 or cur_seg_time[dex[-1] - 1, 1] - start_point <= 0.7 or only_one_t:
                        # 如果这个词跟前面一个词更近或者前面累计的词时间长度<0.7秒且前面累计的就一个字）
                        split_point.append([start_point, last_e])
                        start_point = last_e
                        txt_use = txt_use + txt_list[dex[0]:dex[-1]+1]
                    
                    else:

                        split_point.append([start_point, cur_seg_time[dex[-1] - 1, 1]])
                        start_point = cur_seg_time[dex[-1] - 1, 1]
                        txt_use = txt_use + txt_list[dex[0]:dex[-1]]

                leaf_txt = self.cal_leaf_txt(txt_list, txt_use)
                if cur_seg_time[-1, 1] - start_point <= self.short_time:
                    # 最后剩不到0.5秒时，合并到前面去
                    split_point[-1][1] = cur_seg_time[-1, 1]
                    break

                if len(leaf_txt) == 1:
                    # 最后剩单个汉字、单个英文单词的，合并到前面去
                    t = leaf_txt[0]
                    if cal_txt_len(t) == 1:
                        split_point[-1][1] = cur_seg_time[-1, 1]
                        break
        
        return split_point 
    
    def get_coincident_area(self, box0, box_arr):
        # 计算重叠面积
        if len(box0) * len(box_arr) == 0:
            return np.array([0])
            
        xx1 = np.maximum(box0[0], box_arr[:,0])
        xx2 = np.minimum(box0[1], box_arr[:,1])

        w = np.maximum(0.0, xx2 - xx1)

        return w
    
    def crop_winds(self, time_list, txt_list, english_class=0):
        
        seg_time = []
        seg_txt = []
        split_dex = []
        if len(time_list) == 0:
            return [], [], []

        # # 以0.2s的间隔，分割片段
        if english_class:
            crop_thresh = 0.5
        else:
            crop_thresh = 0.3
        last_time = time_list[0][0]
        
        sub_seg_time = []
        sub_seg_txt = []
        num = -1
        
        for ii in range(len(time_list)):
            num += 1
            next_time = time_list[ii][0]
            next_txt = txt_list[ii]
            if next_time - last_time >= crop_thresh:
                seg_time.append(sub_seg_time)
                seg_txt.append(sub_seg_txt)
                sub_seg_time = [time_list[ii]]
                sub_seg_txt = [next_txt]  
                split_dex.append(num - 1)
                
            else:
                sub_seg_time.append(time_list[ii])
                sub_seg_txt.append(next_txt)
            
            if ii == len(time_list) - 1 and len(sub_seg_time) > 0:
                seg_time.append(sub_seg_time)
                seg_txt.append(sub_seg_txt)
                split_dex.append(num)
                break

            last_time = time_list[ii][1]

        return seg_time, seg_txt, split_dex

    def seg_pk(self, seg_time, seg_txt):
        # 对分割完的片段，每个片段内部分词
        f_time_list = []
        f_txt_list = []
        all_time_list = []
        all_txt_list = []
        for ii in range(len(seg_txt)):

            sub_txt_list = seg_txt[ii]
            sub_time_list = seg_time[ii]

            txt_merge, has_en = join_chinese_and_english(sub_txt_list)
            if has_en > 0:
                txt_seg = sub_txt_list
                f_txt_list.append(sub_txt_list)
                f_time_list.append(sub_time_list)
                continue
            
            protected_text = re.sub(r"([a-zA-Z]+'[a-zA-Z]+)", r" \1 ", txt_merge)
            txt_seg = jieba.lcut(protected_text)
            # has_p = 0
            # if "'" in txt_seg:
            #     return seg_time, seg_txt
            #     txt_seg_tmp = []
            #     for t in txt_seg:
            #         if t != "'":
            #             if has_p and t != " ":
            #                 if len(txt_seg_tmp) > 0:
            #                     txt_seg_tmp[-1] = txt_seg_tmp[-1] + t
                            
            #             else:
            #                 txt_seg_tmp.append(t)
            #             has_p = 0

            #         else:
            #             if len(txt_seg_tmp) > 0:
            #                 txt_seg_tmp[-1] = txt_seg_tmp[-1] + t
            #             has_p = 1
            #     txt_seg = txt_seg_tmp
            #     # text = ' '.join(txt_seg)
            #     # merged_text = re.sub(r"(\w+) ' (\w+)", r"\1'\2", text)
            #     # txt_seg = merged_text.split()

            # if " " in txt_seg:
            #     txt_seg = [x for x in txt_seg if x.strip()] 

            # print (txt_seg)
            # if has_en == 0:
            #     txt_seg = self.seg_model.cut(txt_merge)
            
            # else:
            #     # 如果里面有英文，先将英文替换成A，防止分词，把一个单词分开了
            #     tl = txt_merge.split(" ")
            #     txt_merge_new = []
            #     eng = []
            #     for t in tl:
            #         if isEnglish(t):
            #             txt_merge_new.append("A")
            #             eng.append(t)
            #         else:
            #             txt_merge_new.append(t)

            #     txt_merge_new = " ".join(txt_merge_new)
            #     txt_seg = self.seg_model.cut(txt_merge_new)
                
            #     enm = 0
            #     for jj in range(len(txt_seg)):
            #         if txt_seg[jj] == "A":
            #             txt_seg[jj] = eng[enm]
            #             enm += 1

            # if len(txt_seg) > len(sub_txt_list):
            #     txt_seg = sub_txt_list
            
            # 按照分词结果，存储字、时间戳
            last_num = 0
            new_time_list = []
            new_txt_list = []
            for part_txt in txt_seg:
                txt_num = cal_txt_len(part_txt)
                if txt_num > 4:
                    # 分词后长度最长4，否则从第4个字分开
                    new_txt_list.append(part_txt[:4])
                    sub_time = sub_time_list[last_num:last_num+4]
                    new_time_list.append([sub_time[0][0], sub_time[-1][1]])

                    # all_time_list.append([sub_time[0][0], sub_time[-1][1]])
                    # all_txt_list.append(part_txt[:4])

                    new_txt_list.append(part_txt[4:])
                    sub_time = sub_time_list[last_num+4:last_num+txt_num]
                    new_time_list.append([sub_time[0][0], sub_time[-1][1]])

                    # all_time_list.append([sub_time[0][0], sub_time[-1][1]])
                    # all_txt_list.append(part_txt[4:])

                else:
                    new_txt_list.append(part_txt)
                    sub_time = sub_time_list[last_num:last_num+txt_num]
                    new_time_list.append([sub_time[0][0], sub_time[-1][1]])

                    # all_time_list.append([sub_time[0][0], sub_time[-1][1]])
                    # all_txt_list.append(part_txt)

                last_num = last_num + txt_num
            
            f_txt_list.append(new_txt_list)
            f_time_list.append(new_time_list)
        
        return f_time_list, f_txt_list, all_time_list, all_txt_list

    def sub_crop(self, f_time_list, f_txt_list):
        # 每个片段内部以1秒的窗口，结合字词时间戳，进行划分
        TT = []
        group_dex = []

        for jj in range(len(f_time_list)):
           
            # begin = int(f_time_list[jj][0][0] * 16000)
            # end = int(f_time_list[jj][-1][-1] * 16000)

            cur_seg_time = np.array(f_time_list[jj])

            split_points = self.get_feat(cur_seg_time, f_txt_list[jj])

            during = cur_seg_time[-1, 1] - cur_seg_time[0, 0]
            if len(f_txt_list[jj]) == 1 and cal_txt_len(f_txt_list[jj][0]) == 1 and during < 0.5:
                group_dex = group_dex + [-1] * len(split_points)
            
            else:
                group_dex = group_dex + [jj] * len(split_points)

            TT = TT + split_points

        return TT, group_dex

if __name__ == '__main__':
    import time
    aa = AudioSplit()
    time.sleep(10)
