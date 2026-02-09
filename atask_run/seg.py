from .mute_dec import crop_str_time
from .del_str import is_english_class
from .audio_split import AudioSplit
import numpy as np

AS = AudioSplit()     # 分割算法

def seg_asr(audio, txt_list, time_list):

    english_class = is_english_class(txt_list)
    # 每个字微调时间戳
    crop_time_list = crop_str_time(audio, time_list)
    # 以0.2s的间隔，分割片段
    # split_dex 按0.2秒间隔分割后，切割点在字里的序号，即第几个字后是片段分割点
    seg_time, seg_txt, __ = AS.crop_winds(crop_time_list, txt_list, english_class)

    # 对分割完的片段，每个片段内部分词
    # f_time_list 每个片段内每个字词的时间戳 f_txt_list 每个片段内每个字词的内容
    # all_time_list 所有字词的时间戳 all_txt_list 所有字词
    f_time_list, f_txt_list, __, __ = AS.seg_pk(seg_time, seg_txt)
    
    # 每个片段内部以1秒的窗口，结合字词时间戳，进行划分，提取特征       
    TT, group_dex = AS.sub_crop(f_time_list, f_txt_list)

    TT = np.array(TT)
    group_dex = np.array(group_dex)

    return TT, group_dex, crop_time_list

def alone_merge(TT, FF, group_dex, merge_thresh=0.28):
    # 处理单字，前后合并
    rm_dex_alone = []
    for i in range(1, len(FF) - 1):

        if group_dex[i] >= 0:
            continue
        
        t = TT[i]
        f = FF[i]

        if group_dex[i + 1] < 0 and group_dex[i - 1] < 0:
            # 前后都是单字的，不去判断了
            continue
        
        if group_dex[i + 1] >= 0 and group_dex[i - 1] >= 0:
            t0 = TT[i - 1]
            t1 = TT[i + 1]
            diff_l = t[0] - t0[1]
            diff_r = t1[0] - t[1]

            f0 = FF[i - 1]
            f1 = FF[i + 1]

        elif group_dex[i - 1] >= 0:
            t0 = TT[i - 1]
            diff_l = t[0] - t0[1]
            diff_r = 10 # 不跟后面的比，赋个大的间隔

            f0 = FF[i - 1]
            f1 = np.zeros(192, )

        else:
            t1 = TT[i + 1]
            diff_r = t1[0] - t[1]
            diff_l = 10 # 不跟前面的比，赋个大的间隔

            f1 = FF[i + 1]
            f0 = np.zeros(192, )

        s0 = np.dot(f, f0.T)  # 跟前面比向量
        s1 = np.dot(f, f1.T)  # 跟后面比向量
        ml, mr = 0, 0
        if diff_l <= 0.5 and diff_r <= 0.5:
            # 如果离前后都比较近
            # 如果跟前后比较的得分都超过阈值，那就跟最近的那个合并，否则谁超过了，跟谁合并
            if s0 >= merge_thresh and s1 >= merge_thresh:
                if diff_l <= diff_r:
                    ml = 1
                else:
                    mr = 1

            elif s0 >= s1 and s0 >= merge_thresh:
                ml = 1

            elif s1 > s0 and s1 >= merge_thresh:
                mr = 1
        
        elif diff_l <= 0.5 and s0 >= merge_thresh:
            ml = 1

        elif diff_r <= 0.5 and s1 >= merge_thresh:
            mr = 1
        
        if ml:
            # 前合并
            TT[i - 1, 1] = t[1]
            rm_dex_alone.append(i)

        elif mr:
            # 后合并
            TT[i + 1, 0] = t[0]
            rm_dex_alone.append(i)
    
    return rm_dex_alone

def find_non_overlapping_intervals(main_interval, sub_intervals):
    start, end = main_interval[0], main_interval[1]
    non_overlapping = []
    last_end = start

    for i in range(len(sub_intervals)):
        sub_start = sub_intervals[i, 0]
        sub_end = sub_intervals[i, 1]

        if sub_start > last_end:
            # 如果子区间与前一个不重叠，添加前一个不重叠的部分  
            non_overlapping.append((last_end, sub_start))
        last_end = max(last_end, sub_end)

    # 添加最后一个不重叠的部分（如果有）  
    if last_end < end:
        non_overlapping.append([last_end, end])

    # 过滤掉空区间（如果有）  
    non_overlapping = [[start, end] for start, end in non_overlapping if start < end]

    return non_overlapping

def cal_mean_audio(audio):
    if len(audio) > 100:
        audio_up = audio[audio>0]
        sort_d = np.sort(audio_up)
        keep = sort_d[-int(len(sort_d) * 0.1):]
        if len(keep) == 0:
            return 0.0001
        return float(round(np.mean(keep),5)) + 0.0001
    else:
        return 0.0001