
from collections import Counter 
from .cluster_backend import ClusterBackend
from .find_teacher import cal_cluster_score
import numpy as np

def sample_cluster(FF, single_time, no_single_time):
    # 聚类
    l = ClusterBackend()
    label_spe, label_info, kn = l(FF)
    
    counter_spe = dict(Counter(label_spe))
    print ("spe", counter_spe)
    counter_info = Counter(label_info)
    print ("info", counter_info)
    # 确定老师类
    info_teacher = cal_cluster_score(no_single_time, single_time, label_info, dict(counter_info))
    print ("teacher label", info_teacher)
    info_dex = np.where(label_info == info_teacher)[0]
    # 两个聚类算法中老师类的片段交集，作为老师的确定片段
    same_num = []
    label_k = []
    for k in counter_spe:
        spe_dex0 = np.where(label_spe == k)[0]
        same_dex = list(set(info_dex) & set(spe_dex0))

        same_num.append(len(same_dex))
        label_k.append(same_dex)

    must_t_dex = label_k[np.argmax(same_num)]
    spe_teacher_f = FF[must_t_dex]

    may_dex = []
    if len(spe_teacher_f) > 0 and kn < 200:
        # 如果找到确定的老师
        # 循环召回聚到非老师类里的老师片段
        for ii in range(len(FF)):

            if ii in must_t_dex:
                continue

            fd = FF[ii]
            sss = np.dot(fd, spe_teacher_f.T)
            ratio = len(np.where(sss >= 0.4)[0]) / len(spe_teacher_f)
            if ratio > 0.2:
                
                sd = np.dot(fd, FF.T)
                top_dex = np.argsort(sd)[::-1][1:20]
                match_num = len(set(must_t_dex) & set(top_dex))
                # print (single_time[ii], ratio, match_num)
                if (ratio > 0.4 and match_num >= 5) or (ratio > 0.3 and match_num >= 8):
                    may_dex.append(ii)
        
    LL = np.ones(len(FF)).astype(int)
    LL[must_t_dex] = 0
    LL[may_dex] = 0

    counter = Counter(LL)
    print ("finally", counter)   

    return LL     

