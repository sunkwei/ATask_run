
import numpy as np
from collections import Counter 
import logging

logger = logging.getLogger('asr_runner')

def get_coincident_area(box0, box_arr):
    # 计算重叠面积
    if len(box0) * len(box_arr) == 0:
        return np.array([0])
        
    xx1 = np.maximum(box0[0], box_arr[:,0])
    xx2 = np.minimum(box0[1], box_arr[:,1])

    w = np.maximum(0.0, xx2 - xx1)

    return w

def cal_cluster_score(no_single_time, single_time, label, counter_info={}):

    logger.info("counter_info: " + str(counter_info))
    match_l = []
    short_part = []
    for i in range(len(no_single_time)):
        sub_no_single = np.array(no_single_time[i])
        during = sub_no_single[1] - sub_no_single[0]
        if during < 1:
            # 小于1秒的片段
            left_dex = np.where(single_time[:, 1] <= sub_no_single[0])[0]
            if len(left_dex) == 0:
                continue
            
            if abs(single_time[left_dex[-1], 1] - sub_no_single[0]) <= 5:
                short_part.append(label[left_dex[-1]])


            right_dex = np.where(single_time[:, 0] >= sub_no_single[1])[0]
            if len(right_dex) == 0:
                continue
            
            if abs(single_time[right_dex[0], 0] - sub_no_single[1]) <= 5:
                short_part.append(label[right_dex[0]])

            continue

        if 0:#during >= 15:
            left_time_range = np.array(([sub_no_single[0] - 10, sub_no_single[0]]))
            area_list = get_coincident_area(left_time_range, single_time)
            dex = np.where(area_list > 0)[0]
            if np.sum(area_list) >= 5:
                match_l.append(label[dex[-1]])

            right_time_range = np.array(([sub_no_single[1], sub_no_single[1] + 10]))
            area_list = get_coincident_area(right_time_range, single_time)
            dex = np.where(area_list > 0)[0]
            if np.sum(area_list) >= 5:
                match_l.append(label[dex[-1]])

        else: 
            left_dex = np.where(single_time[:, 1] <= sub_no_single[0])[0]
            if len(left_dex) == 0:
                continue
            
            if 1:#abs(single_time[left_dex[-1], 1] - sub_no_single[0]) <= 5:
                match_l.append(label[left_dex[-1]])


            right_dex = np.where(single_time[:, 0] >= sub_no_single[1])[0]
            if len(right_dex) == 0:
                continue
            
            if 1:#abs(single_time[right_dex[0], 0] - sub_no_single[1]) <= 5:
                match_l.append(label[right_dex[0]])
            # match_l.append(label[right_dex[0]])

    if len(match_l) == 0:
        counter = Counter(label)
        most_common_element = counter.most_common(1)[0] 
        element, __ = most_common_element 
        return element
    
    else:
        counter0 = Counter(match_l)
        counter = counter0.copy()
        logger.info("orgin counter score: " + str(counter))
        print ("cluster counter: ", counter)
        most_common_element0 = counter.most_common(1)[0] 
        element0, __ = most_common_element0
        if len(short_part) != 0:
            sp = Counter(short_part)

            for kp in sp:
                if kp in counter:
                    counter[kp] += sp[kp] * 0.3
        
            logger.info("short counter score: " + str(sp))
            
        logger.info("merge counter score: " + str(counter))
        print ("merge counter: ", counter)

        element1 = counter.most_common(1)[0][0]
        if len(counter) == 1:
            return element1
        
        top2 = counter.most_common(2)
        max_element = max(top2[0][1], top2[1][1])
        min_element = min(top2[0][1], top2[1][1])
        if min_element / max_element > 0.75 or (max_element - min_element < 10 and max_element > 50):
            # top2打分很接近
            top2_0 = counter0.most_common(2)
            if min(top2_0[0][1], top2_0[1][1]) / max(top2_0[0][1], top2_0[1][1]) > 0.75 and \
                (max(top2_0[0][1], top2_0[1][1]) - min(top2_0[0][1], top2_0[1][1])) > 10:

                return element0
            else:
                if counter_info[top2[0][0]] >= counter_info[top2[1][0]]:
                    return top2[0][0]
                else:
                    return top2[1][0]

        else:
            return element1

        # if near_bool:
        #     # 如果两个打分比较接近
        #     if counter_info[element0] >= counter_info[element1]:
        #         return element0
        #     else:
        #         return element1

        # return element1
        
        # if len(counter) > 1:
        #     top2 = counter.most_common(2)
        #     if min(top2[0][1], top2[1][1]) / max(top2[0][1], top2[1][1]) > 0.8 and (max(top2[0][1], top2[1][1]) - min(top2[0][1], top2[1][1])) < 10:
        #         # 如果两个打分比较接近，则取条数多的
        #         if counter_info[top2[0][0]] >= counter_info[top2[1][0]]:
        #             element1 = top2[0][0]
        #         else:
        #             element1 = top2[1][0]
        #     else:
        #         element1 = top2[0][0]
        # else:
        #     element1 = counter.most_common(1)[0][0]

        # if element0 != element1 and len(counter_info) > 0:
        #     if counter_info[element0] >= counter_info[element1]:
        #         return element0
        #     else:
        #         return element1
        # else:
        #     return element1
