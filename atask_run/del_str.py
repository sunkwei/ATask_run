import re,jieba
import numpy as np

def crop_winds(time_list, txt_list, crop_thresh=0.2):
    #
    # 以0.2s的间隔，分割片段
    last_time = time_list[0][0]
    sub_seg_time = []
    num = -1
    split_dex = []
    for ii in range(len(time_list)):
        num += 1
        next_time = time_list[ii][0]
        txt = txt_list[ii]
        if txt in ["哇", "呀", "呢", "嗯", "啊", "吧", "啦", "哦", "嘿", "哟", "嘛", "哈", "呵"
                    "喂", "了", "呗", "诶", "哎", "噢", "幺", "呃", "哼", "呦", "吗"]:
            continue

        if next_time - last_time >= crop_thresh:  
            split_dex.append(num - 1)
        
        if ii == len(time_list) - 1 and len(sub_seg_time) > 0:
            split_dex.append(num)
            break

        last_time = time_list[ii][1]
        
    return split_dex

def count_elements_in_list(a, b):
    """
    统计列表 a 中出现在列表 b 中的元素的个数。

    :param a: 大列表，包含汉字或单词
    :param b: 需要统计的元素列表
    :return: 统计结果，字典形式，键为 b 中的元素，值为在 a 中出现的次数
    """
    # 使用字典推导式初始化统计结果
    count_dict = {element: 0 for element in b}

    # 遍历列表 a，统计 b 中元素的出现次数
    for item in a:
        if item in count_dict:
            count_dict[item] += 1

    return count_dict

def get_coincident_area(box0, box_arr):
    # 计算重叠面积
    if len(box0) * len(box_arr) == 0:
        return np.array([0])
        
    xx1 = np.maximum(box0[0], box_arr[:,0])
    xx2 = np.minimum(box0[1], box_arr[:,1])

    w = np.maximum(0.0, xx2 - xx1)

    return w

def isEnglish(text:str):
    if re.search('^[a-zA-Z\']+$', text):
        return True
    else:
        return False

def opt_punc(txt_list, punc_dex, time_list):
    # 优化标点,判断标点前一个字是否应该归到下一句
    for d in range(1, len(punc_dex) - 1):
        last_dex = punc_dex[d - 1]
        curr_dex = punc_dex[d]

        if last_dex == 1 and curr_dex > 1 and (not isEnglish(txt_list[d-1])) and (not isEnglish(txt_list[d])) and (not isEnglish(txt_list[d+1])):
            txt = "".join(txt_list[d-1:d+2])
            seg_list = jieba.cut(txt, cut_all=False)
            seg_list = list(seg_list)

            if (txt_list[d] + txt_list[d+1]) in seg_list and ((time_list[d][0] - time_list[d-1][1]) - (time_list[d+1][0] - time_list[d][1]) >= 0.1):
                punc_dex[d-1] = punc_dex[d]
                punc_dex[d] = 1
    return punc_dex

def isNumeric(text:str):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, text.strip()))

def cal_txt_len(txt):
    # 计算一个字符串的长度
    if "'" in txt or re.search('^[a-zA-Z\']+$', txt):
        return 1
    else:
        return len(txt)
def isChinese(ch):
    if '\u4e00' <= ch <= '\u9fff' or '\u0030' <= ch <= '\u0039':
        return True
    return False

def cal_char_num(txt):

    split_txt = txt.split(" ")
    chr_num = 0
    for t in split_txt:
        sub_ch_num = 1
        han = 0
        for sub_t in t:
            if isChinese(sub_t):
                han = 1
                sub_ch_num += 1
        
        chr_num = chr_num + sub_ch_num - han

    return chr_num

def isAllAlpha(word):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False
    for ch in word_lists:
        if len(ch) > 0 and isChinese(ch) and ch.isdigit() is False:
            return False

    return True

def is_punctuation(char):
    # 英文标点
    english_punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    # 中文标点  
    chinese_punctuation = r'，。！？、；：“”‘’（）《》【】「」『』｛｝＜＞「」『』' 

    # 使用正则表达式匹配标点
    if re.match(f'[{re.escape(english_punctuation)}]', char):
        return 0
    
    elif re.match(f'[{re.escape(chinese_punctuation)}]', char):
        return 1
    else:
        return -1
    
punctuation_map = {  
        '。': '.',  
        '，': ',',  
        '：': ':',  
        '；': ';',  
        '？': '?',  
        '！': '!',  
        '（': '(',  
        '）': ')',  
        '《': '<',  
        '》': '>',  
        '【': '[',  
        '】': ']',  
        '「': '"',  
        '」': '"',  
        '『': "'",  
        '』': "'",  
        '｛': '{',  
        '｝': '}',  
        '‘': "'",  
        '’': "'",  
        '“': '"',  
        '”': '"',  
        '——': '--',  
        '～': '~',  
        '……': '...',  
        '－': '-',  
        '／': '/',  
        '｜': '|',  
        '\\': '\\',  
    }  

# from flag import get_char_pos
import os
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(WORK_DIR + '/duplicate_word_whitelist.txt'):
    with open(WORK_DIR + '/duplicate_word_whitelist.txt', 'r', encoding='utf8') as f:
        v_str = [line.strip() for line in f.readlines()]
else:   
    v_str = []

def check_duplicate_words(txt):
    # 判断一个字有无可能组成叠字
    return txt in v_str

def find_same_words(txt_list, txt):
    same = False
    for w in txt_list:
        if txt + txt in w:
            same = True
            break
    
    return same

def filt_duplicate_words(txt_list, time_list_all):

    merge_txt = ''.join(txt_list)
    seg_list = jieba.cut(merge_txt, cut_all=False)
    seg_list = list(seg_list)

    txt_list_new = []
    time_list_all_new = []
    long_interval_dex = []

    last_str = ""
    same_zh = 0
    thresh = 2
    for idx, (txt, time) in enumerate(zip(txt_list, time_list_all)):
        
        if isEnglish(txt):
            if txt == 'i':
                txt = "I"
            if "i'" in txt and "i'" == txt[:2]:
                txt = "I'" + txt[2:]

            if len(txt_list_new) == 0:
                txt_list_new.append(txt)
                time_list_all_new.append(time)

            else:
                if txt != last_str or (time[0] - time_list_all_new[-1][1] >= 1000):
                    # 不同的英文或者间隔时间大于1000ms的不去重
                    txt_list_new.append(txt)
                    time_list_all_new.append(time)
                
                else:
                    time_list_all_new[-1][1] = time[1]

            last_str = txt

        else:
            if last_str not in v_str:
                thresh = 1
            else:
                thresh = 2

            if txt == last_str:
                same_zh += 1
            else:
                same_zh = 0

            if same_zh < thresh:# or (same_zh > 0 and time[0] - time_list_all_new[-1][1] >= 1000):
                if same_zh > 0 and time[0] - time_list_all_new[-1][1] < 1000:
                    if find_same_words(seg_list, txt):
                        txt_list_new.append(txt)
                        time_list_all_new.append(time)
                    else:
                        time_list_all_new[-1][1] = time[1]
                else:
                    txt_list_new.append(txt)
                    time_list_all_new.append(time)
            else:
                time_list_all_new[-1][1] = time[1]

            last_str = txt

    if len(time_list_all_new) == 0:
        return txt_list_new, time_list_all_new#, long_interval_dex, []

    # en_dex = [1] if isEnglish(txt_list_new[0]) else [0]
    
    # for i in range(1, len(time_list_all_new)):
    #     if isEnglish(txt_list_new[i]):
    #         en_dex.append(1)
    #     else:
    #         en_dex.append(0)

    #     if time_list_all_new[i][0] - time_list_all_new[i - 1][1] >= 1000:
    #         long_interval_dex.append(i-1)

    return txt_list_new, time_list_all_new#, long_interval_dex, en_dex
    

def join_chinese_and_english(input_list):
    # 内容拼接
    line = ''
    last_english = False
    last_str = ""
    num = 0
    same_zh = 0
    thresh = 2

    for token in input_list:
        punctuation = is_punctuation(token)
        if last_english and punctuation == 1:
            if token in punctuation_map:
                token = punctuation_map[token]
                punctuation = 0
        
        if isEnglish(token) or isNumeric(token):
            num += 1
            if token == 'i':
                token = "I"
            if "i'" in token and "i'" == token[:2]:
                token = "I'" + token[2:]

            if len(line) == 0:
                line = token
            else:
                if 1:#token != last_str:
                    line = line + ' ' + token


            last_english = True
            last_str = token
        else:
            if last_english and punctuation < 0:
                line = line + ' ' + token
                last_str = token

                # if last_str not in v_str:
                #     thresh = 1
                # else:
                #     thresh = 2
            else:

                # if last_str not in v_str:
                #     thresh = 1
                # else:
                #     thresh = 2

                # if token == last_str:
                #     same_zh += 1
                # else:
                #     same_zh = 0

                if 1:#same_zh < thresh:
                    line = line + token

                last_str = token

            last_english = False 

    line = line.strip()
    return line, num

def is_english_class(txt_list, en_ratio_th=0.1):
    # 判断是否是英语课，单词占比超过阈值
    if len(txt_list) == 0:
        return False
    
    num = 0
    for t in txt_list:
        if isEnglish(t):
            num += 1
            if num >= len(txt_list) * en_ratio_th:
                return True
    
    return False

if __name__ == '__main__':
    print (filt_duplicate_words(["hello", "ni", "hello", "hello", "你", "你", "练", "练"],[[1,2],[2,3], [3,4],[6,8],[5,6],[6,7],[7,8],[8,9]]))
    # print(join_chinese_and_english(['i', "'m", 'good', 'good', 'good',"来", "你", '你', '一','一','一',',',"hello", "高", "高", "的", "的"]))
