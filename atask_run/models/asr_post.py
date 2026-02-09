from typing import NamedTuple, Union, Dict, Any, Iterable, List, Tuple
import numpy as np

## 来自 funasr_onnx/utils/utils.py
class Hypothesis(NamedTuple):
    """Hypothesis data type."""
    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()

## 来自 funasr_onnx/utils/utils.py
class TokenIDConverterError(Exception):
    pass

## 来自 funasr_onnx/utils/utils.py
class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[List, str],
    ):
        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise TokenIDConverterError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]

## 来自 funasr_onnx/utils/timestamp_utils.py
# def time_stamp_lfr6_onnx(us_cif_peak:np.ndarray, char_list, begin_time=0.0, total_offset=-1.5):
#     if not len(char_list):
#         return "", []
#     START_END_THRESHOLD = 5
#     MAX_TOKEN_DURATION = 12
#     TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
#     cif_peak = us_cif_peak.reshape(-1)
#     num_frames = cif_peak.shape[-1]
#     if char_list[-1] == "</s>":
#         char_list = char_list[:-1]
#     # char_list = [i for i in text]
#     timestamp_list = []
#     new_char_list = []
#     # for bicif model trained with large data, cif2 actually fires when a character starts
#     # so treat the frames between two peaks as the duration of the former token
#     fire_place = np.where(cif_peak > 1.0 - 1e-4)[0] + total_offset  # np format
#     num_peak = len(fire_place)
#     assert num_peak == len(char_list) + 1  # number of peaks is supposed to be number of tokens + 1
#     # begin silence
#     if fire_place[0] > START_END_THRESHOLD:
#         # char_list.insert(0, '<sil>')
#         timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
#         new_char_list.append("<sil>")
#     # tokens timestamp
#     for i in range(len(fire_place) - 1):
#         new_char_list.append(char_list[i])
#         if (
#             i == len(fire_place) - 2
#             or MAX_TOKEN_DURATION < 0
#             or fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION
#         ):
#             timestamp_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
#         else:
#             # cut the duration to token and sil of the 0-weight frames last long
#             _split = fire_place[i] + MAX_TOKEN_DURATION
#             timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
#             timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
#             new_char_list.append("<sil>")
#     # tail token and end silence
#     if num_frames - fire_place[-1] > START_END_THRESHOLD:
#         _end = (num_frames + fire_place[-1]) / 2
#         timestamp_list[-1][1] = _end * TIME_RATE
#         timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
#         new_char_list.append("<sil>")
#     else:
#         timestamp_list[-1][1] = num_frames * TIME_RATE
#     if begin_time:  # add offset time in model with vad
#         for i in range(len(timestamp_list)):
#             timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
#             timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
#     assert len(new_char_list) == len(timestamp_list)
#     res_str = ""
#     for char, timestamp in zip(new_char_list, timestamp_list):
#         res_str += "{} {} {};".format(char, timestamp[0], timestamp[1])
#     res = []
#     for char, timestamp in zip(new_char_list, timestamp_list):
#         if char != "<sil>":
#             res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
#     return res_str, res
def time_stamp_lfr6_onnx(us_cif_peak, char_list, onnx=1, begin_time=0.0, total_offset=-1.5):
    """生成时间戳的函数，基于LFR6 ONNX模型"""
    if not len(char_list):
        return "", []
    # 静音阈值和最大token持续时间
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 30
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    if onnx:
        cif_peak = us_cif_peak.reshape(-1)
    else:
        cif_peak = us_cif_peak.reshape(-1).cpu()
    num_frames = cif_peak.shape[-1]
    if char_list[-1] == "</s>":
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    new_char_list = []
    # 对于大数据训练的bicif模型，cif2在字符开始时触发
    # 因此将两个峰值之间的帧视为前一个token的持续时间
    fire_place = np.where(cif_peak > 1.0 - 1e-4)[0] + total_offset  # np format
    num_peak = len(fire_place)
    # assert num_peak == len(char_list) + 1  # number of peaks is supposed to be number of tokens + 1
    if num_peak < len(char_list) + 1:
        dif = len(char_list) + 1 - num_peak
        pad = np.array([fire_place[-1]] * dif)
        fire_place = np.hstack((fire_place, pad))
    # 开头静音处理
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
        new_char_list.append("<sil>")
    # token时间戳生成
    for i in range(len(fire_place) - 1):
        if i >= len(char_list):
            continue
        new_char_list.append(char_list[i])
        if (
            i == len(fire_place) - 2
            or MAX_TOKEN_DURATION < 0
            or fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION
        ):
            timestamp_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
        else:
            # 将长持续时间分割为token和静音
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            new_char_list.append("<sil>")
    # 结尾token和静音处理
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) / 2
        timestamp_list[-1][1] = _end * TIME_RATE
        timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames * TIME_RATE
    if begin_time:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
    assert len(new_char_list) == len(timestamp_list)
    res_str = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_str += "{} {} {};".format(char, timestamp[0], timestamp[1])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_str, res

import re
## 来自 funasr_onnx/utils/postprocess_utils.py
def isChinese(ch: str):
    if "\u4e00" <= ch <= "\u9fff" or "\u0030" <= ch <= "\u0039":
        return True
    return False

def isEnglish(text:str):
    if re.search('^[a-zA-Z\']+$', text):
        return True
    else:
        return False
    
## 来自 funasr_onnx/utils/postprocess_utils.py
def isAllChinese(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(" ", "")
        cur = cur.replace("</s>", "")
        cur = cur.replace("<s>", "")
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if isChinese(ch) is False:
            return False
    return True

## 来自 funasr_onnx/utils/postprocess_utils.py
def isAllAlpha(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(" ", "")
        cur = cur.replace("</s>", "")
        cur = cur.replace("<s>", "")
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch != "'":
            return False
        elif ch.isalpha() is True and isChinese(ch) is True:
            return False

    return True

def isAllAlpha1(word):
        word_lists = []
        for i in word:
            cur = i.replace(' ', '')
            cur = cur.replace('</s>', '')
            cur = cur.replace('<s>', '')
            word_lists.append(cur)

        if len(word_lists) == 0:
            return 0
        
        chinese = 0
        en = 0
        for ch in word_lists:
            if len(ch) > 0 and ('\u4e00' <= ch <= '\u9fff' or '\u0030' <= ch <= '\u0039') \
                and ch.isdigit() is False:
                chinese = 1        
            else:
                en = 1

        if chinese == 1 and en == 1:
            # 中英混合
            return 0
        elif chinese == 1 and en == 0:
            # 纯中文
            return 1
        else:
            # 纯英文
            return 2
def merge_en_str(txt_list):

    txt_list_new = []
    m = ""
    for i in range(len(txt_list)):
        t = txt_list[i]
        if not isEnglish(t) or len(t) >1:
            if len(m) > 0:
                txt_list_new.append(m)
                m = ""
            txt_list_new.append(t)

        else:
            m = m + t

        if i == len(txt_list) - 1 and len(m) > 1:
            txt_list_new.append(m)

    return txt_list_new

def check_stamp_txt(txt, time_list, language):
    if len(txt) > len(time_list):
            txt = merge_en_str(txt)

    if len(time_list) != len(txt):
        # 强制保证字数跟时间戳个数相同，否则后面统计出错
        if len(txt) > len(time_list):
            dif_num = len(txt) - len(time_list)
            for ii in range(dif_num):
                time_list.append(time_list[-1])
        else:
            dif_num = len(time_list) - len(txt)
            for ii in range(dif_num):
                txt.append(txt[-1])

    assert len(time_list) == len(txt)
    txt_new = []
    time_list_new = []
    if language == "en":
        # 英文句子
        max_time = 1000
    else:
        max_time = 500
    for ti in range(len(time_list)):
        # 一个字最长不能持续0.5秒的时间
        if time_list[ti][1] - time_list[ti][0] <= max_time:
            txt_new.append(txt[ti])
            time_list_new.append(time_list[ti])
        else:
            txt_new.append(txt[ti])
            time_list[ti][1] = time_list[ti][0] + max_time
            time_list_new.append(time_list[ti])
    
    return txt_new, time_list_new
    
## 参考 funasr_onnx/utils/postprocess_utils.py
def abbr_dispose(
    words: List[Any], 
    time_stamp: List[List]|None=None,
) -> List[Any]|Tuple[List[Any], List[Any]]:
    words_size = len(words)
    word_lists = []
    abbr_begin = []
    abbr_end = []
    last_num = -1
    ts_lists = []
    ts_nums = []
    ts_index = 0
    for num in range(words_size):
        if num <= last_num:
            continue

        if len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
            if (
                num + 1 < words_size
                and words[num + 1] == " "
                and num + 2 < words_size
                and len(words[num + 2]) == 1
                and words[num + 2].encode("utf-8").isalpha()
            ):
                # found the begin of abbr
                abbr_begin.append(num)
                num += 2
                abbr_end.append(num)
                # to find the end of abbr
                while True:
                    num += 1
                    if num < words_size and words[num] == " ":
                        num += 1
                        if (
                            num < words_size
                            and len(words[num]) == 1
                            and words[num].encode("utf-8").isalpha()
                        ):
                            abbr_end.pop()
                            abbr_end.append(num)
                            last_num = num
                        else:
                            break
                    else:
                        break

    for num in range(words_size):
        if words[num] == " ":
            ts_nums.append(ts_index)
        else:
            ts_nums.append(ts_index)
            ts_index += 1
    last_num = -1
    begin = -1
    for num in range(words_size):
        if num <= last_num:
            continue

        if num in abbr_begin:
            if time_stamp is not None:
                begin = time_stamp[ts_nums[num]][0]
            word_lists.append(words[num].upper())
            num += 1
            while num < words_size:
                if num in abbr_end:
                    word_lists.append(words[num].upper())
                    last_num = num
                    break
                else:
                    if words[num].encode("utf-8").isalpha():
                        word_lists.append(words[num].upper())
                num += 1
            if time_stamp is not None:
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
        else:
            word_lists.append(words[num])
            if time_stamp is not None and words[num] != " ":
                begin = time_stamp[ts_nums[num]][0]
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
                begin = end

    if time_stamp is not None:
        return word_lists, ts_lists
    else:
        return word_lists

## 来自 funasr_onnx/utils/postprocess_utils.py
def sentence_postprocess(words: List[Any], time_stamp: List[List] = None):
    middle_lists = []
    word_lists = []
    word_item = ''
    ts_lists = []
    # wash words lists
    for i in words:
        word = ''
        if isinstance(i, str):
            word = i
        else:
            word = i.decode('utf-8')

        if word in ['<s>', '</s>', '<unk>']:
            continue
        else:
            middle_lists.append(word)

    is_all_en = 0
    # all chinese characters
    if isAllChinese(middle_lists):
        for i, ch in enumerate(middle_lists):
            word_lists.append(ch.replace(' ', ''))
        if time_stamp is not None:
            ts_lists = time_stamp

    # all alpha characters
    elif isAllAlpha(middle_lists):
        is_all_en = 1
        ts_flag = True
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ''
            if '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            elif '@' in ch:
                word = ch.replace('@', '')
                word_item += word
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1]
                    ts_lists.append([begin, end])
                    begin = end

    # mix characters
    else:
        alpha_blank = False
        ts_flag = True
        begin = -1
        end = -1

        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ''
            if isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = True
                    ts_lists.append([begin, end])
                    begin = end
            elif '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            elif '@' in ch:
                word = ch.replace('@', '')
                word_item += word
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]

            elif isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
                alpha_blank = True
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1] 
                    ts_lists.append([begin, end])
                    begin = end
            elif len(ch) == 0:
                continue
            else:
                continue
                #raise ValueError('invalid character: {}'.format(ch))

    if time_stamp is not None: 
        word_lists, ts_lists = abbr_dispose(word_lists, ts_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != ' ':
                real_word_lists.append(ch)
        sentence = ' '.join(real_word_lists).strip()
        return sentence, ts_lists, real_word_lists
    else:
        word_lists = abbr_dispose(word_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != ' ':
                real_word_lists.append(ch)
        sentence = ''.join(word_lists).strip()
        return sentence, real_word_lists#, is_all_en

