''' from FunASR
'''

import numpy as np
from onnxruntime import InferenceSession as OrtInferSession
from onnxruntime import SessionOptions
import os.path as osp
from pathlib import Path
import yaml
from .e2e_vad import E2EVadModel
import math
import logging
import time, sys
from typing import cast, Tuple

def read_yaml(yaml_path):
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

class Fsmn_vad():
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    _cmvn = None
    _mel_banks = None       ## F32 (257, 80)
    _mel_center_freqs = None     ## F32 (80,)
    _epsilon = 1.1921e-7

    def __init__(
        self, 
        cmvn_file:str,
        config_file:str,
        model_file:str,
        step_seconds:int=8,
        using_off_stamp:bool=False, 
        debug:bool=False
    ):
        self.debug = debug
        config = read_yaml(config_file)

        ## FIXME: 强制修改配置参数 speech_noise_thres 
        config["speech_noise_thres"] = -4.0

        if Fsmn_vad._cmvn is None:
            Fsmn_vad._cmvn = Fsmn_vad.__load_cmvn(cmvn_file)
        if Fsmn_vad._mel_banks is None:
            Fsmn_vad._mel_banks, _ = Fsmn_vad.__mel_banks(80)

        if self.debug:
            print(f"fsmn_vad: config, step_seconds: {step_seconds}, using_off_stamp: {using_off_stamp}")
            print(f"  max_end_silence_time: {config['vad_post_conf']['max_end_silence_time']}")
            print(f"  max_single_segment_time: {config['vad_post_conf']['max_single_segment_time']}")
            print(f"  frame_in_ms: {config['vad_post_conf']['frame_in_ms']}")
            print(f"  frame_length_ms: {config['vad_post_conf']['frame_length_ms']}")
            print(f"  max_end_silence_time: {config['vad_post_conf']['max_end_silence_time']}")
            print(f"  encoder_conf: {config['encoder_conf']}")
            print(f"  speech_noise_thres: {config['vad_post_conf']['speech_noise_thres']}")
            print(f"  speech_noise_thresh_low: {config['vad_post_conf']['speech_noise_thresh_low']}")
            print(f"  speech_noise_thresh_high: {config['vad_post_conf']['speech_noise_thresh_high']}")

        session_opt = SessionOptions()
        session_opt.intra_op_num_threads = 1
        self.ort_infer = OrtInferSession(model_file, session_opt)
        self.vad_scorer = E2EVadModel(config["vad_post_conf"])
        self.max_end_sil = config["vad_post_conf"]["max_end_silence_time"]
        self.encoder_conf = config["encoder_conf"]
        self.__begin_time_ms:int = 0
        self.__cached_waveform = np.empty([0], np.float32)
        self.__step_second = step_seconds     # 每段时长
        self.__in_cache = self.__prepare_cache([])

    def reset(self):
        self.__cached_waveform = np.empty([0], np.float32)
        self.vad_scorer.AllResetDetection()
        self.__begin_time_ms = 0
        self.__in_cache = self.__prepare_cache([])
    
    def update(self, waveform:np.ndarray=np.empty((0,), np.float32), begin_time_ms:int=-1, last:bool=False) -> list:
        '''
            每次输入一段 pcm，累加到 cache 后，若长度超过 16000 * self.__step_second 进行一次推理
            如果启用 onnx，每片预测，
            如果 tea，每攒到 _step_second，执行一次

        '''
        assert waveform.ndim == 1 and waveform.dtype == np.float32
        return self.__run_onnx(waveform, begin_time_ms, last)
        
    def __preprocess(self, pcm:np.ndarray) -> np.ndarray:
        ## 执行预处理
        assert pcm.ndim == 1 and pcm.dtype == np.float32
        pcm = pcm * (1 << 15)
        inp, _ = self.__get_window(pcm)     ## spectrum
        spectrum = np.abs(np.fft.rfft(inp)).astype(np.float32)
        spectrum = np.power(spectrum, 2.0)      ## 能量谱   (n, 257)
        mel_energy = spectrum @ Fsmn_vad._mel_banks    ## (n, 257) @ (257, 80) = (n, 80)
        mel_energy = np.log(np.maximum(mel_energy, self._epsilon))    ## mel log spect
        inp = self.__lfr(mel_energy, m=5, n=1)
        inp = self.__cmvn(inp)
        return inp

    @staticmethod
    def __load_cmvn(cmvn_file) -> np.ndarray:
        with open(cmvn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == '<AddShift>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    add_shift_line = line_item[3:(len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == '<Rescale>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    rescale_line = line_item[3:(len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float32)
        vars = np.array(vars_list).astype(np.float32)
        cmvn = np.array([means, vars])
        return cmvn
    
    @staticmethod
    def __mel_banks(
        num_bins, 
        window_length_padded=512, 
        sample_freq=16000, 
        low_freq=20, 
        high_freq=8000, 
        vtln_low=50, 
        vtln_high=-1, 
        vtln_warp_factor:float=1.0
    ):
        '''
        计算 Mel 滤波器
        :param num_bins: 频率分辨率
        :param window_length_padded: 窗口长度
        :param sample_freq: 采样频率
        :param low_freq: 低频
        :param high_freq: 高频
        :param vtln_low: 变音低频
        :param vtln_high: 变音高频
        :param vtln_warp_factor: 变音频率
        :return:
        '''
        assert num_bins > 3, "Must have at least 3 mel bins"
        assert window_length_padded % 2 == 0, "Window length must be even."
        assert vtln_warp_factor == 1.0, "Only support vtln_warp_factor = 1.0"

        num_fft_bins = window_length_padded / 2
        nyquist = 0.5 * sample_freq

        if high_freq <= 0.0:
            high_freq += nyquist

        assert 0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist and low_freq < high_freq

        fft_bin_width = sample_freq / window_length_padded
        mel_low_freq = 1127.0 * math.log(1.0 + low_freq / 700.0)
        mel_high_freq = 1127.0 * math.log(1.0 + high_freq / 700.0)

        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

        if vtln_high < 0.0:
            vtln_high += nyquist

        assert vtln_warp_factor == 1.0 or (low_freq < vtln_low < high_freq and 0.0 < vtln_high < high_freq and vtln_low < vtln_high)

        bin = np.arange(num_bins)[..., None]    # (num_bins, 1)
        left_mel = mel_low_freq + bin * mel_freq_delta
        center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta
        right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta

        center_freqs = 700.0 * (np.exp(center_mel / 1127.0) - 1.0)

        freqs = (fft_bin_width * np.arange(num_fft_bins))[None, ...]
        mel = 1127.0 * np.log(1.0 + freqs / 700.0)

        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)

        bins = np.maximum(0, np.minimum(up_slope, down_slope))

        bins = np.pad(bins, ((0,0), (0,1)), mode="constant", constant_values=0.0)       ## (80, 257)
        return bins.T.astype(np.float32), center_freqs

    def __lfr(self, inp:np.ndarray, m:int=7, n:int=6) -> np.ndarray:
        T = inp.shape[0]                              # T: 998
        T_lfr = int(np.ceil(T / n))                   # T_lfr: 167
        left_padding = np.repeat(inp[:1, :], (m - 1) // 2, axis=0)  # (3, 80)
        inps = np.vstack((left_padding, inp))       # (1001, 80)
        T = T + (m - 1) // 2
        feat_dim = inp.shape[1]     # 80
        strides = (n * feat_dim * inp.itemsize, 1 * inp.itemsize) # (480 * 4, 1 * 4)  ## XXX: 注意 np 的 stride 为字节长度
        shapes = (T_lfr, m * feat_dim)                             # (167, 560)     ## np.shape 使用元素数目，而不是字节长度
        last_idx = (T - m) // n + 1
        num_padding = m - (T - last_idx * n)
        if num_padding > 0:
            num_padding = (2 * m - 2 * T + (T_lfr - 1 + last_idx) * n) / 2 * (T_lfr - last_idx)
            inps = np.vstack([inps] + [inps[-1:]] * int(num_padding))
        return np.lib.stride_tricks.as_strided(inps, shape=shapes, strides=strides).copy()  ## 保持连续
    
    def __cmvn(self, inp:np.ndarray) -> np.ndarray:
        assert self._cmvn is not None and self._cmvn.shape == (2, 400)

        means = self._cmvn[0:1, :]  # (1, 400)
        vars = self._cmvn[1:2, :]   # (1, 400)
        return (inp + means) * vars     # (m, 400)
    
    def __get_window(
        self, 
        pcm, 
        padded_win_size:int=512, 
        win_size:int=400, 
        win_shift:int=160,
        win_func=lambda t: np.hamming(t).astype('float32'), 
        snip_edges:bool=True,
        raw_energy:bool=True, 
        energy_floor:float=0.0, 
        dither:float=0.0,
        remove_dc_offset:bool=True, 
        preemphasis:float=0.97
    ):
        stride_input = self.__get_strided(pcm, win_size, win_shift)

        if dither != 0.0:
            rand_gauss = np.random.rand(*stride_input.shape).astype(stride_input.dtype)
            stride_input += rand_gauss * dither

        if remove_dc_offset:
            row_means = np.mean(stride_input, axis=1, keepdims=True)    # (m, 1)
            # print(f"    mean: {row_means.reshape((-1,))[:16]}, {row_means.shape}")
            stride_input = stride_input - row_means  ## XXX: 使用 stride_input -= row_means 居然不是期望结果！！！
            # stride_input -= row_means
            # print(f"    after remove dc: {stride_input[:2, 256:256+16]}")

        if raw_energy:
            signal_log_energy = self.__get_log_energy(stride_input, self._epsilon, energy_floor)

        if preemphasis != 0.0:
            offset_strided_input = np.pad(stride_input, ((0, 0), (1, 0)), mode="edge")
            # stride_input -= preemphasis * offset_strided_input[:, :-1]
            stride_input = stride_input - preemphasis * offset_strided_input[:, :-1]
            # print(f"    after preemphasis: {stride_input[:2, 256:256+16]}")

        stride_input *= win_func(win_size).reshape((1, win_size))
        # print(f"    after window: {stride_input[:2, 256:256+16]}")

        if padded_win_size > win_size:
            padding_right = padded_win_size - win_size
            stride_input = np.pad(stride_input, ((0, 0), (0, padding_right)), mode="constant", constant_values=0)

        if not raw_energy:
            signal_log_energy = self.__get_log_energy(stride_input, self._epsilon, energy_floor)

        return stride_input, signal_log_energy      ## type: ignore

    def __get_log_energy(self, strided_input:np.ndarray, epsilon, energy_floor:float=0.0) -> np.ndarray:
        log_energy = np.log(np.maximum(np.sum(strided_input**2, axis=1), epsilon))
        if energy_floor == 0.0:
            return log_energy
        else:
            return np.maximum(log_energy, energy_floor)

    def __get_strided(self, pcm:np.ndarray, win_size:int, win_shift:int, snip_edges:bool=True):
        assert pcm.ndim == 1 and snip_edges
        num_samples = pcm.shape[0]
        strides = (win_shift * pcm.strides[0], pcm.strides[0])
        m = 1 + (num_samples - win_size) // win_shift
        sizes = (m, win_size)
        return np.lib.stride_tricks.as_strided(pcm, shape=sizes, strides=strides)

    def __prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder-1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache
        
    def __run_onnx(self, waveform:np.ndarray, begin_time_ms:int=-1, last:bool=False) -> list:
        segments = []
        if len(waveform) == 0 and not last:
            return segments
        
        if begin_time_ms >= 0:
            self.__begin_time_ms = begin_time_ms

        ## onnx 完整预测
        if len(waveform) > 0:
            feat, _ = self.__extract_feat(waveform)
            inputs = [ feat ]
            inputs.extend(self.__in_cache)  ## type: ignore
            scores, self.__in_cache = self.__onnx_infer(inputs)
        else:
            scores = np.empty((1,0,248), np.float32)
        segments_part = self.vad_scorer(
            scores, waveform[None, ...], is_final=last,
            max_end_sil=self.max_end_sil, online=False
        )
        if segments_part:
            segments.extend(segments_part[0])
        return segments
    
    def __extract_feat(self, waveform: np.ndarray):

        if len(waveform) == 0:
            return np.empty((1, 0, 400), np.float32), np.array([0], np.int32)

        feats = self.__preprocess(waveform)[None, ...]      # (1, n, 400)
        return feats, np.array([feats.shape[1]], dtype=np.int32)

    def __onnx_infer(self, feats) -> Tuple[np.ndarray, np.ndarray]:
        ort_inps = self.ort_infer.get_inputs()
        assert len(ort_inps) == len(feats)
        inps = {}
        for i in range(len(ort_inps)):
            inps[ort_inps[i].name] = feats[i]
        outputs = self.ort_infer.run(None, input_feed=inps)
        scores, out_caches = outputs[0], outputs[1:]
        return cast(np.ndarray, scores), cast(np.ndarray, out_caches)