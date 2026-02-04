# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Union
import copy
from functools import lru_cache

import numpy as np
import kaldi_native_fbank as knf

class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = "./am.mvn",
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 1,
        lfr_n: int = 1,
        # dither: float = 1.0,
        dither: float = 0.0,
        **kwargs,
    ) -> None:

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file

        if self.cmvn_file:
            self.cmvn = load_cmvn(self.cmvn_file)
        self.fbank_fn = None
        self.fbank_beg_idx = 0
        self.reset_status()

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        if len(waveform) > 0 and waveform[0] == waveform[1]:
            waveform[1] += 1    ## FIXME: 禁用 dither 了，这里防止完全相同的数据
        fbank_fn = knf.OnlineFbank(self.opts)
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def fbank_online(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        # self.fbank_fn = knf.OnlineFbank(self.opts)
        if self.fbank_fn is None:
            self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(self.fbank_beg_idx, frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        # self.fbank_beg_idx += (frames-self.fbank_beg_idx)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def reset_status(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs.astype(np.float32)

@lru_cache()
def load_cmvn(cmvn_file: Union[str, Path]) -> np.ndarray:
    """load cmvn file to numpy array. 

    Args:
        cmvn_file (Union[str, Path]): cmvn file path.

    Raises:
        FileNotFoundError: cmvn file not exits.

    Returns:
        np.ndarray: cmvn array. shape is (2, dim).The first row is means, the second row is vars.
    """

    cmvn_file = Path(cmvn_file)
    if not cmvn_file.exists():
        raise FileNotFoundError("cmvn file not exits")
    
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue

    means = np.array(means_list).astype(np.float64)
    vars = np.array(vars_list).astype(np.float64)
    cmvn = np.array([means, vars])
    return cmvn


if __name__ == "__main__":
    ## 从 /media/pub/tmp/nongyan.wav 读取1秒 pcm
    ## 输出 encoder feat
    import os.path as osp
    import yaml
    curr_dir = osp.dirname(osp.abspath(__file__))

    def sread(fname, meta=False):
        import struct as S
        with open(fname, "rb") as f:
            head = f.read(12)
            assert len(head) == 12
            wave = head[8:12]     # "WAVE"
            assert wave == b'WAVE'

            # read fmt chunk
            head = f.read(8)
            assert head[0:4] == b"fmt "
            fmtsize = S.unpack("<L", head[4:8])[0]
            fmtdata = f.read(fmtsize)

            tformat = S.unpack("<H", fmtdata[0:2])[0]
            channels = S.unpack("<H", fmtdata[2:4])[0]
            rate = S.unpack("<L", fmtdata[4:8])[0]
            bits = S.unpack("<H", fmtdata[14:16])[0]

            # 寻找 data 
            head = f.read(8)
            next_head = head[0:4]
            next_size = S.unpack("<L", head[4:8])[0]
            while next_head != b'data':
                # 跳过不认识的 ...
                # print(f"skip unknown key '{next_head}' with size {next_size}")
                f.read(next_size)
                head = f.read(8)
                next_head = head[0:4]
                next_size = S.unpack("<L", head[4:8])[0]

            assert next_head == b'data'
            data = f.read(next_size)

            if tformat == 1:
                # WAVE_FORMAT_PCM
                if bits == 16:
                    pcm = np.frombuffer(data, dtype=np.int16)
                    if meta:
                        return pcm.astype(np.float32) / 2 ** 15, (channels, rate)
                    else:
                        return pcm.astype(np.float32) / 2 ** 15, rate
                elif bits == 32:
                    pcm = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2 ** 31
                    if meta:
                        return pcm, (channels, rate)
                    else:
                        return pcm, rate
                else:
                    raise NotImplementedError()
            elif tformat == 3:
                # WAVE_FORMAT_IEEE_FLOAT
                if bits == 32:
                    if meta:
                        return np.frombuffer(data, dtype=np.float32), (channels, rate)
                    else:
                        return np.frombuffer(data, dtype=np.float32), rate
                else:
                    raise NotImplementedError()
            elif tformat == 65534:
                # FIXME: 这里需要进一步判断格式 ...
                if bits == 32:
                    if meta:
                        return np.frombuffer(data, dtype=np.float32), (channels, rate)
                    else:
                        return np.frombuffer(data, dtype=np.float32), rate
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()    
            
    fname = "/media/pub/tmp/nongyan.wav"
    pcm, sr = sread(fname)
    pcm = pcm.astype(np.float32)[:16000]

    cfg_fname = osp.join(curr_dir, "..", "pred", "model", "asr", "config.yaml")
    cfg = yaml.load(open(cfg_fname), Loader=yaml.FullLoader)
    front = WavFrontend(
        osp.join(curr_dir, "..", "pred", "model", "asr", "am.mvn"),
        **cfg["frontend_conf"]
    )

    speech, _ = front.fbank(pcm)
    feat, _ = front.lfr_cmvn(speech)

    import numpy as np
    np.set_printoptions(precision=5, suppress=True)
    print(f"pcm:{pcm.shape}\n{pcm}")
    print(f"feat:{feat.shape}\n{feat}")

    T = 250
    N = feat.shape[0]
    pad_right = T - N
    feat = np.pad(feat, [(0, pad_right), (0, 0)], mode="constant", constant_values=0)
    feat = feat.astype(np.float32)[None, ...]   # (1, T, 560)

    mask = np.zeros((1, T)).astype(np.float32)
    mask[:, :N] = 1.0
    mask = mask[None, ...]

    np.save("/media/pub/tmp/feat.npy", feat)
    np.save("/media/pub/tmp/mask.npy", mask)
