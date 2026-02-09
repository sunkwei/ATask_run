import numpy as np
# from uty_mel import Uty
import time, math

class V4Preprocess:

    def __init__(self):
        self.fb = self.build_fb_mat()
    
    def build_fb_mat(self, n_freqs=257, sample_rate=16000, f_min=20, f_max=7600, n_mels=80):
        # 来自 torchaudio/functional.py create_fb_matrix
        all_freqs = np.linspace(0, sample_rate // 2, n_freqs).astype(np.float32)
        m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        m_pts = np.linspace(m_min, m_max, n_mels + 2).astype(np.float32)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.reshape((1, -1)) - all_freqs.reshape((-1, 1)) # (161, 66)
        zero = np.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1] # down_slopes: (161, 64)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (161, 64)
        fb = np.maximum(zero, np.minimum(down_slopes, up_slopes)) # (freq, n_mels)
        return fb.astype(np.float32)
    
    def pcm_mel_spec(self, pcms, duration=2*16000+240):
        feats = []
        pre_emphasis = 0.97
        n_fft = 512
        win_length = 400
        stride = 160
        win_weight = np.hamming(win_length)
        for pcm_seg in pcms:
            # assert pcm_seg.size == duration
            if pcm_seg.size < duration:
                pcm_seg = np.pad(pcm_seg, (0, duration-pcm_seg.size), "wrap")
            elif pcm_seg.size > duration:
                pcm_seg = pcm_seg[:duration]
            # pre emphasis
            pcm_seg = np.pad(pcm_seg, (1, 0), "reflect")            # x2,x1,x2,x3,....,xn
            pcm_seg = pcm_seg[1:] - pre_emphasis * pcm_seg[:-1]     # Xn - 0.97*Xn-1
            # center signal，但这样会导致生成的片数更多了 ...
            pcm_seg = np.pad(pcm_seg, (n_fft // 2, n_fft // 2), "wrap")
            head, tail = 0, pcm_seg.size
            fs = []
            while head < tail:
                n = min(tail - head, win_length)
                s = pcm_seg[head:head+n]
                if n < stride:
                    break
                if n < win_length:
                    s = np.pad(s, (0, win_length-n), "wrap")
                wined = s * win_weight
                f = np.fft.rfft(wined, n=n_fft)     # 
                f = np.abs(f * np.conj(f))          # a^2 + b^2
                fs.append(f)
                head += stride
            spectrum = np.stack(fs).T.astype(np.float32)
            spectrum = spectrum[:,:-2]
            if 0:#fb is None:
                feats.append(spectrum)
            else:
                mel_spec = np.matmul(spectrum.T, self.fb).T + 1e-6
                e = np.log(mel_spec)
                e -= np.mean(e, axis=-1, keepdims=True)
                feats.append(e)
        return feats
    
    def pcm_segs(self, pcm, duration=32240):
        one_size = duration
        if pcm.size < one_size:
            shortage = one_size - pcm.size
            pcm = np.pad(pcm, (0, shortage), "wrap")
        pcms = []
        N = min((pcm.size - one_size) // 16000 + 1, 5)    # 最多5份，最少1份
        startframe = np.linspace(0, pcm.size - one_size, num=N)
        for asf in startframe:
            pcms.append(pcm[int(asf):int(asf) + one_size])
        return pcms
    
    def __call__(self, d):
        # 4分类预测
        pcm16_f32_mono = d.astype(np.float32)
        slice_size = 16000# 1秒
        head, tail = 0, len(pcm16_f32_mono)
        pcm = pcm16_f32_mono
        inps = []
        time_during = []
        inps_list = []

        while head < tail:
            n = min(slice_size, tail - head)
            ## 4分类应该扔掉 < 200ms 的片段
            if n < 3200 and tail >= 16000:
                break

            time_during.append(n / 16000)
            s = pcm[head: head + n]
            if n < slice_size:
                # s = np.pad(s, (0, slice_size - n), "reflect")
                s = np.pad(s, (0, slice_size - n), "wrap")
            head += slice_size
            # 从 1秒补齐到两秒
            segs = self.pcm_segs(s)
            mel_spect = self.pcm_mel_spec(segs)
            assert len(mel_spect) == 1
            inps.append(mel_spect[0])   # (80, 202)

            if len(inps) >= 5:
                inps = np.ascontiguousarray(np.stack(inps))
                inps = inps[None, ...]
                inps = inps.transpose(1, 0, 2, 3)

                inps_list.append(inps)
                inps = []

        if len(inps) > 0:
            inps = np.ascontiguousarray(np.stack(inps))
            inps = inps[None, ...]
            inps = inps.transpose(1, 0, 2, 3)
            inps_list.append(inps)
        
        return inps_list

        
        

if __name__ == "__main__":
    import soundfile as sf
    import sys,time

    VP = V4Preprocess()
    wav, f = sf.read(sys.argv[1])
    wav = wav[:32000]
    s = time.time()
    r = VP(wav)
    print (time.time() - s)
    s = time.time()
    r = VP(wav)
    print (time.time() - s)
    s = time.time()
    r = VP(wav)
    print (time.time() - s)
    
    
