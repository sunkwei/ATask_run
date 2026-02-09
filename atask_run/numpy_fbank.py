import numpy as np
import math
from typing import Tuple

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def get_strided(waveform: np.ndarray, window_size: int, window_shift: int, snip_edges: bool) -> np.ndarray:
    num_samples = waveform.shape[0]
    
    if snip_edges:
        if num_samples < window_size:
            return np.empty((0, window_size), dtype=waveform.dtype)
        num_frames = 1 + (num_samples - window_size) // window_shift
        strides = (window_shift * waveform.strides[0], waveform.strides[0])
        return np.lib.stride_tricks.as_strided(waveform, shape=(num_frames, window_size), strides=strides)
    else:
        # 更精确地实现PyTorch的反射填充逻辑
        reversed_waveform = np.flip(waveform)
        num_frames = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform_padded = np.concatenate((pad_left, waveform))
        else:
            waveform_padded = waveform[-pad:]
        
        # 确保有足够的样本用于填充右侧
        pad_right_needed = max(0, num_frames * window_shift + window_size - len(waveform_padded))
        if pad_right_needed > 0:
            pad_right = reversed_waveform[:min(pad_right_needed, len(reversed_waveform))]
            waveform_padded = np.concatenate((waveform_padded, pad_right))
        
        strides = (window_shift * waveform_padded.strides[0], waveform_padded.strides[0])
        return np.lib.stride_tricks.as_strided(waveform_padded, shape=(num_frames, window_size), strides=strides)

def feature_window_function(window_type: str, window_size: int, blackman_coeff: float) -> np.ndarray:
    if window_type == 'hanning':
        return np.hanning(window_size)
    elif window_type == 'hamming':
        return np.hamming(window_size)
    elif window_type == 'povey':
        return np.hanning(window_size) ** 0.85
    elif window_type == 'rectangular':
        return np.ones(window_size)
    elif window_type == 'blackman':
        a = 2 * np.pi / (window_size - 1)
        window = np.arange(window_size)
        return blackman_coeff - 0.5 * np.cos(a * window) + (0.5 - blackman_coeff) * np.cos(2 * a * window)
    else:
        raise ValueError(f"Invalid window type: {window_type}")

def get_log_energy(strided_input: np.ndarray, energy_floor: float) -> np.ndarray:
    epsilon = np.finfo(strided_input.dtype).eps
    log_energy = np.log(np.maximum(np.sum(strided_input ** 2, axis=1), epsilon))
    if energy_floor > 0.0:
        return np.maximum(log_energy, math.log(energy_floor))
    return log_energy

def mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)

def inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)

def vtln_warp_freq(vtln_low_cutoff: float, vtln_high_cutoff: float, 
                  low_freq: float, high_freq: float, 
                  vtln_warp_factor: float, freq: np.ndarray) -> np.ndarray:
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l
    Fh = scale * h
    
    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)
    
    res = np.zeros_like(freq)
    
    # 创建掩码
    outside_low_high = (freq < low_freq) | (freq > high_freq)
    before_l = freq < l
    before_h = freq < h
    after_h = freq >= h
    
    # 应用分段线性函数
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h & ~before_l] = scale * freq[before_h & ~before_l]  # 中间段
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high] = freq[outside_low_high]
    
    return res

def get_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float,
                 low_freq: float, high_freq: float, vtln_low: float,
                 vtln_high: float, vtln_warp_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    assert num_bins > 3, "Must have at least 3 mel bins"
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    
    if high_freq <= 0.0:
        high_freq += nyquist
    
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale(np.array([low_freq]))[0]
    mel_high_freq = mel_scale(np.array([high_freq]))[0]
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    
    if vtln_high < 0.0:
        vtln_high += nyquist
    
    bin_indices = np.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low_freq + bin_indices * mel_freq_delta
    center_mel = mel_low_freq + (bin_indices + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_indices + 2.0) * mel_freq_delta
    
    if vtln_warp_factor != 1.0:
        left_mel = mel_scale(vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(left_mel)))
        center_mel = mel_scale(vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(center_mel)))
        right_mel = mel_scale(vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(right_mel)))
    
    center_freqs = inverse_mel_scale(center_mel)
    fft_bins = np.arange(num_fft_bins) * fft_bin_width
    mel = mel_scale(fft_bins).reshape(1, -1)
    
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    
    if vtln_warp_factor == 1.0:
        bins = np.maximum(0.0, np.minimum(up_slope, down_slope))
    else:
        bins = np.zeros_like(up_slope)
        up_idx = (mel > left_mel) & (mel <= center_mel)
        down_idx = (mel > center_mel) & (mel < right_mel)
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]
    
    return bins, center_freqs.reshape(-1)

def fbank(waveform: np.ndarray, sample_frequency: float = 16000.0,
         window_type: str = 'povey', blackman_coeff: float = 0.42,
         channel: int = -1, dither: float = 0.0, energy_floor: float = 1.0,
         frame_length: float = 25.0, frame_shift: float = 10.0,
         high_freq: float = 0.0, low_freq: float = 20.0,
         num_mel_bins: int = 80, preemphasis_coefficient: float = 0.97,
         raw_energy: bool = True, remove_dc_offset: bool = True,
         round_to_power_of_two: bool = True, snip_edges: bool = True,
         subtract_mean: bool = True, use_energy: bool = False,
         use_log_fbank: bool = True, use_power: bool = True,
         vtln_high: float = -500.0, vtln_low: float = 100.0,
         vtln_warp: float = 1.0, htk_compat: bool = False,
         min_duration: float = 0.0) -> np.ndarray:
    # 参数转换
    window_shift = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded_window_size = next_power_of_2(window_size) if round_to_power_of_two else window_size
    
    # 通道选择
    if channel >= 0 and waveform.ndim > 1:
        waveform = waveform[channel]
    elif waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    
    # 检查最短时长
    if len(waveform) < min_duration * sample_frequency:
        return np.empty((0, num_mel_bins + (1 if use_energy else 0)), dtype=np.float32)
    
    # 分帧 - 使用优化后的实现
    strided_input = get_strided(waveform, window_size, window_shift, snip_edges)
    if strided_input.size == 0:
        return np.empty((0, num_mel_bins + (1 if use_energy else 0)), dtype=np.float32)
    
    # 加噪 - 使用双精度计算以提高精度
    if dither != 0.0:
        rng = np.random.default_rng(seed=42)  # 固定种子以确保可重复性
        strided_input = strided_input.astype(np.float64) + rng.standard_normal(strided_input.shape) * dither
    
    # 去直流偏移
    if remove_dc_offset:
        row_means = np.mean(strided_input, axis=1, keepdims=True)
        strided_input -= row_means
    
    # 计算原始能量（如果需要）
    if raw_energy:
        signal_log_energy = get_log_energy(strided_input, energy_floor)
    
    # 预加重（在分帧后执行，与PyTorch一致）
    if preemphasis_coefficient != 0.0:
        # 使用双精度计算以提高精度
        offset = np.pad(strided_input[:, :-1], ((0, 0), (1, 0)), mode='edge')
        strided_input = strided_input.astype(np.float64) - preemphasis_coefficient * offset.astype(np.float64)
    
    # 加窗
    window = feature_window_function(window_type, window_size, blackman_coeff)
    strided_input = strided_input.astype(np.float64) * window.astype(np.float64)
    
    # 填充到2的幂
    if padded_window_size > window_size:
        padded_strided_input = np.zeros((strided_input.shape[0], padded_window_size), dtype=np.float64)
        padded_strided_input[:, :window_size] = strided_input
    else:
        padded_strided_input = strided_input
    
    # 计算能量（如果未使用原始能量）
    if not raw_energy:
        signal_log_energy = get_log_energy(padded_strided_input, energy_floor)
    
    # 计算FFT - 使用双精度
    spectrum = np.fft.rfft(padded_strided_input, axis=1)
    magnitude = np.abs(spectrum)
    
    # 使用功率或幅度
    if use_power:
        spectrum_energy = magnitude ** 2
    else:
        spectrum_energy = magnitude
    
    # 创建Mel滤波器组
    mel_banks, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency,
                                low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    
    # 填充Nyquist bin
    mel_banks = np.pad(mel_banks, ((0, 0), (0, 1)), 'constant', constant_values=0)
    
    # 应用Mel滤波器组 - 使用双精度
    mel_energies = np.dot(spectrum_energy.astype(np.float64), mel_banks.T.astype(np.float64))
    
    # 对数变换
    if use_log_fbank:
        mel_energies = np.log(np.maximum(mel_energies, np.finfo(np.float64).eps))
    
    # 添加能量特征
    if use_energy:
        signal_log_energy = signal_log_energy.reshape(-1, 1)
        if htk_compat:
            mel_energies = np.hstack((mel_energies, signal_log_energy))
        else:
            mel_energies = np.hstack((signal_log_energy, mel_energies))
    
    # 减去列均值
    if subtract_mean:
        mel_energies -= np.mean(mel_energies, axis=0, keepdims=True)
    
    # 转换为单精度返回
    return mel_energies.astype(np.float32)

if __name__ == '__main__':
    import librosa,sys

    # 加载音频文件
    waveform, sample_rate = librosa.load(sys.argv[1], sr=16000)

    # 计算FBank特征
    features = fbank(
        waveform,
        sample_frequency=sample_rate,
        num_mel_bins=80,
        use_energy=False,
        subtract_mean=True,
    )

    print(features)
