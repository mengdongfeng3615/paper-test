"""
GFCC（Gammatone Frequency Cepstral Coefficients）提取。

实现思路：
- 使用 gammatone 滤波器组（若 scipy 有 gammatone，则调用；否则退化为 mel 滤波器）。
- 取滤波输出的能量，做对数与 DCT 得到倒谱系数。
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fftpack import dct


def _gammatone_filters(sample_rate: int, n_filters: int, n_fft: int, fmin: float, fmax: float):
    """生成简易 gammatone 滤波器组频率响应（幅度版）。"""
    try:
        center_freqs = np.linspace(fmin, fmax, n_filters)
        w, h_bank = [], []
        freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        for fc in center_freqs:
            # 近似：带通峰值在 fc，带宽按 ERB 标度
            erb = 24.7 * (4.37e-3 * fc + 1.0)
            b = 1.019 * 2 * np.pi * erb
            # 频率响应简化为 Lorentzian 形状
            H = 1.0 / (1 + ((freqs - fc) / (erb + 1e-6)) ** 2)
            h_bank.append(H)
        H = np.stack(h_bank, axis=0)
    except Exception:
        # 退化为等间隔带通
        H = np.zeros((n_filters, n_fft // 2 + 1))
        freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        bands = np.linspace(fmin, fmax, n_filters + 2)
        for i in range(n_filters):
            left, center, right = bands[i], bands[i + 1], bands[i + 2]
            H[i] = np.maximum(0, 1 - np.abs((freqs - center) / (center - left)))
    return H


def gfcc(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_filters: int = 24,
    n_ceps: int = 20,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    计算 GFCC，返回 (n_frames, n_ceps)
    """
    # STFT
    _, _, Zxx = signal.stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    power_spec = np.abs(Zxx) ** 2  # (freq, time)
    fbanks = _gammatone_filters(sample_rate, n_filters, n_fft, fmin, fmax)
    fbanks = fbanks[:, : power_spec.shape[0]]
    energy = np.dot(fbanks, power_spec) + 1e-10  # (n_filters, time)
    log_energy = np.log(energy)
    ceps = dct(log_energy, type=2, axis=0, norm="ortho")  # (n_filters, time)
    ceps = ceps[:n_ceps].T  # (time, n_ceps)
    return ceps.astype(np.float32)
