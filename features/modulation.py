"""
调制谱特征：包络 -> FFT -> 频带能量。
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Sequence, Tuple


def modulation_spectrum(
    waveform: np.ndarray,
    sample_rate: int,
    bandpass: Tuple[float, float] = (500.0, 2000.0),
    n_fft: int = 1024,
    bands_hz: Sequence[Tuple[float, float]] = ((2, 8), (8, 20), (20, 50), (50, 100)),
) -> np.ndarray:
    """
    计算调制谱能量分布。
    返回形状 (len(bands_hz),) 的能量向量。
    """
    # 带通以突出语义相关频段
    sos = signal.butter(4, bandpass, btype="band", fs=sample_rate, output="sos")
    x = signal.sosfilt(sos, waveform)
    # 包络
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    # 低采样避免过长 FFT
    decim = max(1, int(sample_rate / 4000))
    env_ds = signal.decimate(env, decim, ftype="fir", zero_phase=True)
    mod_fs = sample_rate / decim
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / mod_fs)
    spec = np.abs(np.fft.rfft(env_ds, n=n_fft)) ** 2
    energies = []
    for lo, hi in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        energies.append(float(np.sum(spec[mask])))
    energies = np.array(energies, dtype=np.float32)
    # 归一化到能量和
    total = np.sum(energies)
    if total > 0:
        energies = energies / total
    return energies
