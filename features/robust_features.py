"""
鲁棒声学特征封装：
- GFCC
- 调制谱
- 传统统计特征（RMS、ZCR、谱质心、谱带宽、rolloff、谱平坦度、分频带能量比）

返回:
    frame_feat: 时间 x 频率/倒谱维度（供注意力/序列模型）
    global_feat: 1D 特征向量，聚合多种统计
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Tuple, Sequence

from features.gfcc import gfcc
from features.modulation import modulation_spectrum


def _frame_stats(waveform: np.ndarray, sample_rate: int, n_fft: int = 1024, hop: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """返回功率谱和频率坐标。"""
    freqs, _, Zxx = signal.stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    power = np.abs(Zxx) ** 2
    return freqs, power


def _spectral_features(freqs: np.ndarray, power: np.ndarray, sample_rate: int) -> np.ndarray:
    """传统频谱统计特征。"""
    # 频谱质心
    sc = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    # 频谱带宽
    bw = np.sqrt(np.sum(((freqs[:, None] - sc) ** 2) * power, axis=0) / (np.sum(power, axis=0) + 1e-10))
    # 95% rolloff
    cumsum = np.cumsum(power, axis=0)
    thresh = 0.95 * cumsum[-1, :]
    rolloff_idx = [np.searchsorted(cumsum[:, i], thresh[i]) for i in range(cumsum.shape[1])]
    rolloff_freq = np.array([freqs[min(r, len(freqs) - 1)] for r in rolloff_idx])
    # 平坦度
    # 简化：用几何均值/算术均值近似
    gm = np.exp(np.mean(np.log(power + 1e-10), axis=0))
    am = np.mean(power + 1e-10, axis=0)
    flat = gm / (am + 1e-10)
    # 分带能量
    bands = [(0, 3000), (3000, 8000), (8000, 18000)]
    band_energies = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_energies.append(np.sum(power[mask, :], axis=0))
    band_energies = np.stack(band_energies, axis=0)
    band_sum = np.sum(band_energies, axis=0) + 1e-10
    band_ratio = band_energies / band_sum
    # 聚合为均值与标准差
    feats = [
        np.mean(sc),
        np.std(sc),
        np.mean(bw),
        np.std(bw),
        np.mean(rolloff_freq),
        np.std(rolloff_freq),
        np.mean(flat),
        np.std(flat),
    ]
    # 各分带能量比均值
    for k in range(band_ratio.shape[0]):
        feats.append(np.mean(band_ratio[k]))
    return np.array(feats, dtype=np.float32)


def _time_features(waveform: np.ndarray) -> np.ndarray:
    """时域特征：RMS、ZCR、短时能量均值/方差。"""
    rms = np.sqrt(np.mean(waveform ** 2))
    zc = ((waveform[:-1] * waveform[1:]) < 0).mean()
    return np.array([rms, zc], dtype=np.float32)


def extract_robust_features(
    waveform: np.ndarray,
    sample_rate: int,
    use_gfcc: bool = True,
    use_modulation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (frame_feat, global_feat)。
    frame_feat: 默认使用 GFCC 序列 (T, C)；可根据需要替换为 log-mel 等。
    global_feat: 包含时域/频域统计、调制谱、GFCC 均值方差。
    """
    frame_feat = gfcc(waveform, sample_rate) if use_gfcc else None
    gfcc_stats = []
    if frame_feat is not None and frame_feat.size > 0:
        gfcc_stats = [np.mean(frame_feat, axis=0), np.std(frame_feat, axis=0)]
    freqs, power = _frame_stats(waveform, sample_rate)
    spec_stats = _spectral_features(freqs, power, sample_rate)
    time_stats = _time_features(waveform)
    mod_feat = modulation_spectrum(waveform, sample_rate) if use_modulation else np.array([], dtype=np.float32)

    global_feat_parts = [time_stats, spec_stats, mod_feat]
    if gfcc_stats:
        global_feat_parts.extend(gfcc_stats)
    global_feat = np.concatenate(global_feat_parts).astype(np.float32)

    if frame_feat is None:
        # 若未使用 GFCC，则用功率谱作帧特征
        frame_feat = np.log1p(power).T.astype(np.float32)  # (time, freq)
    return frame_feat, global_feat
