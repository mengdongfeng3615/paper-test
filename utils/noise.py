"""
Noise utilities for robust TIG arc sound experiments.

支持白噪声 / 粉噪声 / 布朗噪声 / 工厂噪声的动态注入。
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Sequence, List

import soundfile as sf
from scipy import signal


def _colored_noise(n: int, rng: np.random.Generator, power: float = 1.0) -> np.ndarray:
    """
    生成颜色噪声的基础白噪序列，调用方负责频谱整形。
    """
    return rng.standard_normal(n) * np.sqrt(power)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f 粉噪声，频域幅度按 1/sqrt(f) 衰减。"""
    white = _colored_noise(n, rng)
    freqs = np.fft.rfftfreq(n)
    spectrum = np.fft.rfft(white)
    weights = np.ones_like(freqs)
    nz = freqs > 0
    weights[nz] = 1.0 / np.sqrt(freqs[nz])
    pink = np.fft.irfft(spectrum * weights, n=n)
    pow_val = np.mean(pink ** 2)
    if pow_val > 0:
        pink /= np.sqrt(pow_val)
    return pink


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f^2 布朗噪声，频域幅度按 1/f 衰减。"""
    white = _colored_noise(n, rng)
    freqs = np.fft.rfftfreq(n)
    spectrum = np.fft.rfft(white)
    weights = np.ones_like(freqs)
    nz = freqs > 0
    weights[nz] = 1.0 / freqs[nz]
    brown = np.fft.irfft(spectrum * weights, n=n)
    pow_val = np.mean(brown ** 2)
    if pow_val > 0:
        brown /= np.sqrt(pow_val)
    return brown


def load_factory_noises(factory_dir: Path, target_sr: int) -> List[np.ndarray]:
    """
    读取并归一化 data_raw/factory_audio 下的环境噪声。
    """
    noises: List[np.ndarray] = []
    if not factory_dir.exists():
        return noises
    for wav in sorted(factory_dir.glob("*.wav")):
        data, sr = sf.read(wav)
        data = data.astype(np.float64)
        if sr != target_sr:
            num = int(len(data) * target_sr / sr)
            data = signal.resample(data, num)
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
        pow_val = np.mean(data ** 2)
        if pow_val > 0:
            data = data / np.sqrt(pow_val)
        noises.append(data)
    return noises


def _sample_factory_noise(noise_bank: Sequence[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """随机抽取工厂噪声片段并裁剪/循环以匹配长度。"""
    if not noise_bank:
        return rng.standard_normal(length)
    clip = noise_bank[rng.integers(0, len(noise_bank))]
    if len(clip) >= length:
        start = rng.integers(0, len(clip) - length + 1)
        return clip[start : start + length]
    reps = (length // len(clip)) + 1
    tiled = np.tile(clip, reps)
    return tiled[:length]


def add_noise(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    noise_type: str = "white",
    factory_noises: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    按目标 SNR 动态注入噪声。
    noise_type: white | pink | brown | factory
    """
    if np.isinf(snr_db):
        return x.copy()
    power = np.mean(x ** 2)
    if power == 0:
        return x.copy()
    noise_power = power / (10 ** (snr_db / 10))

    if noise_type == "pink":
        noise = _pink_noise(len(x), rng)
    elif noise_type == "brown":
        noise = _brown_noise(len(x), rng)
    elif noise_type == "factory" and factory_noises is not None:
        noise = _sample_factory_noise(factory_noises, len(x), rng)
        npow = np.mean(noise ** 2)
        if npow > 0:
            noise = noise / np.sqrt(npow)
    else:
        noise = rng.normal(size=x.shape)

    noise = noise * np.sqrt(noise_power)
    return x + noise


def frequency_mask(spec: np.ndarray, max_width: int, rng: np.random.Generator) -> np.ndarray:
    """频率遮挡，模仿 SpecAugment。"""
    if max_width <= 0:
        return spec
    spec = spec.copy()
    freq_dim = spec.shape[-2]
    width = rng.integers(0, max_width + 1)
    if width == 0 or width >= freq_dim:
        return spec
    start = rng.integers(0, freq_dim - width)
    spec[..., start : start + width, :] = 0.0
    return spec


def time_mask(spec: np.ndarray, max_width: int, rng: np.random.Generator) -> np.ndarray:
    """时间遮挡，模仿 SpecAugment。"""
    if max_width <= 0:
        return spec
    spec = spec.copy()
    time_dim = spec.shape[-1]
    width = rng.integers(0, max_width + 1)
    if width == 0 or width >= time_dim:
        return spec
    start = rng.integers(0, time_dim - width)
    spec[..., :, start : start + width] = 0.0
    return spec
