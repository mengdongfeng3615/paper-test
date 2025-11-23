"""
Audio loading, preprocessing, segmentation, and noise injection utilities.
"""
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Iterable, List

from .config import DatasetConfig


def load_audio(path: str, cfg: DatasetConfig) -> np.ndarray:
    """Load mono wav and validate sample rate."""
    data, sr = sf.read(path)
    if sr != cfg.sample_rate:
        raise ValueError(f"Expected sample rate {cfg.sample_rate}, got {sr}")
    return data.astype(np.float64)


def _bandpass_filter(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """4th-order Butterworth band-pass around arc acoustics band."""
    sos = signal.butter(
        4,
        [cfg.bandpass_low, cfg.bandpass_high],
        btype="bandpass",
        fs=cfg.sample_rate,
        output="sos",
    )
    return signal.sosfilt(sos, x)


def _notch(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """50 Hz notch to suppress mains hum."""
    b, a = signal.iirnotch(cfg.notch_freq, 30, fs=cfg.sample_rate)
    return signal.filtfilt(b, a, x)


def preprocess_signal(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """Normalize, band-pass, notch, and pre-emphasize waveform."""
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    x = _bandpass_filter(x, cfg)
    x = _notch(x, cfg)
    x = signal.lfilter([1.0, -cfg.pre_emphasis], [1], x)
    return x


def segment_signal(
    x: np.ndarray, cfg: DatasetConfig
) -> List[np.ndarray]:
    """Slide fixed-length windows with hop size to make clips."""
    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    hop = int(cfg.hop_seconds * cfg.sample_rate)
    segments: List[np.ndarray] = []
    for start in range(0, len(x) - seg_len + 1, hop):
        segments.append(x[start : start + seg_len])
    return segments


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at target SNR; inf means no noise."""
    if np.isinf(snr_db):
        return x.copy()
    power = np.mean(x ** 2)
    if power == 0:
        return x.copy()
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=x.shape)
    return x + noise
