from typing import List
import numpy as np
import soundfile as sf
from scipy import signal

from .config import RobustConfig


def load_audio(path: str, cfg: RobustConfig) -> np.ndarray:
    data, sr = sf.read(path)
    if sr != cfg.sample_rate:
        raise ValueError(f"Expected {cfg.sample_rate} Hz, got {sr}")
    return data.astype(np.float64)


def preprocess(x: np.ndarray, cfg: RobustConfig) -> np.ndarray:
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    sos = signal.butter(
        4,
        [cfg.bandpass_low, cfg.bandpass_high],
        btype="bandpass",
        fs=cfg.sample_rate,
        output="sos",
    )
    x = signal.sosfilt(sos, x)
    b, a = signal.iirnotch(cfg.notch_freq, 30, fs=cfg.sample_rate)
    x = signal.filtfilt(b, a, x)
    x = signal.lfilter([1.0, -cfg.pre_emphasis], [1], x)
    return x


def segment_signal(x: np.ndarray, cfg: RobustConfig) -> List[np.ndarray]:
    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    hop = int(cfg.hop_seconds * cfg.sample_rate)
    segments: List[np.ndarray] = []
    for start in range(0, len(x) - seg_len + 1, hop):
        segments.append(x[start : start + seg_len])
    return segments


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if np.isinf(snr_db):
        return x.copy()
    power = np.mean(x ** 2)
    if power <= 0:
        return x.copy()
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=x.shape)
    return x + noise


def augment_signal(
    x: np.ndarray,
    snr_levels: List[float],
    aug_per_level: int,
    rng: np.random.Generator,
) -> List[tuple]:
    """Return list of (snr, noisy_signal) across specified levels and seeds."""
    augmented: List[tuple] = []
    for snr in snr_levels:
        for _ in range(aug_per_level):
            noise_rng = np.random.default_rng(rng.integers(0, 1_000_000))
            augmented.append((snr, add_noise(x, snr, noise_rng)))
    return augmented
