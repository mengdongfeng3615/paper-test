"""
Audio loading, preprocessing, segmentation, and noise injection utilities.
"""
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Iterable, List, Optional, Sequence

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


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f pink noise via frequency-domain shaping."""
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    spectrum = np.fft.rfft(white)
    # avoid divide-by-zero on DC
    weights = np.ones_like(freqs)
    nz = freqs > 0
    weights[nz] = 1.0 / np.sqrt(freqs[nz])
    pink = np.fft.irfft(spectrum * weights, n=n)
    # normalize to unit power
    pow_pink = np.mean(pink ** 2)
    if pow_pink > 0:
        pink /= np.sqrt(pow_pink)
    return pink


def _sample_factory_noise(noise_bank: Sequence[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """Pick a random factory noise clip and slice/loop to match length."""
    if not noise_bank:
        return rng.standard_normal(length)
    clip = noise_bank[rng.integers(0, len(noise_bank))]
    if len(clip) >= length:
        start = rng.integers(0, len(clip) - length + 1)
        return clip[start:start + length]
    # if shorter, tile then trim
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
    Add noise at target SNR. noise_type supports:
    - "white": Gaussian white noise (default)
    - "pink": 1/f colored noise
    - "factory": mix-in from provided factory_noises bank
    """
    if np.isinf(snr_db):
        return x.copy()
    power = np.mean(x ** 2)
    if power == 0:
        return x.copy()
    noise_power = power / (10 ** (snr_db / 10))

    if noise_type == "pink":
        noise = _pink_noise(len(x), rng)
    elif noise_type == "factory" and factory_noises is not None:
        noise = _sample_factory_noise(factory_noises, len(x), rng)
        # normalize factory noise to unit power
        npow = np.mean(noise ** 2)
        if npow > 0:
            noise = noise / np.sqrt(npow)
    else:
        noise = rng.normal(size=x.shape)

    noise = noise * np.sqrt(noise_power)
    return x + noise
