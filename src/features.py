"""
Feature extraction: basic stats, perceptual cepstra, and optional MDS fusion.
"""
import numpy as np
from typing import List, Optional

from spafe.features.mfcc import mfcc as spafe_mfcc
from spafe.features.pncc import pncc as spafe_pncc
from spafe.features.rplp import rplp as spafe_rplp
from spafe.utils.preprocessing import SlidingWindow

from .config import DatasetConfig


def _zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.mean(np.diff(np.sign(x)) != 0))


def _spectral_stats(x: np.ndarray, sr: int):
    """Compute spectral centroid/bandwidth/rolloff/flatness and band energy ratios."""
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = np.sum(power) + 1e-12
    centroid = float(np.sum(freqs * power) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
    cumsum = np.cumsum(power)
    roll_idx = np.searchsorted(cumsum, 0.95 * total)
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])
    flatness = float(
        np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12)
    )
    bands = [(0, 1000), (1000, 8000), (8000, 18_000)]
    ratios: List[float] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        ratios.append(float(np.sum(power[mask]) / total))
    return centroid, bandwidth, rolloff, flatness, ratios

#时域
def _basic_features(x: np.ndarray, sr: int) -> List[float]:
    rms = float(np.sqrt(np.mean(x ** 2)))
    zcr = _zero_crossing_rate(x)
    centroid, bandwidth, rolloff, flatness, ratios = _spectral_stats(x, sr)
    return [rms, zcr, centroid, bandwidth, rolloff, flatness, *ratios]


def _aggregate(feats: np.ndarray) -> np.ndarray:
    """Summarize frame-level features by mean and std."""
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


def _frame_window(cfg: DatasetConfig) -> SlidingWindow:
    """Use window <= segment length to avoid negative framing."""
    win_len = min(0.025, cfg.segment_seconds)
    win_hop = min(0.01, cfg.hop_seconds)
    return SlidingWindow(win_len, win_hop)


def _mfcc_features(x: np.ndarray, sr: int, cfg: DatasetConfig) -> np.ndarray:
    feats = spafe_mfcc(
        sig=x,
        fs=sr,
        num_ceps=cfg.n_mfcc,
        pre_emph=False,
        window=_frame_window(cfg),
        nfilts=26,
        nfft=1024,
        low_freq=0,
        high_freq=sr / 2,
    )
    return _aggregate(feats)


def _pncc_features(x: np.ndarray, sr: int, cfg: DatasetConfig) -> np.ndarray:
    feats = spafe_pncc(
        sig=x,
        fs=sr,
        num_ceps=cfg.n_pncc,
        pre_emph=False,
        window=_frame_window(cfg),
        nfilts=26,
        nfft=1024,
        low_freq=0,
        high_freq=sr / 2,
    )
    return _aggregate(feats)


def _rasta_features(x: np.ndarray, sr: int, cfg: DatasetConfig) -> np.ndarray:
    feats = spafe_rplp(
        sig=x,
        fs=sr,
        order=cfg.n_rasta,
        pre_emph=False,
        window=_frame_window(cfg),
        nfilts=26,
        nfft=1024,
        low_freq=0,
        high_freq=sr / 2,
    )
    return _aggregate(feats)


def build_feature_vector(
    x: np.ndarray, sr: int, cfg: DatasetConfig, mds_coord: Optional[np.ndarray]
) -> np.ndarray:
    """Concatenate hand-crafted features and optional MDS coordinates."""
    base = np.array(_basic_features(x, sr))
    mfcc_feat = _mfcc_features(x, sr, cfg)
    pncc_feat = _pncc_features(x, sr, cfg)
    rasta_feat = _rasta_features(x, sr, cfg)
    feats = np.concatenate([base, mfcc_feat, pncc_feat, rasta_feat])
    if mds_coord is not None:
        feats = np.concatenate([feats, mds_coord])
    return feats.astype(np.float32)
