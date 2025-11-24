"""
Feature extraction: GFCC, log-mel statistics, and wavelet packet energy entropy.
"""
import numpy as np
import pywt
from typing import Dict

from spafe.features.gfcc import gfcc as spafe_gfcc
from spafe.features.mfcc import mel_spectrogram
from spafe.utils.preprocessing import SlidingWindow

from .config import RobustConfig


def _frame_window() -> SlidingWindow:
    return SlidingWindow(0.025, 0.010)


def gfcc_stats(x: np.ndarray, sr: int, cfg: RobustConfig) -> np.ndarray:
    feats = spafe_gfcc(
        sig=x,
        fs=sr,
        num_ceps=cfg.gfcc_ceps,
        pre_emph=False,
        window=_frame_window(),
        nfilts=40,
        nfft=1024,
        low_freq=0,
        high_freq=sr / 2,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


def logmel_stats(x: np.ndarray, sr: int, cfg: RobustConfig) -> np.ndarray:
    mel_spec, _ = mel_spectrogram(
        x,
        fs=sr,
        pre_emph=False,
        window=_frame_window(),
        nfilts=cfg.mel_bins,
        nfft=1024,
        low_freq=0,
        high_freq=sr / 2,
    )
    mel_spec = np.maximum(mel_spec, 1e-6)
    mel_log = np.log(mel_spec)
    return np.concatenate([mel_log.mean(axis=0), mel_log.std(axis=0)])


def wavelet_entropy(x: np.ndarray, cfg: RobustConfig) -> np.ndarray:
    wp = pywt.WaveletPacket(data=x, wavelet=cfg.wavelet, mode="symmetric", maxlevel=cfg.wavelet_level)
    leaves = wp.get_level(cfg.wavelet_level, order="freq")
    energies = np.array([np.sum(np.square(node.data)) for node in leaves])
    total = energies.sum() + 1e-12
    probs = energies / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    norm_energy = energies / energies.max()
    return np.concatenate([norm_energy, [entropy]])


def extract_features(x: np.ndarray, sr: int, cfg: RobustConfig) -> np.ndarray:
    gfcc_feat = gfcc_stats(x, sr, cfg)
    mel_feat = logmel_stats(x, sr, cfg)
    wave_feat = wavelet_entropy(x, cfg)
    return np.concatenate([gfcc_feat, mel_feat, wave_feat]).astype(np.float32)
