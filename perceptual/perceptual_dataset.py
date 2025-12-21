"""
孪生网络用的主观不相似度数据集构建。

将 15 条焊接声样本及其 15x15 不相似度矩阵转化为样本对：
- 输入: (spec_i, spec_j)
- 目标: 归一化后的不相似度 d_ij in [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from scipy.signal import stft

from src.config import RAW_DIR, SAMPLE_TO_FILE, DatasetConfig
from src import audio_utils
from src.mds_utils import SAMPLE_ORDER


def _compute_spectrogram(waveform: np.ndarray, sample_rate: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """简易幅度谱，后续 CNN 处理。"""
    _, _, Zxx = stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)
    # 对数幅度，避免 0
    mag = np.log1p(mag)
    return mag.astype(np.float32)


def _load_waveforms(cfg: DatasetConfig) -> Dict[str, np.ndarray]:
    """按 SAMPLE_ORDER 读取并预处理 15 条样本。"""
    waves: Dict[str, np.ndarray] = {}
    for sid in SAMPLE_ORDER:
        path = RAW_DIR / SAMPLE_TO_FILE[sid]
        raw = audio_utils.load_audio(str(path), cfg)
        proc = audio_utils.preprocess_signal(raw, cfg)
        waves[sid] = proc
    return waves


class PerceptualPairDataset(Dataset):
    """
    样本对数据集，每个 item: (spec_i, spec_j, target_diss)
    """

    def __init__(self, cfg: DatasetConfig, diss_matrix: np.ndarray, spectrograms: Dict[str, np.ndarray]):
        self.cfg = cfg
        self.diss = diss_matrix
        self.spectrograms = spectrograms
        self.idx_map = {sid: i for i, sid in enumerate(SAMPLE_ORDER)}
        self.pairs: List[Tuple[str, str]] = []
        for i, sid_i in enumerate(SAMPLE_ORDER):
            for j, sid_j in enumerate(SAMPLE_ORDER):
                if i >= j:
                    continue
                self.pairs.append((sid_i, sid_j))
        # 归一化距离到 [0,1]
        self.max_d = float(np.max(self.diss))
        if self.max_d <= 0:
            self.max_d = 1.0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sid_i, sid_j = self.pairs[idx]
        i, j = self.idx_map[sid_i], self.idx_map[sid_j]
        target = float(self.diss[i, j] / self.max_d)
        spec_i = self.spectrograms[sid_i]
        spec_j = self.spectrograms[sid_j]
        return spec_i, spec_j, target


def build_perceptual_dataset(cfg: DatasetConfig, diss_path: Path | None = None) -> Tuple[PerceptualPairDataset, Dict[str, np.ndarray]]:
    """读取不相似度矩阵与波形，生成 PerceptualPairDataset。"""
    if diss_path is None:
        diss_path = RAW_DIR / "assessment_data.xlsx"
    df = pd.read_excel(diss_path)
    matrices = []
    for _, sub in df.groupby("tester"):
        m = sub.set_index("Sample").reindex(SAMPLE_ORDER)[SAMPLE_ORDER].to_numpy(dtype=float)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        matrices.append(m)
    diss = np.mean(np.stack(matrices, axis=0), axis=0)
    diss = diss - np.min(diss)

    waves = _load_waveforms(cfg)
    spectrograms = {sid: _compute_spectrogram(w, cfg.sample_rate) for sid, w in waves.items()}

    ds = PerceptualPairDataset(cfg, diss, spectrograms)
    return ds, spectrograms
