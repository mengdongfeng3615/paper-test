"""
鲁棒数据加载与噪声增强 DataLoader（PyTorch 风格）。

特性：
- on-the-fly 噪声注入：white/pink/brown/factory + 多级 SNR。
- 可选 SpecAugment（频率/时间遮挡）。
- 兼容现有 Segment 结构（src.data_utils.Segment），也可直接传入 waveform 数组。
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, List, Optional, Sequence, Tuple, Dict, Any

from utils.noise import add_noise, load_factory_noises, frequency_mask, time_mask


class RobustLoaderConfig:
    """数据加载与增强配置。"""

    def __init__(
        self,
        sample_rate: int = 40_000,
        noise_types: Sequence[str] = ("white", "pink", "brown", "factory"),
        snr_choices: Sequence[float] = (float("inf"), 10.0, 8.0, 5.0, 2.0, 0.5),
        use_spec_augment: bool = False,
        freq_mask_width: int = 4,
        time_mask_width: int = 8,
        factory_dir: Optional[Path] = None,
        random_seed: int = 42,
    ):
        self.sample_rate = sample_rate
        self.noise_types = tuple(noise_types)
        self.snr_choices = tuple(snr_choices)
        self.use_spec_augment = use_spec_augment
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.factory_dir = factory_dir
        self.random_seed = random_seed


class RobustWeldSoundDataset(Dataset):
    """
    输入 data_index: 序列，元素包含 {sample_id, label, waveform} 字段。
    transforms: 可选函数，将 waveform -> 特征张量（如声谱图）。
    """

    def __init__(
        self,
        data_index: Sequence[Dict[str, Any]],
        config: RobustLoaderConfig,
        noise_bank: Optional[Sequence[np.ndarray]] = None,
        transforms: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fixed_noise: Optional[Tuple[str, float]] = None,
    ):
        self.data_index = list(data_index)
        self.cfg = config
        self.transforms = transforms
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.factory_noises = (
            list(noise_bank)
            if noise_bank is not None
            else load_factory_noises(self.cfg.factory_dir, self.cfg.sample_rate)
        )
        self.fixed_noise = fixed_noise  # (noise_type, snr_db) 固定评估用

    def __len__(self) -> int:
        return len(self.data_index)

    def _choose_noise(self) -> Tuple[str, float]:
        if self.fixed_noise is not None:
            return self.fixed_noise
        ntype = self.rng.choice(self.cfg.noise_types)
        snr = float(self.rng.choice(self.cfg.snr_choices))
        return ntype, snr

    def __getitem__(self, idx: int):
        item = self.data_index[idx]
        wav: np.ndarray = np.asarray(item["waveform"], dtype=np.float64)
        label = int(item["label"])
        sample_id = item.get("sample_id", f"sample_{idx}")

        ntype, snr_db = self._choose_noise()
        noisy = add_noise(wav, snr_db, self.rng, noise_type=ntype, factory_noises=self.factory_noises)

        feat = noisy
        if self.transforms is not None:
            feat = self.transforms(noisy)
            if self.cfg.use_spec_augment and feat.ndim == 2:
                # 期望形状 (freq, time)
                feat = frequency_mask(feat, self.cfg.freq_mask_width, self.rng)
                feat = time_mask(feat, self.cfg.time_mask_width, self.rng)

        return {
            "feature": torch.from_numpy(feat).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "noise_type": ntype,
            "snr_db": snr_db,
        }


# 训练脚本使用示例（伪代码）：
# from torch.utils.data import DataLoader
# from data.robust_loader import RobustWeldSoundDataset, RobustLoaderConfig
# segments = load_segments(DatasetConfig())  # 复用现有 src.data_utils
# data_index = [{"sample_id": s.sample_id, "label": s.label, "waveform": s.data} for s in segments]
# cfg = RobustLoaderConfig(factory_dir=Path("data_raw/factory_audio"), use_spec_augment=True)
# ds = RobustWeldSoundDataset(data_index, cfg, transforms=to_logmel_fn)
# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
