"""
Config for robust MDS-to-class pipeline with heavy noise augmentation.
"""
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "robust_outputs"


@dataclass(frozen=True)
class RobustConfig:
    sample_rate: int = 40_000
    segment_seconds: float = 0.05
    hop_seconds: float = 0.05
    bandpass_low: float = 300.0
    bandpass_high: float = 18_000.0
    notch_freq: float = 50.0
    pre_emphasis: float = 0.97
    # training only clean; val/test cover multiple SNRs
    snr_train: List[float] = (float("inf"),)
    snr_val: List[float] = (float("inf"), 10.0, 8.0, 5.0, 0.5)
    snr_test: List[float] = (float("inf"), 10.0, 8.0, 5.0, 0.5)
    aug_train: int = 1
    aug_val: int = 3
    aug_test: int = 3
    gfcc_ceps: int = 20
    mel_bins: int = 64
    wavelet: str = "db4"
    wavelet_level: int = 4
    random_seed: int = 123


SAMPLE_TO_FILE: Dict[str, str] = {
    "Sample1": "non-penetration1.wav",
    "Sample2": "non-penetration2.wav",
    "Sample3": "non-penetration3.wav",
    "Sample4": "non-penetration4.wav",
    "Sample5": "non-penetration5.wav",
    "Sample6": "full-penetration1.wav",
    "Sample7": "full-penetration2.wav",
    "Sample8": "full-penetration3.wav",
    "Sample9": "full-penetration4.wav",
    "Sample10": "full-penetration5.wav",
    "Sample11": "excessive-penetration1.wav",
    "Sample12": "excessive-penetration2.wav",
    "Sample13": "excessive-penetration3.wav",
    "Sample14": "excessive-penetration4.wav",
    "Sample15": "excessive-penetration5.wav",
}

CLASS_MAP: Dict[str, int] = {
    "Sample1": 0,
    "Sample2": 0,
    "Sample3": 0,
    "Sample4": 0,
    "Sample5": 0,
    "Sample6": 1,
    "Sample7": 1,
    "Sample8": 1,
    "Sample9": 1,
    "Sample10": 1,
    "Sample11": 2,
    "Sample12": 2,
    "Sample13": 2,
    "Sample14": 2,
    "Sample15": 2,
}

CLASS_NAMES = {0: "未熔透", 1: "全熔透", 2: "过熔透"}

CLASS_BUCKETS = {
    0: ["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"],
    1: ["Sample6", "Sample7", "Sample8", "Sample9", "Sample10"],
    2: ["Sample11", "Sample12", "Sample13", "Sample14", "Sample15"],
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)


def sample_splits(seed: int = 123) -> Dict[str, List[str]]:
    """Deterministic split per class: 3 train, 1 val, 1 test."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for ids in CLASS_BUCKETS.values():
        ids = list(ids)
        rng.shuffle(ids)
        splits["train"].extend(ids[:3])
        splits["val"].append(ids[3])
        splits["test"].append(ids[4])
    return splits
