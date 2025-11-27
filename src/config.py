import pathlib
from dataclasses import dataclass
from typing import Dict, List

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass(frozen=True)
class DatasetConfig:
    sample_rate: int = 40_000
    segment_seconds: float = 0.01  # 10 ms window
    hop_seconds: float = 0.01      # 10 ms hop
    max_segments_per_sample: int = 30  # cap to limit compute
    bandpass_low: float = 300.0
    bandpass_high: float = 18_000.0
    notch_freq: float = 50.0
    pre_emphasis: float = 0.97
    n_mfcc: int = 20
    n_pncc: int = 20
    n_rasta: int = 13
    random_seed: int = 42
    snr_eval_levels: List[float] = (float("inf"), 10.0, 8.0, 5.0, 0.5)


LABELS: Dict[str, int] = {
    "non_penetration": 0,
    "full_penetration": 1,
    "excessive_penetration": 2,
}

CLASS_NAMES = {
    0: "Non",
    1: "Full",
    2: "Excessive",
}


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


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
