"""
Data loading helpers: map sample ids to labels, preprocess audio, and split by sample.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import audio_utils
from .config import DatasetConfig, RAW_DIR, SAMPLE_TO_FILE, LABELS
from .mds_utils import SAMPLE_ORDER


@dataclass
class Segment:
    """A single fixed-length audio segment with label and provenance."""
    sample_id: str
    label: int
    data: np.ndarray


def _label_from_sample(sample_id: str) -> int:
    """Derive class label from sample naming convention."""
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return LABELS["non_penetration"]
    if idx <= 10:
        return LABELS["full_penetration"]
    return LABELS["excessive_penetration"]


def load_segments(cfg: DatasetConfig) -> List[Segment]:
    """Load, preprocess, window, and optionally subsample segments per sample."""
    segments: List[Segment] = []
    rng = np.random.default_rng(cfg.random_seed)
    for sample_id, fname in SAMPLE_TO_FILE.items():
        label = _label_from_sample(sample_id)
        path = RAW_DIR / fname
        raw = audio_utils.load_audio(str(path), cfg)
        proc = audio_utils.preprocess_signal(raw, cfg)
        segs = audio_utils.segment_signal(proc, cfg)
        if cfg.max_segments_per_sample and len(segs) > cfg.max_segments_per_sample:
            idx = rng.choice(len(segs), size=cfg.max_segments_per_sample, replace=False)
            segs = [segs[i] for i in idx]
        for seg in segs:
            segments.append(Segment(sample_id=sample_id, label=label, data=seg))
    return segments


def split_by_sample(cfg: DatasetConfig, seed: int) -> Dict[str, Sequence[str]]:
    """Random 3/1/1 per class split, varies by seed."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    buckets: Dict[int, List[str]] = {
        0: SAMPLE_ORDER[0:5],
        1: SAMPLE_ORDER[5:10],
        2: SAMPLE_ORDER[10:15],
    }
    for sample_ids in buckets.values():
        ids = list(sample_ids)
        rng.shuffle(ids)
        splits["train"].extend(ids[:3])
        splits["val"].append(ids[3])
        splits["test"].append(ids[4])
    return splits
