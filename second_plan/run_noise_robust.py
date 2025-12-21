"""
Noise-domain robustness experiment:
- Train with mixed noise types (white/pink/factory) across multiple SNRs.
- Evaluate per noise type/SNR to see how multi-noise augmentation closes the gap highlighted in ChatGPT 5.1.

Outputs are written to outputs/second_plan/.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.audio_utils import add_noise
from src.config import DatasetConfig, OUTPUT_DIR, RAW_DIR, ensure_output_dirs
from src.data_utils import Segment, load_segments, split_by_sample
from src.features import build_feature_vector

# Defaults keep runtime modest while covering the colored/real-noise shift.
# Trimmed rounds and segments to keep wall-clock practical in this verification run.
N_ROUNDS = 1
NOISE_LEVELS_TRAIN = (float("inf"), 10.0, 8.0, 5.0, 2.0, 0.5)
NOISE_LEVELS_TEST = (float("inf"), 10.0, 8.0, 5.0, 2.0, 0.5)
NOISE_TYPES = ("white", "pink", "factory")
RESULTS_DIR = OUTPUT_DIR / "second_plan"
FACTORY_DIR = RAW_DIR / "factory_audio"


def load_factory_noises(cfg: DatasetConfig) -> List[np.ndarray]:
    """Load and normalize factory noise wavs if present."""
    noises: List[np.ndarray] = []
    if not FACTORY_DIR.exists():
        return noises
    for wav in sorted(FACTORY_DIR.glob("*.wav")):
        data, sr = sf.read(wav)
        data = data.astype(np.float64)
        if sr != cfg.sample_rate:
            num = int(len(data) * cfg.sample_rate / sr)
            data = signal.resample(data, num)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        pow_val = np.mean(data ** 2)
        if pow_val > 0:
            data = data / np.sqrt(pow_val)
        noises.append(data)
    return noises


def _feature_matrix(
    segments: Sequence[Segment],
    cfg: DatasetConfig,
    noise_levels: Iterable[float],
    noise_types: Iterable[str],
    seed_offset: int,
    factory_noises: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[str] = []
    for snr_db in noise_levels:
        for ntype in noise_types:
            # Clean only needs one pass; avoid duplicating it across noise types.
            if np.isinf(snr_db) and ntype != "white":
                continue
            base_seed = cfg.random_seed + seed_offset
            if np.isfinite(snr_db):
                base_seed += int(round(snr_db * 10))
            base_seed += hash(ntype) % 997
            rng = np.random.default_rng(base_seed)
            for seg in segments:
                noisy = add_noise(
                    seg.data,
                    snr_db,
                    rng,
                    noise_type=ntype,
                    factory_noises=factory_noises if ntype == "factory" else None,
                )
                vec = build_feature_vector(noisy, cfg.sample_rate, cfg, mds_coord=None)
                feats.append(vec)
                labels.append(seg.label)
                groups.append(seg.sample_id)
    return np.vstack(feats), np.array(labels), np.array(groups)


def _train_classifier(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )
    param_grid = {"svm__C": [0.5, 1.0, 2.0], "svm__gamma": ["scale", 0.01, 0.001]}
    n_splits = max(2, min(3, len(np.unique(groups))))
    cv = GroupKFold(n_splits=n_splits)
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def _evaluate(
    model: Pipeline,
    test_segments: Sequence[Segment],
    cfg: DatasetConfig,
    factory_noises: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ntype in NOISE_TYPES:
        for snr_db in NOISE_LEVELS_TEST:
            base_seed = cfg.random_seed + 100
            if np.isfinite(snr_db):
                base_seed += int(round(snr_db * 10))
            base_seed += hash(ntype) % 997
            rng = np.random.default_rng(base_seed)
            feats: List[np.ndarray] = []
            labels: List[int] = []
            for seg in test_segments:
                noisy = add_noise(
                    seg.data,
                    snr_db,
                    rng,
                    noise_type=ntype,
                    factory_noises=factory_noises if ntype == "factory" else None,
                )
                feats.append(build_feature_vector(noisy, cfg.sample_rate, cfg, mds_coord=None))
                labels.append(seg.label)
            X_test = np.vstack(feats)
            y_test = np.array(labels)
            preds = model.predict(X_test)
            rows.append(
                {
                    "noise_type": ntype,
                    "snr": "clean" if np.isinf(snr_db) else f"{snr_db} dB",
                    "accuracy": accuracy_score(y_test, preds),
                    "macro_f1": f1_score(y_test, preds, average="macro"),
                }
            )
    return rows


def run_round(round_idx: int, cfg: DatasetConfig, segments: Sequence[Segment], factory_noises: Sequence[np.ndarray]) -> List[Dict[str, object]]:
    seed = cfg.random_seed + round_idx
    splits = split_by_sample(cfg, seed=seed)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])
    train_segments = [s for s in segments if s.sample_id in train_ids]
    test_segments = [s for s in segments if s.sample_id in test_ids]

    X_train, y_train, groups = _feature_matrix(
        train_segments,
        cfg,
        noise_levels=NOISE_LEVELS_TRAIN,
        noise_types=NOISE_TYPES,
        seed_offset=round_idx,
        factory_noises=factory_noises,
    )
    clf = _train_classifier(X_train, y_train, groups)

    rows = _evaluate(clf, test_segments, cfg, factory_noises)
    for r in rows:
        r["round"] = round_idx
        r["model"] = "multi_noise_baseline"
    return rows


def main() -> None:
    # Reduce per-sample segments to speed up validation.
    cfg = DatasetConfig(max_segments_per_sample=15)
    ensure_output_dirs()
    RESULTS_DIR.mkdir(exist_ok=True)

    factory_noises = load_factory_noises(cfg)
    segments = load_segments(cfg)

    all_rows: List[Dict[str, object]] = []
    for r in range(N_ROUNDS):
        all_rows.extend(run_round(r, cfg, segments, factory_noises))

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = RESULTS_DIR / "metrics_noise_robust_rounds.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = metrics_df.groupby(["model", "noise_type", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary_path = RESULTS_DIR / "metrics_noise_robust_summary.csv"
    summary.to_csv(summary_path, index=False)

    with open(RESULTS_DIR / "summary_noise_robust.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    print("Wrote noise-robust outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
