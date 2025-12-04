"""
Multi-noise training (white + pink + factory) with predictive MDS fusion.
Models:
- baseline: physical features only, trained with multi-noise augmentation
- fusion_pred_mds: physical features + predicted MDS (regressor), with weighting

Training noise: clean + 10/8/5/0.5 dB, noise types white/pink/factory
Testing noise: same SNR grid, per noise type

Outputs:
  outputs/metrics_rounds_multinoise_predictive.csv
  outputs/metrics_summary_multinoise_predictive.csv
  outputs/summary_multinoise_predictive.json
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.audio_utils import add_noise
from src.config import DatasetConfig, OUTPUT_DIR, RAW_DIR, ensure_output_dirs
from src.data_utils import Segment, load_segments, split_by_sample
from src.features import build_feature_vector
from src.mds_utils import SAMPLE_ORDER, load_subjective_matrix

# To keep runtime reasonable; adjust higher if needed
N_ROUNDS = 3
NOISE_LEVELS_TRAIN = (float("inf"), 10.0, 8.0, 5.0, 0.5)
NOISE_LEVELS_TEST = (float("inf"), 10.0, 8.0, 5.0, 0.5)
NOISE_TYPES = ("white", "pink", "factory")
MDS_WEIGHT_ALPHA = 5.0  # emphasize perceptual feature contribution (tunable)
FACTORY_DIR = RAW_DIR / "factory_audio"


def sample_label(sample_id: str) -> int:
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return 0
    if idx <= 10:
        return 1
    return 2


def load_factory_noises(cfg: DatasetConfig) -> List[np.ndarray]:
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
            if np.isinf(snr_db):
                seed = cfg.random_seed + seed_offset + hash(ntype) % 997
            else:
                seed = cfg.random_seed + seed_offset + int(round(snr_db * 10)) + hash(ntype) % 997
            rng = np.random.default_rng(seed)
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


def _feature_matrix_clean_only(
    segments: Sequence[Segment],
    cfg: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean (no-noise) features for training the perception regressor."""
    feats: List[np.ndarray] = []
    groups: List[str] = []
    for seg in segments:
        vec = build_feature_vector(seg.data, cfg.sample_rate, cfg, mds_coord=None)
        feats.append(vec)
        groups.append(seg.sample_id)
    return np.vstack(feats), np.array(groups)


def _train_classifier(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )
    param_grid = {
        "svm__C": [0.5, 1.0, 2.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }
    n_splits = max(2, min(3, len(np.unique(groups))))
    cv = GroupKFold(n_splits=n_splits)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def _train_perception_regressor(X: np.ndarray, groups: np.ndarray, mds_map: Dict[str, np.ndarray]):
    y_mds = np.stack([mds_map[g] for g in groups], axis=0)
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=5.0, gamma=0.05, epsilon=0.05)),
        ]
    )
    reg = MultiOutputRegressor(base)
    reg.fit(X, y_mds)
    return reg


def _evaluate(
    model: Pipeline,
    regressor,
    segments: Sequence[Segment],
    cfg: DatasetConfig,
    factory_noises: Sequence[np.ndarray],
    include_pred_mds: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ntype in NOISE_TYPES:
        for snr_db in NOISE_LEVELS_TEST:
            if np.isinf(snr_db):
                seed = cfg.random_seed + 100 + hash(ntype) % 997
            else:
                seed = cfg.random_seed + 100 + int(round(snr_db * 10)) + hash(ntype) % 997
            rng = np.random.default_rng(seed)
            feats: List[np.ndarray] = []
            labels: List[int] = []
            for seg in segments:
                noisy = add_noise(
                    seg.data,
                    snr_db,
                    rng,
                    noise_type=ntype,
                    factory_noises=factory_noises if ntype == "factory" else None,
                )
                base = build_feature_vector(noisy, cfg.sample_rate, cfg, mds_coord=None)
                if include_pred_mds:
                    pred = regressor.predict(base.reshape(1, -1))
                    base = np.concatenate([base, MDS_WEIGHT_ALPHA * pred.ravel()])
                feats.append(base)
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


def run_round(round_idx: int, cfg: DatasetConfig, segments: Sequence[Segment], mds_map: Dict[str, np.ndarray], factory_noises):
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
        seed_offset=0,
        factory_noises=factory_noises,
    )

    # baseline
    clf_baseline = _train_classifier(X_train, y_train, groups)

    # predictive MDS fusion
    # 回归器只用 clean 段训练，以稳定感知坐标预测
    X_reg, groups_reg = _feature_matrix_clean_only(train_segments, cfg)
    reg = _train_perception_regressor(X_reg, groups_reg, mds_map)
    mds_pred_train = reg.predict(X_train)
    X_fused = np.hstack([X_train, MDS_WEIGHT_ALPHA * mds_pred_train])
    clf_fusion = _train_classifier(X_fused, y_train, groups)

    rows = []
    rows.extend(
        [
            {"model": "baseline", **r}
            for r in _evaluate(clf_baseline, reg, test_segments, cfg, factory_noises, include_pred_mds=False)
        ]
    )
    rows.extend(
        [
            {"model": "fusion_pred_mds", **r}
            for r in _evaluate(clf_fusion, reg, test_segments, cfg, factory_noises, include_pred_mds=True)
        ]
    )
    for r in rows:
        r["round"] = round_idx
    return rows


def main():
    cfg = DatasetConfig()
    ensure_output_dirs()

    # subjective MDS anchors
    diss, mds_coords, _ = load_subjective_matrix(with_isomap=False)
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    factory_noises = load_factory_noises(cfg)
    segments = load_segments(cfg)

    all_rows: List[Dict[str, object]] = []
    for r in range(N_ROUNDS):
        all_rows.extend(run_round(r, cfg, segments, mds_map, factory_noises))

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_rounds_multinoise_predictive.csv", index=False)

    summary = metrics_df.groupby(["model", "noise_type", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "metrics_summary_multinoise_predictive.csv", index=False)
    with open(OUTPUT_DIR / "summary_multinoise_predictive.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    print("Wrote multi-noise predictive MDS outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
