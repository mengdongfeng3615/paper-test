"""Ablation experiments on feature sets and classifiers.

Feature sets:
- base: traditional time/frequency stats
- base_cepstra: base + MFCC/PNCC/RASTA-PLP
- fusion_mds: base + MFCC/PNCC/RASTA-PLP + 2D MDS coords

Classifiers:
- svm_rbf (same as mainline)
- rf (random forest)

Training noise: clean + 10/8/5 dB
Eval noise: clean, 10, 8, 5, 0.5 dB
Splits: per-class 3/1/1 sample split, multiple rounds (default 2)
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DatasetConfig, OUTPUT_DIR
from src.data_utils import load_segments, split_by_sample
from src.audio_utils import add_noise
from src.mds_utils import load_subjective_matrix, SAMPLE_ORDER
from src.features import (
    _basic_features,
    _mfcc_features,
    _pncc_features,
    _rasta_features,
)

# experiment settings
N_ROUNDS = 2
TRAIN_NOISE = (float("inf"), 10.0, 8.0, 5.0)
EVAL_NOISE = (float("inf"), 10.0, 8.0, 5.0, 0.5)


def build_feature(seg_data: np.ndarray, cfg: DatasetConfig, mds_coord, mode: str) -> np.ndarray:
    """Construct feature vector according to mode."""
    parts: List[np.ndarray] = []
    if mode in {"base", "base_cepstra", "fusion_mds"}:
        parts.append(np.array(_basic_features(seg_data, cfg.sample_rate)))
    if mode in {"base_cepstra", "fusion_mds"}:
        parts.append(_mfcc_features(seg_data, cfg.sample_rate, cfg))
        parts.append(_pncc_features(seg_data, cfg.sample_rate, cfg))
        parts.append(_rasta_features(seg_data, cfg.sample_rate, cfg))
    if mode == "fusion_mds" and mds_coord is not None:
        parts.append(mds_coord)
    return np.concatenate(parts).astype(np.float32)


def feature_matrix(segments, cfg, mds_map: Dict[str, np.ndarray], mode: str, noise_levels, seed_offset: int):
    feats = []
    labels = []
    groups = []
    for snr in noise_levels:
        seed = cfg.random_seed + seed_offset if np.isinf(snr) else cfg.random_seed + seed_offset + int(round(snr * 10))
        rng = np.random.default_rng(seed)
        for seg in segments:
            noisy = add_noise(seg.data, snr, rng)
            mds = mds_map.get(seg.sample_id)
            vec = build_feature(noisy, cfg, mds, mode)
            feats.append(vec)
            labels.append(seg.label)
            groups.append(seg.sample_id)
    return np.vstack(feats), np.array(labels), np.array(groups)


def train_classifier(train_segments, cfg, mds_map, mode: str, clf_name: str):
    X_train, y_train, groups = feature_matrix(train_segments, cfg, mds_map, mode, TRAIN_NOISE, seed_offset=0)

    if clf_name == "svm_rbf":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False)),
        ])
        param_grid = {
            "clf__C": [0.5, 1.0, 2.0],
            "clf__gamma": ["scale", 0.01, 0.001],
        }
    elif clf_name == "rf":
        pipe = Pipeline([
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=cfg.random_seed)),
        ])
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 20],
            "clf__min_samples_leaf": [1, 2],
        }
    else:
        raise ValueError(f"Unknown classifier {clf_name}")

    cv = GroupKFold(n_splits=max(2, min(3, len(np.unique(groups)))))
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_


def eval_classifier(model, test_segments, cfg, mds_map, mode: str):
    results = []
    for snr in EVAL_NOISE:
        X_test, y_test, _ = feature_matrix(test_segments, cfg, mds_map, mode, (snr,), seed_offset=100)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        results.append({
            "snr": snr,
            "accuracy": acc,
            "macro_f1": f1,
            "confusion": cm,
        })
    return results


def split_segments(all_segments, cfg: DatasetConfig, seed: int):
    splits = split_by_sample(cfg, seed=seed)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])
    train_segments = [s for s in all_segments if s.sample_id in train_ids]
    test_segments = [s for s in all_segments if s.sample_id in test_ids]
    return train_segments, test_segments


def main():
    cfg = DatasetConfig()
    OUTPUT_DIR.mkdir(exist_ok=True)

    _, mds_coords, _ = load_subjective_matrix()
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    all_segments = load_segments(cfg)

    modes = ["base", "base_cepstra", "fusion_mds"]
    clfs = ["svm_rbf", "rf"]

    rows = []
    for round_idx in range(N_ROUNDS):
        train_segments, test_segments = split_segments(all_segments, cfg, seed=cfg.random_seed + round_idx)
        for mode in modes:
            for clf_name in clfs:
                model, params = train_classifier(train_segments, cfg, mds_map, mode, clf_name)
                evals = eval_classifier(model, test_segments, cfg, mds_map, mode)
                for ev in evals:
                    rows.append({
                        "round": round_idx,
                        "feature_set": mode,
                        "classifier": clf_name,
                        "snr": "clean" if np.isinf(ev["snr"]) else f"{ev['snr']} dB",
                        "accuracy": ev["accuracy"],
                        "macro_f1": ev["macro_f1"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)

    summary = df.groupby(["feature_set", "classifier", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "ablation_metrics_summary.csv", index=False)

    print("Ablation done. Results saved to outputs/ablation_metrics*.csv")


if __name__ == "__main__":
    main()
