"""Ablation: compare MDS vs Isomap fusion with lighter search (fast run)."""
import json
from pathlib import Path
from typing import Dict, List

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
from src.features import _basic_features, _mfcc_features, _pncc_features, _rasta_features

N_ROUNDS = 1
TRAIN_NOISE = (float("inf"), 10.0, 8.0, 5.0)
EVAL_NOISE = (float("inf"), 10.0, 8.0, 5.0, 0.5)


def build_feature(seg_data: np.ndarray, cfg: DatasetConfig, coord, mode: str) -> np.ndarray:
    parts: List[np.ndarray] = []
    if mode in {"base", "base_cepstra", "fusion_mds", "fusion_iso"}:
        parts.append(np.array(_basic_features(seg_data, cfg.sample_rate)))
    if mode in {"base_cepstra", "fusion_mds", "fusion_iso"}:
        parts.append(_mfcc_features(seg_data, cfg.sample_rate, cfg))
        parts.append(_pncc_features(seg_data, cfg.sample_rate, cfg))
        parts.append(_rasta_features(seg_data, cfg.sample_rate, cfg))
    if mode in {"fusion_mds", "fusion_iso"} and coord is not None:
        parts.append(coord)
    return np.concatenate(parts).astype(np.float32)


def feature_matrix(segments, cfg, coord_map: Dict[str, np.ndarray], mode: str, noise_levels, seed_offset: int):
    feats = []
    labels = []
    groups = []
    for snr in noise_levels:
        seed = cfg.random_seed + seed_offset if np.isinf(snr) else cfg.random_seed + seed_offset + int(round(snr * 10))
        rng = np.random.default_rng(seed)
        for seg in segments:
            noisy = add_noise(seg.data, snr, rng)
            coord = coord_map.get(seg.sample_id)
            vec = build_feature(noisy, cfg, coord, mode)
            feats.append(vec)
            labels.append(seg.label)
            groups.append(seg.sample_id)
    return np.vstack(feats), np.array(labels), np.array(groups)


def train_classifier(train_segments, cfg, coord_map, mode: str, clf_name: str):
    X_train, y_train, groups = feature_matrix(train_segments, cfg, coord_map, mode, TRAIN_NOISE, seed_offset=0)

    if clf_name == "svm_rbf":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False)),
        ])
        param_grid = {
            "clf__C": [1.0],
            "clf__gamma": ["scale"],
        }
    elif clf_name == "rf":
        pipe = Pipeline([
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=cfg.random_seed)),
        ])
        param_grid = {
            "clf__n_estimators": [200],
            "clf__max_depth": [None],
            "clf__min_samples_leaf": [1],
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


def eval_classifier(model, test_segments, cfg, coord_map, mode: str):
    results = []
    for snr in EVAL_NOISE:
        X_test, y_test, _ = feature_matrix(test_segments, cfg, coord_map, mode, (snr,), seed_offset=100)
        y_pred = model.predict(X_test)
        results.append({
            "snr": snr,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
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

    _, mds_coords, iso_coords = load_subjective_matrix()
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}
    iso_map = {sid: iso_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    all_segments = load_segments(cfg)

    modes = [
        ("base", {}),
        ("base_cepstra", {}),
        ("fusion_mds", mds_map),
        ("fusion_iso", iso_map),
    ]
    clfs = ["svm_rbf", "rf"]

    rows = []
    for round_idx in range(N_ROUNDS):
        train_segments, test_segments = split_segments(all_segments, cfg, seed=cfg.random_seed + round_idx)
        for mode, cmap in modes:
            coord_map = cmap if mode in {"fusion_mds", "fusion_iso"} else {}
            for clf_name in clfs:
                model, params = train_classifier(train_segments, cfg, coord_map, mode, clf_name)
                evals = eval_classifier(model, test_segments, cfg, coord_map, mode)
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
    df.to_csv(OUTPUT_DIR / "ablation_mds_iso_metrics_fast.csv", index=False)

    summary = df.groupby(["feature_set", "classifier", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "ablation_mds_iso_summary_fast.csv", index=False)

    print("Fast MDS/Isomap ablation done. Results saved to outputs/ablation_mds_iso_*fast.csv")


if __name__ == "__main__":
    main()
