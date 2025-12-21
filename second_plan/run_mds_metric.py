"""
Perceptual-metric SVM experiment:
- Use the averaged subjective dissimilarity matrix to build a kernel term.
- Combine that with the acoustic RBF kernel instead of appending 2D MDS coords.
- Evaluate across SNRs to see whether the perceptual prior stabilizes decisions under noise.

Outputs are written to outputs/second_plan/.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.audio_utils import add_noise
from src.config import DatasetConfig, OUTPUT_DIR, ensure_output_dirs
from src.data_utils import Segment, load_segments, split_by_sample
from src.features import build_feature_vector
from src.mds_utils import SAMPLE_ORDER, load_subjective_matrix

N_ROUNDS = 1
NOISE_LEVELS_TRAIN = (float("inf"), 10.0, 8.0, 5.0)
NOISE_LEVELS_TEST = (float("inf"), 10.0, 8.0, 5.0, 0.5)
NOISE_TYPE = "white"
RESULTS_DIR = OUTPUT_DIR / "second_plan"

# Small grid to keep runtime reasonable while showing the effect of the perceptual kernel.
C_GRID = (0.5, 1.0, 2.0)
ALPHA_GRID = (0.25, 0.5, 0.75)  # weight on perceptual kernel
GAMMA_FEAT_GRID = (0.01, 0.001)
GAMMA_MDS_GRID = (0.25, 0.5, 1.0)


def _feature_matrix(
    segments: Sequence[Segment],
    cfg: DatasetConfig,
    noise_levels: Iterable[float],
    noise_type: str,
    seed_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []
    for snr_db in noise_levels:
        base_seed = cfg.random_seed + seed_offset
        if np.isfinite(snr_db):
            base_seed += int(round(snr_db * 10))
        rng = np.random.default_rng(base_seed)
        for seg in segments:
            noisy = add_noise(seg.data, snr_db, rng, noise_type=noise_type)
            vec = build_feature_vector(noisy, cfg.sample_rate, cfg, mds_coord=None)
            feats.append(vec)
            labels.append(seg.label)
            sample_ids.append(seg.sample_id)
    return np.vstack(feats), np.array(labels), np.array(sample_ids)


def _mds_kernel(
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    diss_matrix: np.ndarray,
    idx_map: Dict[str, int],
    gamma_mds: float,
) -> np.ndarray:
    dist = np.empty((len(ids_a), len(ids_b)), dtype=float)
    for i, sa in enumerate(ids_a):
        ia = idx_map[sa]
        for j, sb in enumerate(ids_b):
            ib = idx_map[sb]
            dist[i, j] = diss_matrix[ia, ib]
    return np.exp(-gamma_mds * (dist ** 2))


def _fit_metric_svm(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    diss_matrix: np.ndarray,
    idx_map: Dict[str, int],
) -> Tuple[SVC, Dict[str, object]]:
    # GroupKFold keeps all segments from one physical sample in the same fold.
    n_splits = max(2, min(3, len(np.unique(sample_ids))))
    gkf = GroupKFold(n_splits=n_splits)

    best: Dict[str, object] | None = None
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    for alpha in ALPHA_GRID:
        for gamma_feat in GAMMA_FEAT_GRID:
            K_feat = rbf_kernel(X_scaled, X_scaled, gamma=gamma_feat)
            for gamma_mds in GAMMA_MDS_GRID:
                K_mds = _mds_kernel(sample_ids, sample_ids, diss_matrix, idx_map, gamma_mds)
                K_combo = (1.0 - alpha) * K_feat + alpha * K_mds
                for C in C_GRID:
                    scores: List[float] = []
                    for train_idx, val_idx in gkf.split(K_combo, y, groups=sample_ids):
                        clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
                        clf.fit(K_combo[np.ix_(train_idx, train_idx)], y[train_idx])
                        preds = clf.predict(K_combo[np.ix_(val_idx, train_idx)])
                        scores.append(f1_score(y[val_idx], preds, average="macro"))
                    mean_score = float(np.mean(scores))
                    if best is None or mean_score > best["score"]:
                        best = {
                            "score": mean_score,
                            "C": C,
                            "alpha": alpha,
                            "gamma_feat": gamma_feat,
                            "gamma_mds": gamma_mds,
                        }

    assert best is not None
    # Fit final model on full training set with best hyperparameters.
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    K_feat = rbf_kernel(X_scaled, X_scaled, gamma=best["gamma_feat"])
    K_mds = _mds_kernel(sample_ids, sample_ids, diss_matrix, idx_map, best["gamma_mds"])
    K_train = (1.0 - best["alpha"]) * K_feat + best["alpha"] * K_mds

    clf = SVC(kernel="precomputed", C=best["C"], class_weight="balanced")
    clf.fit(K_train, y)

    ctx: Dict[str, object] = {
        "scaler": scaler,
        "X_train_scaled": X_scaled,
        "sample_ids_train": sample_ids,
        "params": best,
        "diss": diss_matrix,
        "idx_map": idx_map,
    }
    return clf, ctx


def _predict_with_kernel(
    clf: SVC,
    ctx: Dict[str, object],
    X_test: np.ndarray,
    sample_ids_test: np.ndarray,
) -> np.ndarray:
    params = ctx["params"]
    scaler: StandardScaler = ctx["scaler"]
    X_train_scaled: np.ndarray = ctx["X_train_scaled"]
    sample_ids_train: np.ndarray = ctx["sample_ids_train"]
    diss: np.ndarray = ctx["diss"]
    idx_map: Dict[str, int] = ctx["idx_map"]

    X_test_scaled = scaler.transform(X_test)
    K_feat = rbf_kernel(X_test_scaled, X_train_scaled, gamma=params["gamma_feat"])
    K_mds = _mds_kernel(sample_ids_test, sample_ids_train, diss, idx_map, params["gamma_mds"])
    K_combo = (1.0 - params["alpha"]) * K_feat + params["alpha"] * K_mds
    return clf.predict(K_combo)


def run_round(
    round_idx: int,
    cfg: DatasetConfig,
    segments: Sequence[Segment],
    diss_matrix: np.ndarray,
    idx_map: Dict[str, int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    seed = cfg.random_seed + round_idx
    splits = split_by_sample(cfg, seed=seed)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])
    train_segments = [s for s in segments if s.sample_id in train_ids]
    test_segments = [s for s in segments if s.sample_id in test_ids]

    X_train, y_train, sample_ids_train = _feature_matrix(
        train_segments,
        cfg,
        noise_levels=NOISE_LEVELS_TRAIN,
        noise_type=NOISE_TYPE,
        seed_offset=round_idx,
    )
    clf, ctx = _fit_metric_svm(X_train, y_train, sample_ids_train, diss_matrix, idx_map)

    rows: List[Dict[str, object]] = []
    for snr_db in NOISE_LEVELS_TEST:
        X_test, y_test, sample_ids_test = _feature_matrix(
            test_segments,
            cfg,
            noise_levels=(snr_db,),
            noise_type=NOISE_TYPE,
            seed_offset=100 + round_idx,
        )
        preds = _predict_with_kernel(clf, ctx, X_test, sample_ids_test)
        rows.append(
            {
                "round": round_idx,
                "model": "mds_metric_kernel",
                "snr": "clean" if np.isinf(snr_db) else f"{snr_db} dB",
                "accuracy": accuracy_score(y_test, preds),
                "macro_f1": f1_score(y_test, preds, average="macro"),
            }
        )
    return rows, ctx["params"]


def main() -> None:
    # Reduce per-sample segments to keep runtime practical.
    cfg = DatasetConfig(max_segments_per_sample=15)
    ensure_output_dirs()
    RESULTS_DIR.mkdir(exist_ok=True)

    diss, _, _ = load_subjective_matrix(with_isomap=False)
    idx_map = {sid: i for i, sid in enumerate(SAMPLE_ORDER)}
    segments = load_segments(cfg)

    all_rows: List[Dict[str, object]] = []
    params_per_round: List[Dict[str, object]] = []
    for r in range(N_ROUNDS):
        rows, params = run_round(r, cfg, segments, diss, idx_map)
        all_rows.extend(rows)
        params["round"] = r
        params_per_round.append(params)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = RESULTS_DIR / "metrics_mds_metric_rounds.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = metrics_df.groupby(["model", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary_path = RESULTS_DIR / "metrics_mds_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    with open(RESULTS_DIR / "best_params_mds_metric.json", "w", encoding="utf-8") as f:
        json.dump(params_per_round, f, indent=2)

    print("Wrote perceptual-metric SVM outputs to", RESULTS_DIR)
    print("Best params per round:", params_per_round)


if __name__ == "__main__":
    main()
