"""
Model training and evaluation utilities for SVM classifiers with noise augmentation.
"""
import numpy as np
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .audio_utils import add_noise
from .config import DatasetConfig
from .data_utils import Segment
from .features import build_feature_vector


def _feature_matrix(
    segments: Sequence[Segment],
    cfg: DatasetConfig,
    mds_map: Dict[str, np.ndarray],
    include_mds: bool,
    noise_levels: Iterable[float],
    seed_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[str] = []
    for snr_db in noise_levels:
        if np.isinf(snr_db):
            seed = cfg.random_seed + seed_offset
        else:
            seed = cfg.random_seed + seed_offset + int(round(snr_db * 10))
        rng = np.random.default_rng(seed)
        for seg in segments:
            noisy = add_noise(seg.data, snr_db, rng)
            mds = mds_map.get(seg.sample_id) if include_mds else None
            vec = build_feature_vector(noisy, cfg.sample_rate, cfg, mds)
            feats.append(vec)
            labels.append(seg.label)
            groups.append(seg.sample_id)
    return np.vstack(feats), np.array(labels), np.array(groups)


def train_model(
    train_segments: Sequence[Segment],
    cfg: DatasetConfig,
    mds_map: Dict[str, np.ndarray],
    include_mds: bool,
) -> Tuple[Pipeline, Dict[str, float]]:
    # training使用多档噪声增强（clean + 10/8/5 dB）
    train_noise = (float("inf"), 10.0, 8.0, 5.0)
    X_train, y_train, groups = _feature_matrix(
        train_segments,
        cfg,
        mds_map,
        include_mds,
        noise_levels=train_noise,
        seed_offset=0,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(kernel="rbf", class_weight="balanced", probability=False),
            ),
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
    search.fit(X_train, y_train, groups=groups)
    best_params = {k: search.best_params_[k] for k in search.best_params_}
    return search.best_estimator_, best_params


def evaluate_model(
    model: Pipeline,
    test_segments: Sequence[Segment],
    cfg: DatasetConfig,
    mds_map: Dict[str, np.ndarray],
    include_mds: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for snr_db in cfg.snr_eval_levels:
        X_test, y_test, _ = _feature_matrix(
            test_segments,
            cfg,
            mds_map,
            include_mds,
            noise_levels=(snr_db,),
            seed_offset=100,
        )
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        results.append(
            {
                "snr": snr_db,
                "accuracy": acc,
                "macro_f1": f1,
                "confusion": cm,
                "y_true": y_test,
                "y_pred": preds,
            }
        )
    return results
