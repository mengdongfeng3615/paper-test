"""
SVR mapping from noisy features to clean 4D MDS anchors, then SVM classification.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

from .config import CLASS_NAMES


@dataclass
class EvalResult:
    metrics: Dict[str, Dict[str, float]]
    confusion: np.ndarray


def train_svr_models(X: np.ndarray, Y: np.ndarray) -> List[Pipeline]:
    models: List[Pipeline] = []
    for dim in range(Y.shape[1]):
        y = Y[:, dim]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ])
        param_grid = {
            "svr__C": [1, 5, 10],
            "svr__gamma": ["scale", 0.01, 0.001],
            "svr__epsilon": [0.01, 0.1],
        }
        search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, y)
        models.append(search.best_estimator_)
    return models


def predict_mds(models: List[Pipeline], X: np.ndarray) -> np.ndarray:
    preds = [m.predict(X) for m in models]
    return np.stack(preds, axis=1)


def train_classifier(X_mds: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    param_grid = {
        "svm__C": [0.5, 1, 2],
        "svm__gamma": ["scale", 0.01, 0.001],
    }
    search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
    search.fit(X_mds, y)
    return search.best_estimator_


def evaluate_sets(train, val, test) -> EvalResult:
    """
    train/val/test are tuples of (X_features, Y_anchor, labels).
    Returns per-split metrics and test confusion.
    """
    X_tr, Y_tr, y_tr = train
    X_va, Y_va, y_va = val
    X_te, Y_te, y_te = test

    svr_models = train_svr_models(X_tr, Y_tr)

    Y_tr_pred = predict_mds(svr_models, X_tr)
    Y_va_pred = predict_mds(svr_models, X_va)
    Y_te_pred = predict_mds(svr_models, X_te)

    svm_model = train_classifier(Y_tr_pred, y_tr)

    def split_metrics(split_name: str, Y_true, Y_pred, labels_true):
        rmse = float(np.sqrt(mean_squared_error(Y_true, Y_pred)))
        y_cls = svm_model.predict(Y_pred)
        acc = accuracy_score(labels_true, y_cls)
        f1 = f1_score(labels_true, y_cls, average="macro")
        return {"rmse_mds": rmse, "acc": acc, "macro_f1": f1}

    metrics = {
        "train": split_metrics("train", Y_tr, Y_tr_pred, y_tr),
        "val": split_metrics("val", Y_va, Y_va_pred, y_va),
        "test": split_metrics("test", Y_te, Y_te_pred, y_te),
    }

    y_test_pred = svm_model.predict(Y_te_pred)
    cm = confusion_matrix(y_te, y_test_pred, labels=[0, 1, 2])
    return EvalResult(metrics=metrics, confusion=cm)


def baseline_svm(train, val, test) -> EvalResult:
    """Direct feature -> class baseline using RBF-SVM."""
    X_tr, _, y_tr = train
    X_va, _, y_va = val
    X_te, _, y_te = test

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    param_grid = {
        "svm__C": [0.5, 1, 2],
        "svm__gamma": ["scale", 0.01, 0.001],
    }
    search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
    search.fit(X_tr, y_tr)
    model = search.best_estimator_

    def split_metrics(X, y):
        y_pred = model.predict(X)
        return {
            "rmse_mds": float("nan"),
            "acc": accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
        }

    metrics = {
        "train": split_metrics(X_tr, y_tr),
        "val": split_metrics(X_va, y_va),
        "test": split_metrics(X_te, y_te),
    }
    cm = confusion_matrix(y_te, model.predict(X_te), labels=[0, 1, 2])
    return EvalResult(metrics=metrics, confusion=cm)
