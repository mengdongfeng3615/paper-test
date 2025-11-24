"""
Robust pipeline: augment noisy audio, map to 4D MDS anchors with SVR, classify with SVM.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure package imports when run as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from robust_pipeline.config import (
    RobustConfig,
    SAMPLE_TO_FILE,
    CLASS_MAP,
    CLASS_NAMES,
    ensure_dirs,
    RAW_DIR,
    OUTPUT_DIR,
    sample_splits,
)
from robust_pipeline.audio_utils import load_audio, preprocess, augment_signal, segment_signal
from robust_pipeline.features import extract_features
from robust_pipeline.mds_utils import load_mds_4d, SAMPLE_ORDER
from robust_pipeline.modeling import evaluate_sets, baseline_svm


def build_split_dataset(cfg: RobustConfig, rng: np.random.Generator):
    """Return dict split->{records} with isolated sample IDs and SNRs."""
    splits = sample_splits(cfg.random_seed)
    _, coords = load_mds_4d()
    coord_map = {sid: coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    split_cfg = {
        "train": (cfg.snr_train, cfg.aug_train),
        "val": (cfg.snr_val, cfg.aug_val),
        "test": (cfg.snr_test, cfg.aug_test),
    }

    split_records = {"train": [], "val": [], "test": []}
    for split_name, ids in splits.items():
        snr_levels, aug_num = split_cfg[split_name]
        for sample_id in ids:
            fname = SAMPLE_TO_FILE[sample_id]
            path = RAW_DIR / fname
            raw = load_audio(str(path), cfg)
            proc = preprocess(raw, cfg)
            anchor = coord_map[sample_id]
            label = CLASS_MAP[sample_id]
            for seg in segment_signal(proc, cfg):
                for snr, noisy in augment_signal(seg, snr_levels, aug_num, rng):
                    feats = extract_features(noisy, cfg.sample_rate, cfg)
                    split_records[split_name].append(
                        {
                            "sample": sample_id,
                            "label": label,
                            "snr": snr,
                            "features": feats,
                            "anchor": anchor,
                        }
                    )
    return split_records


def save_confusion(cm: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([CLASS_NAMES[i] for i in [0, 1, 2]])
    ax.set_yticklabels([CLASS_NAMES[i] for i in [0, 1, 2]])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def records_to_arrays(records):
    X = np.stack([r["features"] for r in records], axis=0)
    Y = np.stack([r["anchor"] for r in records], axis=0)
    labels = np.array([r["label"] for r in records])
    return X, Y, labels


def main():
    cfg = RobustConfig()
    ensure_dirs()
    rng = np.random.default_rng(cfg.random_seed)

    split_records = build_split_dataset(cfg, rng)
    train = records_to_arrays(split_records["train"])
    val = records_to_arrays(split_records["val"])
    test = records_to_arrays(split_records["test"])

    svr_result = evaluate_sets(train, val, test)
    base_result = baseline_svm(train, val, test)

    metrics = {
        "svr_svm": svr_result.metrics,
        "baseline_svm": base_result.metrics,
    }

    with open(OUTPUT_DIR / "robust_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([
        {"model": model, "split": split, **vals}
        for model, splits in metrics.items()
        for split, vals in splits.items()
    ]).to_csv(OUTPUT_DIR / "robust_metrics.csv", index=False)

    save_confusion(svr_result.confusion, OUTPUT_DIR / "robust_confusion_svr.png")
    save_confusion(base_result.confusion, OUTPUT_DIR / "robust_confusion_baseline.png")

    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
