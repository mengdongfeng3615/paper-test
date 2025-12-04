"""
Run pipeline variant: train only on clean segments, evaluate across SNR levels.
Outputs are written with `*_cleantrain.*` suffixes to avoid clobbering default runs.
"""
import json
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, DatasetConfig, OUTPUT_DIR, ensure_output_dirs
from src.data_utils import Segment, load_segments, split_by_sample
from src.mds_utils import SAMPLE_ORDER, load_subjective_matrix
from src.modeling import evaluate_model, train_model
from src.plots import plot_confusion, plot_mds

N_ROUNDS = 5
TRAIN_NOISE = (float("inf"),)  # clean-only training


def sample_label(sample_id: str) -> int:
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return 0
    if idx <= 10:
        return 1
    return 2


def run_round(round_idx: int, cfg: DatasetConfig, segments, mds_map):
    seed = cfg.random_seed + round_idx
    splits = split_by_sample(cfg, seed=seed)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])

    train_segments = [seg for seg in segments if seg.sample_id in train_ids]
    test_segments = [seg for seg in segments if seg.sample_id in test_ids]

    metrics_rows = []
    for include_mds, name in [(False, "baseline"), (True, "fusion_mds")]:
        model, params = train_model(
            train_segments,
            cfg,
            mds_map,
            include_mds,
            train_noise_levels=TRAIN_NOISE,
        )
        eval_results = evaluate_model(model, test_segments, cfg, mds_map, include_mds)
        for item in eval_results:
            metrics_rows.append(
                {
                    "round": round_idx,
                    "model": name,
                    "snr": "clean" if np.isinf(item["snr"]) else f"{item['snr']} dB",
                    "accuracy": item["accuracy"],
                    "macro_f1": item["macro_f1"],
                }
            )
        # Confusion matrix on clean for quick inspection
        clean_res = next(r for r in eval_results if np.isinf(r["snr"]))
        plot_confusion(
            clean_res["confusion"],
            OUTPUT_DIR / f"confusion_cleantrain_{name}_round{round_idx}.png",
            title=f"{name} clean-train (clean, round {round_idx})",
        )
    return metrics_rows


def main() -> None:
    cfg = DatasetConfig()
    ensure_output_dirs()

    # only MDS coordinates; Isomap disabled by default here too
    diss, mds_coords, _ = load_subjective_matrix(with_isomap=False)
    sample_labels = [sample_label(sid) for sid in SAMPLE_ORDER]
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    pd.DataFrame(
        {
            "sample": SAMPLE_ORDER,
            "class": sample_labels,
            "mds1": mds_coords[:, 0],
            "mds2": mds_coords[:, 1],
        }
    ).to_csv(OUTPUT_DIR / "mds_coordinates_cleantrain.csv", index=False)
    plot_mds(mds_coords, sample_labels, OUTPUT_DIR / "fig_mds_cleantrain.png")

    segments = load_segments(cfg)

    all_rows = []
    for r in range(N_ROUNDS):
        all_rows.extend(run_round(r, cfg, segments, mds_map))

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_rounds_cleantrain.csv", index=False)

    summary = metrics_df.groupby(["model", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "metrics_summary_cleantrain.csv", index=False)

    with open(OUTPUT_DIR / "summary_cleantrain.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    print("Wrote clean-train outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
