"""
Entry point to run preprocessing, feature extraction, training, and evaluation.
"""
import json
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, DatasetConfig, OUTPUT_DIR, ensure_output_dirs
from src.data_utils import Segment, load_segments, split_by_sample
from src.mds_utils import SAMPLE_ORDER, load_subjective_matrix
from src.modeling import evaluate_model, train_model
from src.plots import plot_confusion, plot_mds, plot_snr_curve


def sample_label(sample_id: str) -> int:
    """Convert Sample id to coarse label index."""
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return 0
    if idx <= 10:
        return 1
    return 2


def main() -> None:
    """Execute full pipeline and write metrics/plots to outputs directory."""
    cfg = DatasetConfig()
    ensure_output_dirs()

    diss, mds_coords, iso_coords = load_subjective_matrix()
    sample_labels = [sample_label(sid) for sid in SAMPLE_ORDER]
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    pd.DataFrame(
        {
            "sample": SAMPLE_ORDER,
            "class": sample_labels,
            "mds1": mds_coords[:, 0],
            "mds2": mds_coords[:, 1],
        }
    ).to_csv(OUTPUT_DIR / "mds_coordinates.csv", index=False)
    plot_mds(mds_coords, sample_labels, OUTPUT_DIR / "fig_mds.png")

    segments = load_segments(cfg)
    splits = split_by_sample(cfg)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])

    train_segments = [seg for seg in segments if seg.sample_id in train_ids]
    test_segments = [seg for seg in segments if seg.sample_id in test_ids]

    metrics_rows = []
    summaries = {}

    for include_mds, name in [(False, "baseline"), (True, "fusion_mds")]:
        model, params = train_model(train_segments, cfg, mds_map, include_mds)
        eval_results = evaluate_model(model, test_segments, cfg, mds_map, include_mds)
        summaries[name] = {
            "best_params": params,
            "clean_macro_f1": next(r for r in eval_results if np.isinf(r["snr"]))[
                "macro_f1"
            ],
        }
        for item in eval_results:
            metrics_rows.append(
                {
                    "model": name,
                    "snr": "clean" if np.isinf(item["snr"]) else f"{item['snr']} dB",
                    "accuracy": item["accuracy"],
                    "macro_f1": item["macro_f1"],
                }
            )
        clean_res = next(r for r in eval_results if np.isinf(r["snr"]))
        plot_confusion(
            clean_res["confusion"],
            OUTPUT_DIR / f"confusion_{name}.png",
            title=f"{name} (clean)",
        )
        if include_mds:
            plot_snr_curve(eval_results, OUTPUT_DIR / "snr_curve_fusion.png")

    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("Wrote outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
