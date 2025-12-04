"""Window-length sensitivity experiment: compare 10/20/25 ms for Fusion+MDS and Baseline."""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import DatasetConfig, OUTPUT_DIR
from src.data_utils import load_segments, split_by_sample
from src.mds_utils import load_subjective_matrix, SAMPLE_ORDER
from src.modeling import train_model, evaluate_model

OUTPUT_DIR.mkdir(exist_ok=True)

WINDOWS = [0.01, 0.02, 0.025]  # 10 ms, 20 ms, 25 ms
N_ROUNDS = 1  # 小实验，1 轮即可看趋势


def sample_label(sample_id: str) -> int:
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return 0
    if idx <= 10:
        return 1
    return 2


def main() -> None:
    avg, mds_coords, _ = load_subjective_matrix(with_isomap=False)
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}

    rows: List[Dict[str, object]] = []

    for win in WINDOWS:
        cfg = DatasetConfig(
            segment_seconds=win,
            hop_seconds=win,
            max_segments_per_sample=50,
        )
        all_segments = load_segments(cfg)

        for round_idx in range(N_ROUNDS):
            splits = split_by_sample(cfg, seed=cfg.random_seed + round_idx)
            train_ids = set(splits["train"] + splits["val"])
            test_ids = set(splits["test"])
            train_segments = [s for s in all_segments if s.sample_id in train_ids]
            test_segments = [s for s in all_segments if s.sample_id in test_ids]

            for include_mds, name in [(False, "baseline"), (True, "fusion_mds")]:
                model, params = train_model(train_segments, cfg, mds_map, include_mds)
                evals = evaluate_model(model, test_segments, cfg, mds_map, include_mds)
                for ev in evals:
                    rows.append(
                        {
                            "window_ms": int(win * 1000),
                            "round": round_idx,
                            "model": name,
                            "snr": "clean" if np.isinf(ev["snr"]) else f"{ev['snr']} dB",
                            "accuracy": ev["accuracy"],
                            "macro_f1": ev["macro_f1"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "window_metrics.csv", index=False)

    summary = df.groupby(["window_ms", "model", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "window_metrics_summary.csv", index=False)

    print("Window-length experiment done. Results in outputs/window_metrics*.csv")


if __name__ == "__main__":
    main()
