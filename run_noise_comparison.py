"""
Train with默认白噪声增强（clean+10/8/5 dB），测试对比不同噪声类型（白/粉/工厂噪声）。
输出带 `_noisecompare` 后缀，不影响既有结果。
"""
import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from src.config import DatasetConfig, OUTPUT_DIR, RAW_DIR, ensure_output_dirs
from src.data_utils import load_segments, split_by_sample
from src.mds_utils import SAMPLE_ORDER, load_subjective_matrix
from src.modeling import evaluate_model, train_model
from src.plots import plot_mds

N_ROUNDS = 5
FACTORY_DIR = RAW_DIR / "factory_audio"


def sample_label(sample_id: str) -> int:
    idx = int(sample_id.replace("Sample", ""))
    if idx <= 5:
        return 0
    if idx <= 10:
        return 1
    return 2


def load_factory_noises(cfg: DatasetConfig) -> List[np.ndarray]:
    """Load all wavs under factory_audio, resample if needed, normalize to unit power."""
    noises: List[np.ndarray] = []
    if not FACTORY_DIR.exists():
        return noises
    for wav in sorted(FACTORY_DIR.glob("*.wav")):
        data, sr = sf.read(wav)
        data = data.astype(np.float64)
        if sr != cfg.sample_rate:
            # simple resample to target sr
            num = int(len(data) * cfg.sample_rate / sr)
            data = signal.resample(data, num)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        pow_val = np.mean(data ** 2)
        if pow_val > 0:
            data = data / np.sqrt(pow_val)
        noises.append(data)
    return noises


def run_round(round_idx, cfg, segments, mds_map, factory_noises):
    seed = cfg.random_seed + round_idx
    splits = split_by_sample(cfg, seed=seed)
    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])
    train_segments = [s for s in segments if s.sample_id in train_ids]
    test_segments = [s for s in segments if s.sample_id in test_ids]

    metrics_rows = []
    noise_settings = {
        "white": {"noise_type": "white", "factory": None},
        "pink": {"noise_type": "pink", "factory": None},
        "factory": {"noise_type": "factory", "factory": factory_noises},
    }

    for include_mds, name in [(False, "baseline"), (True, "fusion_mds")]:
        # training仍用默认白噪声增强
        model, _ = train_model(train_segments, cfg, mds_map, include_mds)
        for noise_label, params in noise_settings.items():
            eval_results = evaluate_model(
                model,
                test_segments,
                cfg,
                mds_map,
                include_mds,
                noise_type=params["noise_type"],
                factory_noises=params["factory"],
            )
            for item in eval_results:
                metrics_rows.append(
                    {
                        "round": round_idx,
                        "model": name,
                        "noise_type": noise_label,
                        "snr": "clean" if np.isinf(item["snr"]) else f"{item['snr']} dB",
                        "accuracy": item["accuracy"],
                        "macro_f1": item["macro_f1"],
                    }
                )
    return metrics_rows


def main():
    cfg = DatasetConfig()
    ensure_output_dirs()

    diss, mds_coords, _ = load_subjective_matrix(with_isomap=False)
    sample_labels = [sample_label(sid) for sid in SAMPLE_ORDER]
    mds_map = {sid: mds_coords[i] for i, sid in enumerate(SAMPLE_ORDER)}
    pd.DataFrame(
        {"sample": SAMPLE_ORDER, "class": sample_labels, "mds1": mds_coords[:, 0], "mds2": mds_coords[:, 1]}
    ).to_csv(OUTPUT_DIR / "mds_coordinates_noisecompare.csv", index=False)
    plot_mds(mds_coords, sample_labels, OUTPUT_DIR / "fig_mds_noisecompare.png")

    factory_noises = load_factory_noises(cfg)
    segments = load_segments(cfg)

    all_rows = []
    for r in range(N_ROUNDS):
        all_rows.extend(run_round(r, cfg, segments, mds_map, factory_noises))

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_rounds_noisecompare.csv", index=False)

    summary = metrics_df.groupby(["model", "noise_type", "snr"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "metrics_summary_noisecompare.csv", index=False)

    with open(OUTPUT_DIR / "summary_noisecompare.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    print("Wrote noise comparison outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
