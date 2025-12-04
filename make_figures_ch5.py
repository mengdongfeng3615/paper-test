import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import OUTPUT_DIR
from src.font_config import configure_chinese_font

# 配置中文字体
configure_chinese_font()

OUTPUT_DIR.mkdir(exist_ok=True)


def fig_5_1_snr_curve():
    df = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
    order = ["clean", "10.0 dB", "8.0 dB", "5.0 dB", "0.5 dB"]
    models = ["baseline", "fusion_mds"]
    colors = {"baseline": "#1f77b4", "fusion_mds": "#d62728"}

    plt.figure(figsize=(6.5, 4))
    for m in models:
        sub = df[df["model"] == m].set_index("snr")
        y = [sub.loc[s, "macro_f1_mean"] for s in order]
        yerr = [sub.loc[s, "macro_f1_std"] for s in order]
        x = np.arange(len(order))
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=m,
            marker="o",
            capsize=3,
            color=colors[m],
        )
    plt.xticks(np.arange(len(order)), order)
    plt.ylim(0, 1.05)
    plt.xlabel("SNR")
    plt.ylabel("Macro-F1")
    plt.title("Fig. 5-1 Macro-F1 vs SNR (Baseline vs Fusion+MDS)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_5-1_snr_curve.png", dpi=300)
    plt.close()


def fig_5_2_ablation_0_5db():
    df = pd.read_csv(OUTPUT_DIR / "ablation_mds_iso_summary_fast.csv")
    sub = df[(df["classifier"] == "svm_rbf") & (df["snr"] == "0.5 dB")]
    order = ["base", "base_cepstra", "fusion_mds", "fusion_iso"]
    labels = ["Base", "Base+Cepstra", "Fusion+MDS", "Fusion+Isomap"]
    vals = [sub[sub["feature_set"] == k]["macro_f1_mean"].values[0] for k in order]

    x = np.arange(len(order))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, vals, color=["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"])
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Macro-F1")
    plt.title("Fig. 5-2 Ablation at 0.5 dB (SVM)")
    for bx, v in zip(bars, vals):
        plt.text(bx.get_x() + bx.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_5-2_ablation_0_5dB.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    fig_5_1_snr_curve()
    fig_5_2_ablation_0_5db()
    print("Chapter 5 figures written to", OUTPUT_DIR)