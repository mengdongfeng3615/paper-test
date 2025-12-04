"""
Plotting helpers for MDS scatter, confusion matrices, and SNR curves.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Sequence

from .config import CLASS_NAMES
from .font_config import configure_chinese_font

# 配置中文字体
configure_chinese_font()

COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}


def plot_mds(coords: np.ndarray, labels: Sequence[int], path: Path) -> None:
    """Scatter plot of 2D MDS coordinates with class colors and sample ids."""
    plt.figure(figsize=(6, 5))
    labels_arr = np.array(labels)
    for cls, name in CLASS_NAMES.items():
        mask = labels_arr == cls
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=name,
            s=60,
            alpha=0.85,
            color=COLORS.get(cls, "gray"),
        )
    for idx, (x, y) in enumerate(coords):
        plt.text(x + 0.01, y + 0.01, str(idx + 1), fontsize=8, alpha=0.8)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.title("Subjective MDS embedding")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(cm: np.ndarray, path: Path, title: str = "Confusion") -> None:
    """Visualize confusion matrix with integer counts."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_snr_curve(results: List[Dict[str, float]], path: Path) -> None:
    """Plot macro-F1 across SNR test conditions."""
    snrs = []
    f1s = []
    for item in results:
        snr = item["snr"]
        label = "clean" if np.isinf(snr) else f"{snr} dB"
        snrs.append(label)
        f1s.append(item["macro_f1"])
    plt.figure(figsize=(6, 3.5))
    plt.plot(snrs, f1s, marker="o")
    plt.xlabel("SNR")
    plt.ylabel("Macro-F1")
    plt.title("Noise robustness")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()