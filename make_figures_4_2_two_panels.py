from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from src.config import DatasetConfig, RAW_DIR, OUTPUT_DIR, SAMPLE_TO_FILE


def fig_4_2_spectrum_two_panels() -> None:
    """Two-panel Welch-PSD spectrum comparison.

    Top:   raw vs. band-pass+notch.
    Bottom: band-pass+notch vs. pre-emphasis.
    """
    cfg = DatasetConfig()
    sample_id = "Sample1"
    wav_path = RAW_DIR / SAMPLE_TO_FILE[sample_id]

    raw, sr = sf.read(wav_path)
    raw = raw.astype(np.float64)

    # use middle 0.5 s
    center = len(raw) // 2
    half = int(0.25 * sr)
    seg = raw[center - half : center + half]

    if np.max(np.abs(seg)) > 0:
        norm = seg / np.max(np.abs(seg))
    else:
        norm = seg

    sos = signal.butter(
        4,
        [cfg.bandpass_low, cfg.bandpass_high],
        btype="bandpass",
        fs=cfg.sample_rate,
        output="sos",
    )
    bp = signal.sosfilt(sos, norm)
    b, a = signal.iirnotch(cfg.notch_freq, 30, fs=cfg.sample_rate)
    bp_notch = signal.filtfilt(b, a, bp)
    pre = signal.lfilter([1.0, -cfg.pre_emphasis], [1], bp_notch)

    def welch_db(sig: np.ndarray):
        f, Pxx = signal.welch(sig, fs=sr, window="hann", nperseg=2048, noverlap=1024)
        Pxx_db = 10 * np.log10(Pxx + 1e-15)
        return f, Pxx_db

    f_raw, psd_raw = welch_db(seg)
    _, psd_bp = welch_db(bp_notch)
    _, psd_pre = welch_db(pre)

    mask = (f_raw >= 0) & (f_raw <= 20000)
    f_khz = f_raw[mask] / 1000.0
    psd_raw = psd_raw[mask]
    psd_bp = psd_bp[mask]
    psd_pre = psd_pre[mask]

    peak = max(psd_raw.max(), psd_bp.max(), psd_pre.max())
    y_top = peak + 5
    y_bottom = max(peak - 80, -120)

    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)

    # Top: raw vs filtered
    ax = axes[0]
    ax.plot(f_khz, psd_raw, color="#1f77b4", lw=0.9, label="Raw")
    ax.plot(f_khz, psd_bp, color="#d62728", lw=0.9, label="Band-pass + notch")
    ax.set_xlim(0, 20)
    ax.set_ylim(y_bottom, y_top)
    ax.set_ylabel("Magnitude (dB)", fontsize=11)
    ax.set_title("(a) Raw vs. band-pass + notch", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # Bottom: filtered vs pre-emphasis
    ax = axes[1]
    ax.plot(f_khz, psd_bp, color="#d62728", lw=0.9, label="Band-pass + notch")
    ax.plot(f_khz, psd_pre, color="#4d4d4d", lw=0.9, label="After pre-emphasis")
    ax.set_xlim(0, 20)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("Frequency (kHz)", fontsize=11)
    ax.set_ylabel("Magnitude (dB)", fontsize=11)
    ax.set_title("(b) Band-pass + notch vs. pre-emphasis", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "Fig. 4-2 Spectrum comparison at different preprocessing stages (Welch PSD)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = OUTPUT_DIR / "fig_4-2_spectrum_two_panels.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_4_2_spectrum_two_panels()
