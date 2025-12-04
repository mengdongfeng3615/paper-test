from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from src.config import DatasetConfig, RAW_DIR, OUTPUT_DIR, SAMPLE_TO_FILE
from src.font_config import configure_chinese_font

# 配置中文字体，避免乱码
configure_chinese_font()

def fig_4_2_spectrum_refined() -> None:
    """Refined spectrum comparison using a short segment and subplots."""
    cfg = DatasetConfig()
    sample_id = "Sample1"
    wav_path = RAW_DIR / SAMPLE_TO_FILE[sample_id]

    raw, sr = sf.read(wav_path)
    raw = raw.astype(np.float64)

    # use middle 0.5 s segment to avoid transients
    center = len(raw) // 2
    half = int(0.25 * sr)
    seg = raw[center - half : center + half]

    # normalization
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

    def mag_spectrum(sig: np.ndarray):
        n = len(sig)
        win = np.hanning(n)
        spec = np.fft.rfft(sig * win)
        freq = np.fft.rfftfreq(n, 1 / sr)
        mag_db = 20 * np.log10(np.abs(spec) + 1e-12)
        return freq, mag_db

    freq, mag_raw = mag_spectrum(seg)
    _, mag_bp = mag_spectrum(bp_notch)
    _, mag_pre = mag_spectrum(pre)

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    for ax, mag, title in zip(
        axes,
        [mag_raw, mag_bp, mag_pre],
        ["原始信号", "带通滤波+陷波", "预加重"],
    ):
        ax.plot(freq / 1000.0, mag, color="C0", lw=0.8)
        ax.set_xlim(0, 10)
        ax.set_ylim(mag.max() - 80, mag.max() + 5)
        ax.set_ylabel("幅度 / dB")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("频率 / kHz")
    fig.suptitle("图 4-2 预处理前后频谱对比(0.5 s 片段, 0–10 kHz)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "fig_4-2_spectrum_refined.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print("Run this module via make_figures.py main().")