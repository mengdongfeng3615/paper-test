from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from src.config import DatasetConfig, RAW_DIR, OUTPUT_DIR, SAMPLE_TO_FILE


def fig_4_2_spectrum_zoomed() -> None:
    """Spectrum comparison with narrowed axes and high-frequency inset.

    主图：0–10 kHz，dB 轴约束在 [peak-80, peak+5]；
    右上角插入 10–20 kHz 的频谱局部放大图。
    """
    cfg = DatasetConfig()
    sample_id = "Sample1"
    wav_path = RAW_DIR / SAMPLE_TO_FILE[sample_id]

    raw, sr = sf.read(wav_path)
    raw = raw.astype(np.float64)

    # 取中间 0.5 s，避免起止瞬态干扰
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

    # 主图：0–10 kHz
    fig, ax_main = plt.subplots(figsize=(6.5, 4.5))
    for mag, label, style in [
        (mag_raw, "原始", "C0"),
        (mag_bp, "滤波后", "C1"),
        (mag_pre, "预加重后", "C2"),
    ]:
        ax_main.plot(freq / 1000.0, mag, label=label, alpha=0.8)

    ax_main.set_xlim(0, 10)
    peak = max(mag_raw.max(), mag_bp.max(), mag_pre.max())
    ax_main.set_ylim(peak - 80, peak + 5)
    ax_main.set_xlabel("频率 / kHz")
    ax_main.set_ylabel("幅度 / dB")
    ax_main.set_title("图 4-2(c) 预处理前后典型频谱对比（0.5 s，0–10 kHz）")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper right")

    # 插图：10–20 kHz 放大
    ax_inset = ax_main.inset_axes([0.55, 0.5, 0.4, 0.4])
    for mag, label, style in [
        (mag_raw, "原始", "C0"),
        (mag_bp, "滤波后", "C1"),
        (mag_pre, "预加重后", "C2"),
    ]:
        ax_inset.plot(freq / 1000.0, mag, alpha=0.8)
    ax_inset.set_xlim(10, 20)
    ax_inset.set_ylim(peak - 80, peak + 5)
    ax_inset.set_xticks([10, 15, 20])
    ax_inset.set_yticks([])
    ax_inset.set_title("10–20 kHz", fontsize=8)
    ax_inset.grid(True, alpha=0.2)

    fig.tight_layout()
    (OUTPUT_DIR / "fig_4-2_spectrum_zoomed.png").parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig_4-2_spectrum_zoomed.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_4_2_spectrum_zoomed()
