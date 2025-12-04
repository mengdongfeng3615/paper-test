"""Generate Chapter 4 figures and tables for the welding sound project (UTF-8, 中文正常显示)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from src.config import DatasetConfig, RAW_DIR, OUTPUT_DIR, SAMPLE_TO_FILE, CLASS_NAMES
from src import audio_utils
from src import features as feat_mod
from src.mds_utils import load_subjective_matrix, SAMPLE_ORDER
from src.font_config import configure_chinese_font

configure_chinese_font()


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def fig_4_1_flowchart() -> None:
    """声信号预处理流程图。"""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")

    steps = [
        "原始信号",
        "幅值归一化",
        "带通滤波\n+ 50 Hz 陷波",
        "预加重",
        "10 ms 分帧",
        "噪声增强\n(多 SNR)",
    ]

    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.5

    for i, (x, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.06, y),
                xytext=(x + 0.06, y),
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )

    ax.set_title("图 4-1 声信号预处理流程框图", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_4-1_flowchart.png", dpi=300)
    plt.close(fig)


def fig_4_2_spectrum() -> None:
    """预处理前后频谱对比（全段 0–20 kHz）。"""
    _ensure_dirs()
    cfg = DatasetConfig()
    sample_id = "Sample1"
    wav_path = RAW_DIR / SAMPLE_TO_FILE[sample_id]

    raw, sr = sf.read(wav_path)
    raw = raw.astype(np.float64)

    if np.max(np.abs(raw)) > 0:
        norm = raw / np.max(np.abs(raw))
    else:
        norm = raw

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

    freq, mag_raw = mag_spectrum(raw)
    _, mag_bp = mag_spectrum(bp_notch)
    _, mag_pre = mag_spectrum(pre)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freq / 1000.0, mag_raw, label="原始", alpha=0.7)
    ax.plot(freq / 1000.0, mag_bp, label="滤波后", alpha=0.7)
    ax.plot(freq / 1000.0, mag_pre, label="预加重后", alpha=0.7)
    ax.set_xlim(0, 20)
    ax.set_xlabel("频率 / kHz")
    ax.set_ylabel("幅度 / dB")
    ax.set_title("图 4-2 预处理前后典型频谱对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_4-2_spectrum.png", dpi=300)
    plt.close(fig)


def fig_4_2_spectrum_refined() -> None:
    """预处理前后频谱对比（取中间 0.5 s，0–10 kHz，三子图）。"""
    _ensure_dirs()
    cfg = DatasetConfig()
    sample_id = "Sample1"
    wav_path = RAW_DIR / SAMPLE_TO_FILE[sample_id]

    raw, sr = sf.read(wav_path)
    raw = raw.astype(np.float64)
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

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    for ax, mag, title in zip(
        axes,
        [mag_raw, mag_bp, mag_pre],
        ["原始信号", "带通+陷波", "预加重后"],
    ):
        ax.plot(freq / 1000.0, mag, color="C0", lw=0.8)
        ax.set_xlim(0, 10)
        ax.set_ylim(mag.max() - 80, mag.max() + 5)
        ax.set_ylabel("幅度 / dB")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("频率 / kHz")
    fig.suptitle("图 4-2(b) 预处理前后典型频谱对比（0.5 s 片段，0–10 kHz）", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "fig_4-2_spectrum_refined.png", dpi=300)
    plt.close(fig)


def tables_4_1_4_2() -> None:
    """输出表 4-1 和表 4-2 的 Markdown 文本。"""
    _ensure_dirs()
    path = OUTPUT_DIR / "tables_ch4.md"
    lines = []

    lines.append("## 表 4-1 传统声学特征列表及物理含义\n")
    lines.append("| 特征名称 | 计算方式简述 | 物理/听感含义 |\n")
    lines.append("|---------|--------------|----------------|\n")
    lines.append("| RMS 能量 | 短时均方根 (sqrt(E[x^2])) | 反映声强和放电能量水平，与电弧稳定性相关 |\n")
    lines.append("| 零交叉率 ZCR | 单帧符号变化次数占比 | 间接表征高频成分比例，熔透不足时 ZCR 通常偏低 |\n")
    lines.append("| 频谱质心 | 能量加权平均频率 | 感知明亮度/尖锐度，键孔状态或飞溅会提升质心 |\n")
    lines.append("| 频谱带宽 | 以质心为中心的能量方差平方根 | 频带扩展程度，反映电弧–熔池耦合的频谱扩散 |\n")
    lines.append("| 谱滚降频率 | 累积能量达到总能量 95% 的频率 | 能量主集中频带上界，区分低频轰鸣与高频尖啸 |\n")
    lines.append("| 谱平坦度 | 能量谱几何均值与算术均值之比 | 接近 1 表示接近白噪声，接近 0 表示接近纯音或窄带峰 |\n")
    lines.append("| 分频带能量比 | 0–1/1–8/8–18 kHz 能量占比 | 对不同熔透状态下低中高频能量迁移进行分段描述 |\n")
    lines.append("\n")

    lines.append("## 表 4-2 感知相关特征配置说明\n")
    lines.append("| 特征类型 | 滤波器数量 | 倒谱阶数/阶数 | 帧长/步长 | 是否预加重 | 备注 |\n")
    lines.append("|----------|------------|----------------|-----------|------------|------|\n")
    lines.append("| MFCC | 26 滤波器 | 20 维倒谱 | 25 ms / 10 ms | 否（外部已预加重） | 梅尔尺度，强调人耳频率分辨能力 |\n")
    lines.append("| PNCC | 26 滤波器 | 20 维倒谱 | 25 ms / 10 ms | 否 | 引入噪声抑制和功率归一化，提升噪声鲁棒性 |\n")
    lines.append("| RASTA-PLP | 26 伪临界带 | 13 阶 PLP 系数 | 25 ms / 10 ms | 否 | 对时间包络做带通滤波，抑制慢变干扰 |\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def fig_4_3_feature_scatter() -> None:
    """RMS vs 频谱质心散点图。"""
    _ensure_dirs()
    cfg = DatasetConfig()
    from src.data_utils import load_segments

    all_segments = load_segments(cfg)

    xs = []
    ys = []
    cs = []
    for seg in all_segments:
        base = feat_mod._basic_features(seg.data, cfg.sample_rate)
        rms = base[0]
        centroid = base[2]
        xs.append(rms)
        ys.append(centroid / 1000.0)
        cs.append(seg.label)

    xs = np.array(xs)
    ys = np.array(ys)
    cs = np.array(cs)

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, name in CLASS_NAMES.items():
        mask = cs == label
        ax.scatter(xs[mask], ys[mask], s=12, alpha=0.6, label=name)
    ax.set_xlabel("RMS 能量")
    ax.set_ylabel("频谱质心 / kHz")
    ax.set_title("图 4-3 典型特征空间散点图（RMS vs 频谱质心）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_4-3_feature_scatter.png", dpi=300)
    plt.close(fig)


def fig_4_4_mds_isomap() -> None:
    """2D MDS（默认）与 Isomap（可选）嵌入。"""
    _ensure_dirs()
    avg, mds_coords, iso_coords = load_subjective_matrix(with_isomap=True)

    cls = []
    for sid in SAMPLE_ORDER:
        idx = int(sid.replace("Sample", ""))
        if idx <= 5:
            cls.append(0)
        elif idx <= 10:
            cls.append(1)
        else:
            cls.append(2)
    cls = np.array(cls)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, coords, title in zip(
        axes,
        [mds_coords, iso_coords],
        ["MDS", "Isomap"],
    ):
        for label, name in CLASS_NAMES.items():
            mask = cls == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=60,
                alpha=0.85,
                label=name,
            )
        for i, (x, y) in enumerate(coords):
            ax.text(x + 0.01, y + 0.01, str(i + 1), fontsize=8, alpha=0.8)
        ax.set_xlabel("维度 1")
        ax.set_ylabel("维度 2")
        ax.set_title(title)
    axes[0].legend(loc="best")
    fig.suptitle("图 4-4 主观感知嵌入二维投影图（MDS 与 Isomap）")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "fig_4-4_mds_isomap.png", dpi=300)
    plt.close(fig)


def main():
    fig_4_1_flowchart()
    fig_4_2_spectrum()
    fig_4_2_spectrum_refined()
    tables_4_1_4_2()
    fig_4_3_feature_scatter()
    fig_4_4_mds_isomap()
    print("Figures and tables written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
