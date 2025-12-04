import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import os

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义要分析的文件路径
file_paths = [
    'data_raw/excessive-penetration1.wav',
    'data_raw/non-penetration1.wav',
    'data_raw/full-penetration1.wav'
]

# 定义对应的标签名称
labels = [
    '过熔透',
    '未熔透',
    '完全熔透'
]

# 定义噪声增强的SNR级别 (dB)
snr_levels = [10, 8, 5, 0.5]

def add_noise(signal, target_snr_db):
    """向信号添加高斯白噪声以达到目标 SNR"""
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    return signal + noise_volts

def process_and_visualize_audio(file_path, label):
    """处理单个音频文件并生成可视化"""
    try:
        # 加载音频，保持原始采样率 (sr=None)
        y_raw, sr = librosa.load(file_path, sr=None)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请确保文件在当前目录下。")
        # 创建一个虚拟信号用于演示，如果找不到文件
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        y_raw = 0.5 * np.sin(2 * np.pi * 1000 * t) + \
                0.1 * np.sin(2 * np.pi * 50 * t) + \
                0.05 * np.random.randn(len(t))
        print("已使用虚拟信号进行演示。")

    # 为了清晰展示，我们截取信号中间较稳定的一段进行可视化 (例如 0.2 秒)
    duration_to_plot = 0.2
    start_sample = int(len(y_raw) // 2 - (duration_to_plot * sr) // 2)
    end_sample = int(start_sample + duration_to_plot * sr)
    y_segment = y_raw[start_sample:end_sample]
    t_segment = np.arange(len(y_segment)) / sr

    # --- 步骤 1: 幅值归一化 ---
    # 将信号幅值缩放到 [-1, 1] 范围
    y_norm = y_segment / np.max(np.abs(y_segment))

    # --- 步骤 2: 带通滤波 + 50 Hz 陷波 ---
    # 4阶 Butterworth 带通滤波器 (300Hz ~ 18kHz)
    sos_bp = signal.butter(4, [300, 18000], btype='bandpass', fs=sr, output='sos')
    y_bandpass = signal.sosfilt(sos_bp, y_norm)

    # 50 Hz 陷波器 (去除工频干扰)
    # Q 值决定了陷波的宽度，Q=30 是一个常用值
    b_notch, a_notch = signal.iirnotch(w0=50, Q=30, fs=sr)
    y_filtered = signal.lfilter(b_notch, a_notch, y_bandpass)

    # --- 步骤 3: 预加重 ---
    # 一阶预加重滤波器，系数为 0.97
    # y[n] = x[n] - 0.97 * x[n-1]
    y_pre = np.append(y_filtered[0], y_filtered[1:] - 0.97 * y_filtered[:-1])

    # --- 步骤 4: 10 ms 分帧 ---
    # 从处理后的信号中截取一个 10ms 的帧
    frame_len_samples = int(0.01 * sr) # 10ms 对应的采样点数
    # 取片段中间的一个帧
    frame_start_idx = len(y_pre) // 2
    y_frame = y_pre[frame_start_idx : frame_start_idx + frame_len_samples]
    t_frame = np.arange(len(y_frame)) / sr * 1000 # 时间轴转换为毫秒 (ms)

    # --- 步骤 5: 多级噪声增强 ---
    # 为每个SNR级别生成带噪声的帧
    noisy_frames = {}
    for snr in snr_levels:
        noisy_frames[snr] = add_noise(y_frame, target_snr_db=snr)
    
    return t_segment, y_norm, y_filtered, y_pre, t_frame, y_frame, noisy_frames, label

# 处理所有文件
results = []
for file_path, label in zip(file_paths, labels):
    if os.path.exists(file_path):
        result = process_and_visualize_audio(file_path, label)
        results.append(result)
    else:
        print(f"警告：文件 {file_path} 不存在")

# --- 创建可视化图表 ---
if results:
    # 计算总列数：原始信号 + 各个噪声级别
    total_cols = 1 + len(snr_levels)  # 1列原始信号 + 4列噪声信号
    
    fig, axes = plt.subplots(len(results), total_cols, figsize=(4*total_cols, 3*len(results)), constrained_layout=True)
    fig.suptitle('声音信号预处理流程可视化对比', fontsize=16)
    
    # 如果只有一个文件，调整axes的形状
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    # 为每种类型的文件生成一行可视化
    for i, (t_segment, y_norm, y_filtered, y_pre, t_frame, y_frame, noisy_frames, label) in enumerate(results):
        # 1. 显示原始帧（无噪声）
        axes[i, 0].plot(t_frame, y_frame, color='tab:red', marker='o', markersize=2, linestyle='-')
        axes[i, 0].set_title(f'{label}: 原始帧 (无噪声)', fontweight='bold')
        axes[i, 0].set_ylabel('幅值')
        axes[i, 0].set_xlabel('时间 (毫秒)')
        axes[i, 0].set_xlim(0, 10)
        axes[i, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # 2. 显示各个噪声级别的帧
        for j, snr in enumerate(snr_levels):
            axes[i, j+1].plot(t_frame, noisy_frames[snr], color='tab:purple', marker='.', markersize=1, alpha=0.7, label=f'SNR={snr}dB')
            # 为了对比，淡化显示原始帧
            axes[i, j+1].plot(t_frame, y_frame, color='tab:red', alpha=0.3, linewidth=1, label='原始帧 (参考)')
            axes[i, j+1].set_title(f'{label}: 噪声增强 (SNR={snr}dB)', fontweight='bold')
            axes[i, j+1].set_ylabel('幅值')
            axes[i, j+1].set_xlabel('时间 (毫秒)')
            axes[i, j+1].set_xlim(0, 10)
            axes[i, j+1].legend(loc='upper right')
            axes[i, j+1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()
else:
    print("没有找到任何有效的音频文件进行处理。")