import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载音频文件 data_raw/excessive-penetration1.wav
file_path = 'data_raw/excessive-penetration1.wav'
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

# --- 步骤 5: 噪声增强 (示例: 添加一个特定 SNR 的噪声) ---
def add_noise(signal, target_snr_db):
    """向信号添加高斯白噪声以达到目标 SNR"""
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    return signal + noise_volts

# 在 10ms 帧上添加噪声，模拟例如 15dB 的 SNR
y_noisy_frame = add_noise(y_frame, target_snr_db=15)


# --- 创建可视化图表 ---
fig, axes = plt.subplots(5, 1, figsize=(10, 12), constrained_layout=True)
fig.suptitle('声音信号预处理流程可视化\n(基于文件: excessive-penetration1.wav)', fontsize=14)

# 1. 幅值归一化后的信号
axes[0].plot(t_segment, y_norm, color='tab:blue', alpha=0.7, linewidth=1)
axes[0].set_title('1. 幅值归一化后的信号 (0.2s 片段)', fontweight='bold')
axes[0].set_ylabel('归一化幅值')
axes[0].set_xlim(0, t_segment[-1])
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# 2. 滤波后的信号
axes[1].plot(t_segment, y_filtered, color='tab:orange', alpha=0.7, linewidth=1)
axes[1].set_title('2. 带通滤波 (300-18kHz) + 50Hz 陷波后', fontweight='bold')
axes[1].set_ylabel('幅值')
axes[1].set_xlim(0, t_segment[-1])
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# 3. 预加重后的信号
axes[2].plot(t_segment, y_pre, color='tab:green', alpha=0.7, linewidth=1)
axes[2].set_title('3. 预加重后 (系数 0.97)', fontweight='bold')
axes[2].set_ylabel('幅值')
axes[2].set_xlim(0, t_segment[-1])
axes[2].set_xlabel('时间 (秒)')
axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

# 4. 10ms 分帧
axes[3].plot(t_frame, y_frame, color='tab:red', marker='o', markersize=2, linestyle='-')
axes[3].set_title('4. 单个 10ms 帧', fontweight='bold')
axes[3].set_ylabel('幅值')
axes[3].set_xlabel('时间 (毫秒)')
axes[3].set_xlim(0, 10)
axes[3].grid(True, which='both', linestyle='--', linewidth=0.5)

# 5. 噪声增强后的帧
axes[4].plot(t_frame, y_noisy_frame, color='tab:purple', marker='.', markersize=1, alpha=0.7, label='加噪信号')
# 为了对比，淡化显示原始帧
axes[4].plot(t_frame, y_frame, color='tab:red', alpha=0.3, linewidth=1, label='原始帧 (参考)')
axes[4].set_title('5. 噪声增强 (模拟 SNR=15dB)', fontweight='bold')
axes[4].set_ylabel('幅值')
axes[4].set_xlabel('时间 (毫秒)')
axes[4].set_xlim(0, 10)
axes[4].legend(loc='upper right')
axes[4].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()