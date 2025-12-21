import matplotlib.pyplot as plt
import numpy as np
import os
snrs = ["clean", "10 dB", "8 dB", "5 dB", "2 dB", "0.5 dB"]
pos = np.arange(len(snrs))
acc = {
    "factory": [0.8666666667, 0.8666666667, 0.8666666667, 0.8444444444, 0.8, 0.8222222222],
    "white":   [0.8666666667, 0.8, 0.8, 0.7555555556, 0.7111111111, 0.8],
    "pink":    [0.8666666667, 0.8222222222, 0.8, 0.8222222222, 0.6444444444, 0.6888888889],
}
f1 = {
    "factory": [0.8642533937, 0.8650189168, 0.8642533937, 0.8428731762, 0.7959595960, 0.8155850727],
    "white":   [0.8642533937, 0.7908159393, 0.7908159393, 0.7498412698, 0.6902728816, 0.7980158730],
    "pink":    [0.8642533937, 0.8110507246, 0.7949242424, 0.8110507246, 0.6381209781, 0.6929493596],
}
colors = {"factory": "#2ca02c", "white": "#1f77b4", "pink": "#ff7f0e"}
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for noise, vals in acc.items():
    axes[0].plot(pos, vals, marker='o', label=f"{noise.capitalize()} noise", color=colors[noise])
axes[0].axhline(1/3, color='gray', linestyle='--', linewidth=1, label='Chance level (1/3)')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Noise-robust SVM performance across SNR levels')
axes[0].set_ylim(0.3, 1.0)
axes[0].grid(True, linestyle='--', alpha=0.3)
for noise, vals in f1.items():
    axes[1].plot(pos, vals, marker='o', label=f"{noise.capitalize()} noise", color=colors[noise])
axes[1].axhline(1/3, color='gray', linestyle='--', linewidth=1, label='Chance level (1/3)')
axes[1].set_ylabel('Macro-F1')
axes[1].set_xlabel('SNR condition')
axes[1].set_ylim(0.3, 1.0)
axes[1].grid(True, linestyle='--', alpha=0.3)
axes[1].set_xticks(pos)
axes[1].set_xticklabels(snrs)
handles_acc, labels_acc = axes[0].get_legend_handles_labels()
handles_f1, labels_f1 = axes[1].get_legend_handles_labels()
axes[0].legend(handles_acc[:3] + [handles_acc[-1]], labels_acc[:3] + [labels_acc[-1]], loc='lower left', frameon=False)
axes[1].legend(handles_f1[:3] + [handles_f1[-1]], labels_f1[:3] + [labels_f1[-1]], loc='lower left', frameon=False)
fig.tight_layout()
os.makedirs('outputs/figures', exist_ok=True)
fig.savefig('outputs/figures/noise_robust_performance.png', dpi=300)
print('Saved figure to outputs/figures/noise_robust_performance.png')
