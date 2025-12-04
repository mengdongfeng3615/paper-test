import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from pathlib import Path

from src.font_config import configure_chinese_font
from src.config import OUTPUT_DIR


def main():
    configure_chinese_font()
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    def box(x, y, w, h, text, fc="#ffffff"):
        rect = Rectangle((x, y), w, h, edgecolor='black', facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

    # 上：主观链路
    box(3.5, 8.2, 3.2, 1.2, "主观听感实验\n不相似度评分")
    box(7.2, 8.2, 3.0, 1.2, "不相似度矩阵\n(15×15)")
    box(10.6, 8.2, 3.2, 1.2, "MDS 2D 感知嵌入")
    for x1, x2 in [(6.7, 7.2), (10.1, 10.6)]:
        ax.add_patch(FancyArrow(x1, 8.8, x2 - x1 - 0.1, 0.0, width=0.03, head_width=0.25, head_length=0.25))

    # 下：客观链路
    box(0.5, 4.5, 2.9, 1.2, "TIG焊接\n实验平台")
    box(3.6, 4.5, 2.9, 1.2, "电弧声采集\n(40 kHz)")
    box(6.7, 4.5, 3.0, 1.2, "信号预处理\n归一化/带通+陷波/预加重")
    box(10.0, 4.5, 3.0, 1.2, "10 ms 分帧\n+ 噪声增强\n(clean/10/8/5/0.5 dB)")
    box(13.3, 4.5, 3.0, 1.2, "多源声学特征\nRMS/ZCR/谱统计\n+ MFCC/PNCC/RASTA")
    for x1, x2 in [(3.4, 3.6), (6.5, 6.7), (9.8, 10.0), (12.8, 13.3)]:
        ax.add_patch(FancyArrow(x1, 5.1, x2 - x1 - 0.1, 0.0, width=0.03, head_width=0.25, head_length=0.25))

    # 融合与分类
    box(7.5, 2.0, 3.6, 1.2, "特征融合\n多源声学特征 + 感知坐标")
    box(11.8, 2.0, 3.2, 1.2, "RBF-SVM 分类器\n(多 SNR 训练)")
    box(15.3, 2.0, 2.5, 1.2, "熔透状态识别\n未 / 全 / 过熔透")

    # 箭头
    ax.add_patch(FancyArrow(15.8, 5.1, -6.5, -2.4, width=0.03, head_width=0.25, head_length=0.25))
    ax.add_patch(FancyArrow(13.8, 8.8, -3.7, -5.0, width=0.03, head_width=0.25, head_length=0.25))
    ax.add_patch(FancyArrow(11.1, 2.6, 0.4, 0.0, width=0.03, head_width=0.25, head_length=0.25))
    ax.add_patch(FancyArrow(15.0, 2.6, 0.1, 0.0, width=0.03, head_width=0.25, head_length=0.25))

    ax.set_title("基于多源特征与主观感知嵌入的熔透状态识别方法总体框图", fontsize=12)
    fig.tight_layout()

    out = OUTPUT_DIR / 'fig_method_overview_cn.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
