"""字体配置模块，确保 Matplotlib 中文不乱码。"""

import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path


def configure_chinese_font():
    """配置 Matplotlib 支持中文字体并解决负号显示问题。"""
    candidates = [
        Path(r"C:\\Windows\\Fonts\\simhei.ttf"),
        Path(r"C:\\Windows\\Fonts\\msyh.ttc"),
        Path(r"C:\\Windows\\Fonts\\simsun.ttc"),
    ]
    font_names = []
    for fp in candidates:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            name_lower = fp.name.lower()
            if "simhei" in name_lower:
                font_names.append("SimHei")
            elif "msyh" in name_lower:
                font_names.append("Microsoft YaHei")
            elif "simsun" in name_lower:
                font_names.append("SimSun")
    if not font_names:
        font_names = ["DejaVu Sans"]

    plt.rcParams["font.sans-serif"] = font_names
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
