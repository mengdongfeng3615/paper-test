import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##
# 假设 data 是你的 15x15 矩阵，我的文件在output目录下的 dissimilarity_matrix_paper_format.csv
df = pd.read_csv('outputs/dissimilarity_matrix_paper_format.csv', index_col=0)



import seaborn as sns
import matplotlib.pyplot as plt

# 假设 df 是你的数据框
# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 绘图
g = sns.clustermap(df,
                   cmap="YlGnBu",       # 建议换个颜色，深色背景字会自动变白，或者对比度更好
                   annot=True,          # 显示数值
                   fmt=".2f",           # 保留两位小数
                   annot_kws={"size": 8}, # 调整格子内数字大小
                   figsize=(10, 10),    # 图片大小
                   tree_kws=dict(linewidths=1.5), # 加粗树状图线条
                   cbar_pos=(0.02, 0.85, 0.03, 0.12) # 或者干脆不写这行，让它默认在右边
                   )

# 调整轴标签大小
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=12)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)

plt.show()