基于主观不相似度–MDS 与多源声学特征的焊接熔透状态识别系统说明
1 系统总体目标与思路
本工程围绕直流正接 TIG 焊接过程中电弧声信号与熔透状态之间的关系，构建了一个结合主观评价模型和客观听觉特征的识别系统。其主线功能是：

利用焊工主观“不相似度”评价构建焊接声样本的感知嵌入空间（MDS 坐标），作为“人耳感知层”的锚点。
在此基础上，设计多源声学特征（传统时频特征 + 感知相关特征），并与 MDS 嵌入进行融合。
通过噪声增强策略，在多级 SNR（clean、10、8、5、0.5 dB）条件下训练/评估 SVM 分类器，实现对未熔透、全熔透、过熔透三种典型状态的鲁棒识别。
从论文视角，本系统实现的是“主观评价模型 → 客观识别模型”的完整落地：以主观不相似度矩阵为 anchor，构建低维“人耳音色空间”，再将该几何结构融入基于多源特征的熔透状态分类框架中，并在强噪声环境下验证其有效性。

2 数据与标注
2.1 原始声学数据
数据目录：data_raw/

文件类型：单通道 .wav，采样率 40 kHz，长度约 5 s。

命名与类别对应关系：

未熔透（Non-penetration）：
non-penetration1.wav ~ non-penetration5.wav ⇔ Sample1 ~ Sample5
完全熔透（Full penetration）：
full-penetration1.wav ~ full-penetration5.wav ⇔ Sample6 ~ Sample10
过熔透（Excessive penetration）：
excessive-penetration1.wav ~ excessive-penetration5.wav ⇔ Sample11 ~ Sample15
类别索引在代码中统一为：

0：未熔透（Non）
1：全熔透（Full）
2：过熔透（Excessive）
2.2 主观不相似度矩阵与 MDS 锚点
文件：data_raw/assessment_data.xlsx

内容：11 名受试者对 15 段焊接声样本进行成对“不相似度”评分（Sample1~Sample15）。

处理流程（src/mds_utils.py）：

逐测试者读取评分子表，得到 15×15 矩阵；
对角线置零，矩阵对称化：(M + M^T) / 2；
对 11 名受试者矩阵取算术平均，得到全体平均不相似度矩阵；
使用 sklearn.manifold.MDS 在：
主线方案：生成 2 维 MDS 坐标（用于分类特征融合）；
鲁棒方案中还用到 4 维嵌入（见 robust_pipeline/mds_utils.py）。
输出：

outputs/mds_coordinates.csv：包含每个 Sample 的类别及其 MDS 坐标；
outputs/fig_mds.png：MDS 嵌入散点图（颜色区分类别，数字标注 Sample 编号）。
这一步为后续“主观感知空间”提供了绝对锚点，使得所有客观特征与模型的学习都围绕这一感知几何结构展开。

3 信号预处理与分段策略
3.1 预处理流程（src/audio_utils.py）
针对每个原始 wav 信号，执行如下步骤：

幅值归一化：
将整体最大绝对值归一到 1，消除绝对幅度差异。
带通滤波：
4 阶 Butterworth 带通：300–18 kHz，保留与电弧声相关的主要频段。
50 Hz 陷波：
IIR notch 抑制工频干扰及其相邻频带。
预加重：
一阶高通滤波 y[n] = x[n] - α x[n-1]（α≈0.97），补偿高频能量、增强音色相关细节。
3.2 时域分段（10 ms）
参数在 src/config.py: DatasetConfig 中配置：
segment_seconds = 0.01（10 ms）
hop_seconds = 0.01（10 ms，无重叠）
每条 5 s 信号约被切分为 500 段 10 ms 片段。
为控制计算量和样本冗余，引入 max_segments_per_sample：
当前配置：每个原始样本最多随机选取 30 个分段参与特征提取和模型训练/测试。
分段与抽样逻辑见 src/data_utils.load_segments：
按上述预处理得到整条波形；
利用滑动窗口分段；
若分段数超过阈值，采用随机下采样保留固定数量的片段。
4 特征提取：多源声学与感知相关特征
特征提取实现于 src/features.py，面向每个 10 ms 片段，构建如下多源向量：

4.1 传统时频特征
时域：
RMS（短时能量 √E）
零交叉率 ZCR（反映高频成分丰富程度）
频域统计量（基于 FFT 能量谱）：
频谱质心（Spectral Centroid）
频谱带宽（Bandwidth）
95% 能量滚降频率（Spectral Rolloff）
谱平坦度（Spectral Flatness）
分频带能量占比：
低频：0–1 kHz
中频：1–8 kHz
高频：8–18 kHz
通过这三段能量比刻画熔透状态下频带能量迁移特性。
4.2 感知相关特征：MFCC / PNCC / RASTA-PLP
对每个 10 ms 段，再内部使用一个短时窗参与 spafe 库的特征计算：
自适应帧窗：_frame_window(cfg)；窗口长度取 25 ms 和片段长度中的较小值，步长取 10 ms 和 hop 的较小值，以避免短片段导致负维度。
提取：
MFCC（梅尔倒谱系数）
PNCC（感知噪声鲁棒倒谱）
RASTA-PLP（相对谱–感知线性预测）
对时序帧特征做均值 + 标准差聚合，得到每种特征的统计向量。
4.3 多特征融合与 MDS 坐标拼接
将上述：
传统特征向量（RMS + ZCR + 频谱统计 + 分带能量比）
MFCC 统计
PNCC 统计
RASTA-PLP 统计
按照顺序拼接为一个长向量。
在 “fusion_mds” 模式下，同时将该片段对应的 2 维 MDS 坐标（通过 sample_id 查表）拼接到特征尾部，从而在特征空间中显式引入“人耳感知坐标”。
最终得到“Baseline 特征”（无 MDS）和“Fusion+MDS 特征”两种输入模式，用于后续分类模型对比。

5 噪声增强与多轮随机划分方案
5.1 噪声增强（Data Augmentation）
噪声增强逻辑在 src/modeling._feature_matrix 和 add_noise 中实现：

训练端噪声档：
使用 (clean, 10 dB, 8 dB, 5 dB) 四个 SNR 水平叠加高斯白噪声进行增强。
测试端（评估）噪声档：
snr_eval_levels = (clean, 10, 8, 5, 0.5 dB)。
噪声生成：
对每个 SNR，计算当前片段的信号功率；
根据目标 SNR 推出噪声功率，并按高斯分布生成噪声；
用固定 seed 方案保证同一 SNR 下不同 round 的可复现性。
训练阶段使用 clean+多级中高 SNR 进行增强，旨在让分类器学到更稳定、更具有物理意义且主观一致的判别边界。

5.2 多轮随机 3/1/1 样本划分
为充分评估模型的统计稳定性与泛化能力，本工程在 run_pipeline.py 中采用多轮 random split：

每轮按熔透状态独立划分（3/1/1）：
对每类的 Sample 编号（1–5, 6–10, 11–15），随机打乱后，取：
3 个作为训练 + 验证样本；
1 个作为测试样本；
训练集使用 train+val 所有片段（在当前实现中 val 与 train 合并，仅用于调参的 GroupKFold 交叉验证）；
测试集仅使用 test 样本对应片段；
通过 N_ROUNDS（当前 2 轮）重复上述划分，每轮 seed 不同，保证测试结果不依赖于某一固定划分。
每轮训练中，模型在 clean+10/8/5 dB 噪声增强下进行拟合，然后在 clean/10/8/5/0.5 dB 多级噪声条件下评估。

6 分类模型与评估指标
6.1 分类模型：RBF-SVM
实现在 src/modeling.train_model：
管道：StandardScaler + SVC(kernel='rbf')；
类别权重：class_weight='balanced'，缓解类间样本数量略有差异的影响。
超参数搜索：
惩罚系数 C ∈ {0.5, 1.0, 2.0}；
核参数 gamma ∈ {scale, 0.01, 0.001}。
交叉验证：
采用 GroupKFold，分组依据 sample_id，保证同一物理焊缝的不同片段不会同时出现在训练和验证折中，防止数据泄漏。
6.2 评估指标与输出
每轮、每模型、每 SNR 档位上计算：
总体准确率（Accuracy）；
宏平均 F1 值（Macro-F1）；
3×3 混淆矩阵（类别顺序：未熔透、全熔透、过熔透）。
输出文件：
outputs/metrics_rounds.csv：包含每轮、每模型、每 SNR 的 Acc 与 Macro-F1；
outputs/metrics_summary.csv：对 rounds 维度做平均与标准差汇总；
每轮 clean 条件的混淆矩阵图：
outputs/confusion_baseline_round*.png
outputs/confusion_fusion_mds_round*.png。
通过对 clean 及多级噪声条件下的性能曲线和混淆矩阵进行对比，可以定量分析：

多源特征 + MDS 嵌入对识别精度的提升；
噪声增强训练下模型在不同 SNR 的鲁棒性变化趋势；
不同熔透状态（特别是未熔透 vs 过熔透）之间的易混淆程度。
7 使用说明与复现实验步骤
7.1 环境准备
Python 版本：3.11+（当前工程在 3.14 环境下运行）
主要依赖：
numpy, scipy, pandas, matplotlib
scikit-learn
soundfile
python-docx（用于 Word 文档自动生成）
spafe（MFCC/PNCC/RASTA-PLP 特征）
安装示例（如需）：
pip install numpy scipy pandas matplotlib scikit-learn soundfile python-docx spafe
7.2 跑完整主线流程
在项目根目录下执行：

python run_pipeline.py
程序将：
读取主观评分表，计算 2D MDS 坐标并绘制 MDS 散点图；
对 15 条焊接声进行预处理、10 ms 分段、下采样；
多轮随机 3/1/1 样本划分，训练 Baseline 和 Fusion+MDS 模型（含噪声增强）；
在 clean/10/8/5/0.5 dB 下评估各轮性能，输出 per-round 报告与汇总表；
绘制 clean 条件下的混淆矩阵，并输出图像文件。
7.3 与论文文本集成
工程已提供 update_paper.py 脚本（以及 paper_with_results*.docx 文件），可将最新实验指标和图表插入到论文初稿中对应章节。具体做法是：

确保 outputs/ 中已有最新的 metrics.csv / metrics_rounds.csv 和图像。

运行：

python update_paper.py
在生成的 paper_with_results_100ms.docx 或后续版本中检查“实验流程与结果”章节，里面包含：

当前数据集 / 切分 / 噪声增强策略说明；
Baseline 与 Fusion+MDS 的主要数值结果表；
MDS 嵌入图、混淆矩阵图、噪声鲁棒性曲线。
8 小结与扩展方向
本工程实现了一条完整的、可重复的实验主线：

从焊工主观不相似度矩阵出发，构建 MDS 感知嵌入；
对原始焊接电弧声进行预处理与精细分段，设计多源时频与感知特征；
在多级 SNR 条件下使用噪声增强训练 RBF-SVM 分类器；
在多轮随机样本划分下统计 clean 与不同噪声级别的识别精度与宏平均 F1。
在此基础上，可进一步拓展的研究方向包括：

引入更丰富的深度学习特征（如 CNN 光谱嵌入），与 MDS 坐标进行后端融合；
探索 SVR 或更复杂的回归模型，将“噪声特征 → 干净 MDS 坐标”的非线性映射显式建模；
扩大试验数据集（更多材料、板厚与工况），验证方法的工程泛化能力；
结合电压、电流、视觉信息构建多模态焊接质量在线监测框架。
如果你希望，我也可以基于这份内容生成结构化的 Word 模板大纲（带章节、图表占位符），便于直接投稿使用。