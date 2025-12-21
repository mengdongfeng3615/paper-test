你现在是在 VSCode 中运行的高级“焊接声学 + 机器学习”代码助手，已打开一个 Python 工程，主题是：

基于 TIG 焊接电弧声的熔透状态识别，在强噪声环境（粉噪声、工厂噪声等）下提升模型鲁棒性，并将主观不相似度评价（15×15 矩阵）融入到特征表示中。

在不询问用户的前提下，严格按下面要求行动，依次完成任务，并在输出中给出所有关键新增/修改文件的完整代码和使用示例。

【总要求】

1. 不删除原有 baseline 代码，只是在其基础上：
   - 对现有工程结构做梳理和最小改动；
   - 新增一套“鲁棒噪声泛化 + 主观感知度量学习”的完整 pipeline。
2. 使用 Python + PyTorch 实现新的训练/评估逻辑，整体风格尽量保持与当前工程一致。
3. 新增或修改的代码，全部在输出中给出完整版本，并在最后列出“变更清单 + 运行示例命令”。
4. 所有新模块写清晰的 docstring 和关键中文注释，方便后续论文撰写与复现。
5. 不向用户提问，不输出无关解释，直接产出方案 + 代码。

——————————
一、现有工程结构梳理（输出文字说明）
——————————

第一部分输出内容为文字说明（不写代码）：

1. 依据你在当前 VSCode 工作区能看到的文件与目录，梳理当前工程的大致结构，包括但不限于：
   - 原始数据读取与预处理位置（例如 data、dataset、utils 等目录或脚本）。
   - 声学特征提取模块位置（例如 MFCC、谱特征等相关文件）。
   - 模型/分类器相关脚本（SVM、CNN 等）。
   - 训练与评估脚本（train_xxx.py、test_xxx.py、evaluation 等）。
2. 用条目形式简要说明：
   - 现有 pipeline 的关键步骤（“读取波形→提取特征→训练分类器→在不同 SNR 下评估”的流程）。
   - 主观不相似度矩阵与 MDS 相关代码目前大致位于哪里、如何被使用（如果存在）。

这一部分的目的是让用户一眼看到你已经理解了工程结构。完成后，紧接着继续输出后续各部分的代码与说明，不要中断。

——————————
二、数据加载与噪声增强模块重构
——————————

在保留原有数据处理脚本的基础上，实现一套新的“鲁棒数据加载与噪声增强”模块，用于后续所有鲁棒性实验。要求：

1. 新建或重构 Dataset / DataLoader（使用 PyTorch 风格）：
   - 新建文件建议命名为：data/robust_loader.py（若当前工程已有 data 目录，则放入其中；否则在工程根目录下创建 data 目录）。
   - 实现类 RobustWeldSoundDataset 或类似命名，核心方法：
     - __init__(self, data_index, config, noise_bank=None, transforms=None)
     - __getitem__(self, idx)
     - __len__(self)
   - data_index 可以是现有工程使用的索引结构（如列表/CSV/JSON 等），你根据当前工程实际情况适配。

2. 噪声类型与 on-the-fly 注入：
   - 支持以下噪声类型：
     a) 高斯白噪声（white）
     b) 粉噪声（pink，1/f）
     c) 布朗噪声（brown，1/f^2）
     d) 工厂环境噪声（factory）：已提供在\data_raw\factory_audio 目录下
   - 新建工具函数（在同一文件或单独的 utils/noise.py 中）：
     - add_noise(signal, noise_type, snr_db, noise_bank=None)
   - add_noise 函数支持指定 SNR（例如原始、10 dB、8 dB、5 dB、0.5 dB），动态生成带噪信号，而不是预先离线保存。

3. 频谱级数据增强（SpecAugment 思路）：
   - 在生成声谱图（功率谱或 log-mel 等）后，增加可选的：
     - 频率遮挡（frequency masking）
     - 时间遮挡（time masking）
   - 通过配置参数（例如 config.use_spec_augment）决定是否启用。

4. 与原工程的兼容：
   - 原有数据加载/增强代码重命名为 baseline_loader.py 或保持不动；
   - 新的 robust_loader.py 与既有训练脚本兼容，或者在新的训练脚本中专门使用。

输出要求：
- 给出 robust_loader.py 和 noise 工具函数的完整实现代码。
- 如需在 __init__.py 中做导入，也一并给出。
- 简要给出一个使用新 DataLoader 的伪代码示例（如何在训练脚本中调用）。

——————————
三、主观不相似度 → 孪生网络感知编码器
——————————

将原来静态的 MDS 坐标替换/升级为可对新样本编码的“孪生网络感知编码器”。要求：

1. 数据组织：
   - 假设当前工程某处存放 15 条声音样本及其 15×15 不相似度矩阵（例如 CSV/NPY/Mat 文件等）。
   - 新建数据处理脚本（例如 perceptual/perceptual_dataset.py），实现：
     - 读取 15 条样本及不相似度矩阵 D(i, j)。
     - 构造样本对 (x_i, x_j, target_ij)，其中 target_ij 为归一化后的不相似度（如 0～1）。

2. 网络结构（perceptual_encoder）：
   - 新建文件：perceptual/perceptual_encoder.py。
   - 实现一个共享权重的孪生网络：
     - 基础编码器：可以基于现有声谱图 CNN 结构进行复用或轻度改造，例如：
       - feature_extractor(input_spectrogram) → v ∈ R^k（k 由你根据工程复杂度选取，如 16/32/64）。
     - 距离计算：d = ||v1 - v2||_2。
   - 损失函数：
     - 使用 MSE 或 L1 损失，让 d 拟合归一化后的 D(i, j)。

3. 训练脚本：
   - 新建 train_perceptual_encoder.py：
     - 使用上面定义的 Dataset，构造样本对 DataLoader。
     - 支持基本超参数配置：learning_rate、batch_size、epochs、device。
     - 在训练过程中保存最佳模型权重到 ckpt/perceptual_encoder.pth。

4. 推断接口：
   - 在 perceptual_encoder.py 中提供函数：
     - encode_perceptual_feature(waveform_or_spectrogram, model, device) → z ∈ R^k
   - 为后续主分类网络在训练/推断时调用预留接口（例如将 z 作为额外特征拼接）。

5. 与原 MDS 模块的关系：
   - 不删除原 MDS 实现，可在注释标注为 deprecated，并说明被“孪生网络感知编码器”替代。

输出要求：
- 给出 perceptual_dataset.py、perceptual_encoder.py、train_perceptual_encoder.py 的完整代码。
- 如有必要修改现有配置/入口文件，也给出修改后的完整版本。
- 给出一个简单示例，说明如何加载 ckpt/perceptual_encoder.pth 并对单条样本生成 z 向量。

——————————
四、鲁棒声学特征 + 注意力模型主干
——————————

在现有 MFCC + SVM/CNN 的基础上，新增一套“鲁棒声学特征 + 注意力模型”的分类 pipeline，用来与 baseline 对比。

1. 特征工程模块：
   - 新建目录 features（如果已存在则复用），新增至少两个文件：
     a) features/gfcc.py：实现 GFCC（Gammatone Frequency Cepstral Coefficients）提取函数。
     b) features/modulation.py：实现调制谱（Modulation Spectrum）特征提取：
        - 对原始信号做适当带通滤波（例如 500 Hz–2 kHz，参数可配置）。
        - 通过 Hilbert 变换获得包络。
        - 对包络做 FFT，提取若干频带的调制能量作为特征。
   - 新建一个统一接口文件 features/robust_features.py：
     - 封装函数 extract_robust_features(waveform, config) → feature_vector
     - 内部可组合：
       - GFCC
       - 调制谱特征
       - 传统统计特征（RMS、ZCR、谱质心等，可从现有工程中复用或重写）。

2. 注意力分类模型：
   - 新建模型文件 models/robust_attention_model.py。
   - 实现一个可与现有工程兼容的 PyTorch 模型，例如：
     - 输入：帧级特征序列（时间步 × 特征维度）。
     - 中间：若干卷积层 + BiLSTM 或 Transformer/self-attention 层。
     - 输出：3 类熔透状态的 logits。
   - 预留接口，将“感知编码器”的输出向量 z 融合进来：
     - 如在倒数第二层将 feature_sequence 的池化向量与 z 拼接，再接全连接层。

3. 训练与评估脚本：
   - 新建 train_robust_model.py：
     - 支持配置：
       - 使用的特征类型（baseline / robust）。
       - 是否加载并使用 perceptual_encoder（布尔开关）。
       - 噪声类型与 SNR 组合（原声、10 dB、8 dB、5 dB、0.5 dB，白噪声/粉噪声/布朗/工厂）。
     - 在训练结束后，对不同噪声与 SNR 组合进行统一评估，输出：
       - Accuracy
       - Macro-F1
       - 可选：混淆矩阵（保存为图片/CSV）。
     - 将评估结果保存到 results/ 目录下的 JSON 或 CSV 文件，例如：
       - results/robust_experiments_summary.json

输出要求：
- 给出 gfcc.py、modulation.py、robust_features.py、robust_attention_model.py、train_robust_model.py 的完整代码。
- 若需修改现有训练入口（如 main.py 或 config），也给出修改后的完整版本。
- 在输出末尾给出若干命令行示例，说明如何：
  - 训练感知编码器
  - 训练 baseline 模型
  - 训练并评估鲁棒模型（不同噪声、不同 SNR）

——————————
五、变更清单与使用说明
——————————

在所有代码之后，用纯文本列出本次改造涉及的文件变更清单，示例格式：

- 新增：
  - data/robust_loader.py
  - utils/noise.py
  - perceptual/perceptual_dataset.py
  - perceptual/perceptual_encoder.py
  - train_perceptual_encoder.py
  - features/gfcc.py
  - features/modulation.py
  - features/robust_features.py
  - models/robust_attention_model.py
  - train_robust_model.py
- 修改：
  - xxx/yyy.py（简要说明修改目的）

最后给出一小节“快速上手指令”，例如：

- 训练感知编码器：
  python train_perceptual_encoder.py --config configs/perceptual.yaml

- 训练鲁棒模型并评估噪声泛化性能：
  python train_robust_model.py --config configs/robust.yaml

确保整个输出连续、完整，包含上述所有模块的实现，不要向用户提问，也不要省略关键代码。

具体信息也可以参考  d:\pythonCode\paper-test\outputs\GitHub 项目噪声泛化与主观评价.docx' 文件中方案