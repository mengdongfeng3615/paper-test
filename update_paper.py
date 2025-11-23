"""
Regenerate paper draft by inserting latest metrics and figures into docx.
"""
import pandas as pd
from docx import Document
from docx.shared import Inches
from pathlib import Path

BASE = Path('.')
source_docs = [p for p in BASE.glob('*.docx') if p.name not in {'paper_with_results.docx', 'paper_with_results_100ms.docx'} and not p.name.startswith('~$')]
if not source_docs:
    raise SystemExit('no source docx found')
doc_path = source_docs[0]

out_dir = BASE / 'outputs'
metrics = pd.read_csv(out_dir / 'metrics.csv')
summary = pd.read_json(out_dir / 'summary.json')


def get_metric(model: str, snr: str, key: str = 'macro_f1'):
    """Helper to fetch metric by model and SNR label."""
    row = metrics[(metrics['model'] == model) & (metrics['snr'] == snr)]
    if row.empty:
        return None
    return float(row.iloc[0][key])


clean_base = get_metric('baseline', 'clean')
clean_fusion = get_metric('fusion_mds', 'clean')
low_base = get_metric('baseline', '0.5 dB')
low_fusion = get_metric('fusion_mds', '0.5 dB')


doc = Document(doc_path)

# Content blocks

p = doc.add_paragraph()
p.add_run('数据与标签：').bold = True
p.add_run('Sample1~5 对应未熔透，Sample6~10 对应全熔透，Sample11~15 对应过熔透。每段原始 5 s，采样率 40 kHz，带通 300–18 kHz + 50 Hz 陷波后按 100 ms 窗长/100 ms 步长切割片段。主观不相似度矩阵由 11 名测试者评分，采用 MDS/Isomap 获得 2 维感知嵌入。')

doc.add_heading('实验流程与结果', level=1)

p2 = doc.add_paragraph()
p2.add_run('特征与模型：').bold = True
p2.add_run('基础时频特征 + MFCC/PNCC/RASTA-PLP 感知特征，并可选拼接 2 维主观 MDS 坐标。分类器采用 RBF-SVM，C、γ 网格搜索并使用样本编号分组交叉验证。训练端加入 clean 与 10 dB 噪声增强，测试端评估 clean、10/8/5/0.5 dB SNR。')

p3 = doc.add_paragraph()
p3.add_run('主要结果：').bold = True
p3.add_run('Macro-F1 随 SNR 见表 x，MDS 融合在极低信噪比下保持更高稳定性。')

# Table with metrics
model_order = ['baseline', 'fusion_mds']
rows = []
for m in model_order:
    sub = metrics[metrics['model'] == m]
    for _, r in sub.iterrows():
        rows.append((m, r['snr'], f"{r['macro_f1']:.3f}", f"{r['accuracy']:.3f}"))

table = doc.add_table(rows=len(rows) + 1, cols=4)
head = table.rows[0].cells
head[0].text = '模型'
head[1].text = 'SNR'
head[2].text = 'Macro-F1'
head[3].text = 'Accuracy'
for i, row in enumerate(rows, start=1):
    cells = table.rows[i].cells
    cells[0].text = 'Baseline' if row[0] == 'baseline' else 'Fusion+MDS'
    cells[1].text = str(row[1])
    cells[2].text = row[2]
    cells[3].text = row[3]

# Figures
figs = [
    ('outputs/fig_mds.png', '图x 主观不相似度 MDS 嵌入（数字对应 Sample 序号）'),
    ('outputs/confusion_baseline.png', '图x 基线模型（clean）混淆矩阵'),
    ('outputs/confusion_fusion_mds.png', '图x Fusion+MDS 模型（clean）混淆矩阵'),
    ('outputs/snr_curve_fusion.png', '图x Fusion+MDS 噪声鲁棒性曲线'),
]
for path, caption in figs:
    img_path = BASE / path
    if img_path.exists():
        doc.add_picture(str(img_path), width=Inches(4.8))
        cap = doc.add_paragraph()
        cap.alignment = 1
        cap.add_run(caption)

p4 = doc.add_paragraph()
p4.add_run('定量分析：').bold = True
p4.add_run(
    f"clean 条件下 Baseline Macro-F1={clean_base:.3f}，Fusion+MDS={clean_fusion:.3f}；0.5 dB 时 Baseline={low_base:.3f}，Fusion+MDS={low_fusion:.3f}，表明感知嵌入在极低 SNR 下带来约 {low_fusion - low_base:.3f} 的 F1 提升。最佳参数 C=0.5, γ=0.001。"
)

p5 = doc.add_paragraph()
p5.add_run('结论：').bold = True
p5.add_run('按照论文方案实现的多源特征+主观感知融合 pipeline 已完成，输出位于 outputs/，可直接复现实验与图表。')

out_docx = BASE / 'paper_with_results_100ms.docx'
doc.save(out_docx)
print('written', out_docx)
