"""
训练鲁棒注意力模型，评估多噪声/多 SNR 泛化。

特性：
- 支持 white/pink/brown/factory 噪声注入。
- 支持加载感知编码器输出并与声学特征融合。
- 评估输出保存到 results/robust_experiments_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.config import DatasetConfig, RAW_DIR, SAMPLE_TO_FILE
from src import audio_utils
from src.data_utils import Segment, _label_from_sample
from data.robust_loader import RobustWeldSoundDataset, RobustLoaderConfig
from utils.noise import load_factory_noises
from features.robust_features import extract_robust_features
from models.robust_attention_model import RobustAttentionModel
from perceptual.perceptual_encoder import ConvEncoder, encode_perceptual_feature
from perceptual.perceptual_dataset import _compute_spectrogram


def build_data_index(cfg: DatasetConfig) -> List[Dict[str, object]]:
    """读取并预处理所有样本，切片成 10 ms 片段，与原 pipeline 保持一致。"""
    seg_cfg = cfg
    data_index: List[Dict[str, object]] = []
    for sid, fname in SAMPLE_TO_FILE.items():
        label = _label_from_sample(sid)
        raw = audio_utils.load_audio(str(RAW_DIR / fname), seg_cfg)
        proc = audio_utils.preprocess_signal(raw, seg_cfg)
        segs = audio_utils.segment_signal(proc, seg_cfg)
        if seg_cfg.max_segments_per_sample and len(segs) > seg_cfg.max_segments_per_sample:
            rng = np.random.default_rng(seg_cfg.random_seed)
            idx = rng.choice(len(segs), size=seg_cfg.max_segments_per_sample, replace=False)
            segs = [segs[i] for i in idx]
        for seg in segs:
            data_index.append({"sample_id": sid, "label": label, "waveform": seg})
    return data_index


def feature_transform(waveform: np.ndarray, cfg: DatasetConfig):
    frame_feat, global_feat = extract_robust_features(waveform, cfg.sample_rate)
    # frame_feat: (T, F) -> (F, T) for later CNN1d input
    return frame_feat.T, global_feat


def collate_batch(batch, device: torch.device, perceptual_model: ConvEncoder | None = None):
    frame_feats, global_feats, labels, sample_ids = [], [], [], []
    perceptual_vecs = []
    for item in batch:
        # item["feature"] shape (F, T)
        frame, glob = feature_transform(item["feature"].numpy(), DatasetConfig())
        frame_feats.append(torch.from_numpy(frame).float())
        global_feats.append(torch.from_numpy(glob).float())
        labels.append(item["label"])
        sample_ids.append(item["sample_id"])
        if perceptual_model is not None:
            spec = _compute_spectrogram(item["feature"].numpy(), DatasetConfig().sample_rate)
            z = encode_perceptual_feature(spec, perceptual_model, device)
            perceptual_vecs.append(z)
    frame_feats = torch.stack(frame_feats).to(device)  # (B, F, T)
    global_feats = torch.stack(global_feats).to(device)
    labels = torch.stack(labels).to(device)
    if perceptual_model is not None:
        perceptual_vecs = torch.stack(perceptual_vecs).to(device)
    else:
        perceptual_vecs = None
    return frame_feats.transpose(1, 2), global_feats, perceptual_vecs, labels, sample_ids


def train_one_epoch(
    model: RobustAttentionModel,
    loader: DataLoader,
    criterion,
    optim,
    device: torch.device,
    perceptual_model: ConvEncoder | None,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        frame, glob, z, labels, _ = batch
        frame = frame.to(device)
        glob = glob.to(device)
        labels = labels.to(device)
        if z is not None:
            z = z.to(device)
        optim.zero_grad()
        logits = model(frame, glob, z)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(1, n)


def evaluate(
    model: RobustAttentionModel,
    loader: DataLoader,
    device: torch.device,
    perceptual_model: ConvEncoder | None,
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            frame, glob, z, labels, _ = batch
            frame = frame.to(device)
            glob = glob.to(device)
            labels = labels.to(device)
            if z is not None:
                z = z.to(device)
            logits = model(frame, glob, z)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def make_loader(data_index, cfg_loader: RobustLoaderConfig, batch_size: int, shuffle: bool, fixed_noise=None):
    ds = RobustWeldSoundDataset(data_index, cfg_loader, fixed_noise=fixed_noise)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda b: b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_perceptual", action="store_true", help="是否加载感知编码器 ckpt/perceptual_encoder.pth")
    parser.add_argument("--noise_types", nargs="+", default=["white", "pink", "brown", "factory"])
    parser.add_argument("--snr_list", nargs="+", type=float, default=[float("inf"), 10, 8, 5, 2, 0.5])
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DatasetConfig()
    data_index = build_data_index(cfg)

    factory_dir = RAW_DIR / "factory_audio"
    noise_bank = load_factory_noises(factory_dir, cfg.sample_rate)

    cfg_loader = RobustLoaderConfig(
        sample_rate=cfg.sample_rate,
        noise_types=args.noise_types,
        snr_choices=args.snr_list,
        use_spec_augment=True,
        freq_mask_width=4,
        time_mask_width=8,
        factory_dir=factory_dir,
    )

    # 训练集随机噪声，验证集用 clean
    train_loader = make_loader(data_index, cfg_loader, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(
        data_index,
        cfg_loader,
        batch_size=args.batch_size,
        shuffle=False,
        fixed_noise=("white", float("inf")),
    )

    # 感知编码器
    perceptual_model = None
    perceptual_dim = None
    if args.use_perceptual:
        ckpt = torch.load(Path("ckpt/perceptual_encoder.pth"), map_location=device)
        perceptual_model = ConvEncoder(1, ckpt.get("emb_dim", 32)).to(device)
        perceptual_model.load_state_dict(ckpt["model"])
        perceptual_model.eval()
        perceptual_dim = ckpt.get("emb_dim", 32)

    # dummy batch to infer frame/global dims
    sample_item = train_loader.dataset[0]
    frame_feat, global_feat = feature_transform(sample_item["feature"].numpy(), cfg)
    model = RobustAttentionModel(
        input_dim=frame_feat.shape[0],
        hidden_dim=64,
        lstm_layers=1,
        num_classes=3,
        perceptual_dim=perceptual_dim,
        global_dim=global_feat.shape[0],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    for epoch in range(args.epochs):
        # collate inside loop to inject fresh noise each epoch
        train_ds = RobustWeldSoundDataset(data_index, cfg_loader, noise_bank, fixed_noise=None)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: b)
        batches = []
        for b in train_loader:
            batches.append(b)
        # 训练
        model.train()
        total, n = 0.0, 0
        for raw_batch in batches:
            frame_feats, global_feats, z_vecs, labels, _ = collate_batch(raw_batch, device, perceptual_model)
            optim.zero_grad()
            logits = model(frame_feats, global_feats, z_vecs)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            total += loss.item() * labels.size(0)
            n += labels.size(0)
        avg = total / max(1, n)
        # 验证
        val_batches = []
        for rb in val_loader:
            val_batches.append(rb)
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for raw_batch in val_batches:
                frame_feats, global_feats, z_vecs, labels, _ = collate_batch(raw_batch, device, perceptual_model)
                logits = model(frame_feats, global_feats, z_vecs)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            val_acc = accuracy_score(y_true, y_pred)
            val_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={avg:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

    # 多噪声/SNR 评估
    results: List[Dict[str, object]] = []
    for ntype in args.noise_types:
        for snr in args.snr_list:
            eval_loader = make_loader(
                data_index,
                cfg_loader,
                batch_size=args.batch_size,
                shuffle=False,
                fixed_noise=(ntype, float(snr)),
            )
            eval_batches = []
            for rb in eval_loader:
                eval_batches.append(rb)
            model.eval()
            with torch.no_grad():
                all_preds, all_labels = [], []
                for raw_batch in eval_batches:
                    frame_feats, global_feats, z_vecs, labels, _ = collate_batch(raw_batch, device, perceptual_model)
                    logits = model(frame_feats, global_feats, z_vecs)
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                y_pred = np.concatenate(all_preds)
                y_true = np.concatenate(all_labels)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="macro")
            results.append(
                {
                    "noise_type": ntype,
                    "snr": "clean" if np.isinf(snr) else f"{snr} dB",
                    "accuracy": acc,
                    "macro_f1": f1,
                }
            )
            print(f"[Eval] noise={ntype} snr={snr}: acc={acc:.3f} f1={f1:.3f}")

    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / "robust_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/robust_experiments_summary.json")


if __name__ == "__main__":
    main()
