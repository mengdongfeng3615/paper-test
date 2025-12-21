"""
训练孪生网络感知编码器，将主观不相似度矩阵 D 拟合为嵌入距离。

输出: ckpt/perceptual_encoder.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from src.config import DatasetConfig
from perceptual.perceptual_dataset import build_perceptual_dataset
from perceptual.perceptual_encoder import PerceptualSiamese


def collate_fn(batch):
    specs1, specs2, targets = [], [], []
    for s1, s2, t in batch:
        specs1.append(torch.from_numpy(s1).unsqueeze(0))  # (1,F,T)
        specs2.append(torch.from_numpy(s2).unsqueeze(0))
        targets.append(torch.tensor([t], dtype=torch.float32))
    return torch.stack(specs1, dim=0), torch.stack(specs2, dim=0), torch.cat(targets, dim=0)


def train(args):
    device = torch.device(args.device)
    cfg = DatasetConfig()
    ds, _ = build_perceptual_dataset(cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = PerceptualSiamese(emb_dim=args.emb_dim).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for specs1, specs2, targets in loader:
            specs1 = specs1.to(device)
            specs2 = specs2.to(device)
            targets = targets.to(device)
            _, _, d = model(specs1, specs2)
            loss = criterion(d, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * specs1.size(0)
        avg = total / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "emb_dim": args.emb_dim}, ckpt_dir / "perceptual_encoder.pth")
            print("Saved checkpoint to ckpt/perceptual_encoder.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
