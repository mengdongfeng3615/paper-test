"""
孪生网络感知编码器。

- 基础编码器：CNN -> 全局平均池化 -> 嵌入向量 z
- 前向：encode(x) 生成 z；距离用 L2。
- 推断接口 encode_perceptual_feature 可对新样本生成感知嵌入。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class ConvEncoder(nn.Module):
    """简单声谱图 CNN 编码器，输出固定维度嵌入。"""

    def __init__(self, in_channels: int = 1, emb_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        h = self.features(x)
        z = self.proj(h)
        return z


class PerceptualSiamese(nn.Module):
    """孪生网络，输出两嵌入及距离。"""

    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(1, emb_dim)
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        d = self.dist(z1, z2)
        return z1, z2, d


def encode_perceptual_feature(spec: torch.Tensor, model: ConvEncoder, device: torch.device) -> torch.Tensor:
    """
    单样本推断接口：输入 (F, T) numpy/torch -> 输出 z (emb_dim,)
    """
    if not torch.is_tensor(spec):
        spec = torch.from_numpy(spec)
    spec = spec.float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,F,T)
    model.eval()
    with torch.no_grad():
        z = model(spec)
    return z.squeeze(0).cpu()
