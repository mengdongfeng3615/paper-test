"""
鲁棒注意力分类模型：
- 输入帧级特征序列 (B, T, F)
- CNN + BiLSTM + 注意力池化
- 可选拼接感知编码器输出 z 以及全局特征向量
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class AttentionPool(nn.Module):
    """简单可学习上下文向量的注意力池化。"""

    def __init__(self, dim: int):
        super().__init__()
        self.context = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = torch.tanh(x) @ self.context  # (B, T)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = torch.sum(attn * x, dim=1)  # (B, D)
        return pooled


class RobustAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        num_classes: int = 3,
        perceptual_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = AttentionPool(hidden_dim * 2)
        concat_dim = hidden_dim * 2
        if global_dim is not None:
            concat_dim += global_dim
        if perceptual_dim is not None:
            concat_dim += perceptual_dim
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        frame_feat: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
        perceptual_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # frame_feat: (B, T, F)
        x = frame_feat.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)  # (B, 64, T)
        x = x.transpose(1, 2)  # (B, T, 64)
        out, _ = self.rnn(x)
        pooled = self.attn(out)
        parts = [pooled]
        if global_feat is not None:
            parts.append(global_feat)
        if perceptual_feat is not None:
            parts.append(perceptual_feat)
        fused = torch.cat(parts, dim=-1)
        logits = self.head(fused)
        return logits
