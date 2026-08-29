from __future__ import annotations

import torch
from torch import nn

CNN_SIZE = 128
CNN_EMBED = 48
MLP_EMBED = 32
HEAD_DIM = 48


class TinyCNN(nn.Module):
    """Lightweight CPU CNN. Last spatial map is usable for optional CAM."""

    def __init__(self, embed_dim: int = CNN_EMBED):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 40, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 56, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(56, 72, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(72, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv4(self.stem(x))
        pooled = self.pool(h).flatten(1)
        return self.proj(pooled)


class QualityMLP(nn.Module):
    """Feature-only baseline (ablation)."""

    def __init__(self, in_dim: int, n_issues: int = 6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, MLP_EMBED),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(MLP_EMBED, 1)
        self.issue_head = nn.Linear(MLP_EMBED, n_issues)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        score = torch.sigmoid(self.score_head(h)) * 100.0
        issues = torch.sigmoid(self.issue_head(h))
        return score.squeeze(-1), issues


class CnnOnly(nn.Module):
    """Pixel-only baseline (ablation)."""

    def __init__(self, n_issues: int = 6):
        super().__init__()
        self.cnn = TinyCNN()
        self.score_head = nn.Linear(CNN_EMBED, 1)
        self.issue_head = nn.Linear(CNN_EMBED, n_issues)

    def forward(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.cnn(x_img)
        score = torch.sigmoid(self.score_head(h)) * 100.0
        issues = torch.sigmoid(self.issue_head(h))
        return score.squeeze(-1), issues


class QualityHybrid(nn.Module):
    """CNN embedding concatenated with CV-feature MLP, then joint heads."""

    def __init__(self, in_dim: int, n_issues: int = 6):
        super().__init__()
        self.cnn = TinyCNN()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, MLP_EMBED),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(CNN_EMBED + MLP_EMBED, HEAD_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.score_head = nn.Linear(HEAD_DIM, 1)
        self.issue_head = nn.Linear(HEAD_DIM, n_issues)
        self.issue_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x_feat: torch.Tensor, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse(torch.cat([self.cnn(x_img), self.mlp(x_feat)], dim=1))
        score = torch.sigmoid(self.score_head(fused)) * 100.0
        logits = self.issue_head(fused) / self.issue_temperature.clamp(min=0.25)
        issues = torch.sigmoid(logits)
        return score.squeeze(-1), issues
