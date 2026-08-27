from __future__ import annotations

import torch
from torch import nn


class QualityMLP(nn.Module):
    def __init__(self, in_dim: int = 13, n_issues: int = 6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(32, 1)
        self.issue_head = nn.Linear(32, n_issues)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        score = torch.sigmoid(self.score_head(h)) * 100.0
        issues = torch.sigmoid(self.issue_head(h))
        return score.squeeze(-1), issues
