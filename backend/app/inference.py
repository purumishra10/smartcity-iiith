from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.vision.cnn_input import bgr_to_cnn_tensor
from app.vision.features import FEATURE_NAMES, ISSUE_TYPES
from app.vision.model import QualityHybrid, QualityMLP


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @classmethod
    def from_json(cls, path: Path) -> "StandardScaler":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(data["mean"], dtype=np.float32), np.array(data["std"], dtype=np.float32))


class QualityEngine:
    def __init__(self, model_path: str, scaler_path: str):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.ready = False
        self.error: str | None = None
        self.kind = "hybrid"
        self.device = torch.device("cpu")
        self.model = None
        self.scaler: StandardScaler | None = None
        self.load()

    def load(self) -> None:
        try:
            if not self.model_path.exists() or not self.scaler_path.exists():
                self.error = "model_or_scaler_missing"
                self.ready = False
                return
            self.scaler = StandardScaler.from_json(self.scaler_path)
            in_dim = len(self.scaler.mean)
            raw = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "kind" in raw:
                self.kind = raw["kind"]
                state = raw["state"]
            else:
                self.kind = "mlp"
                state = raw
            if self.kind == "hybrid":
                self.model = QualityHybrid(in_dim=in_dim, n_issues=len(ISSUE_TYPES))
            else:
                self.model = QualityMLP(in_dim=in_dim, n_issues=len(ISSUE_TYPES))
            self.model.load_state_dict(state)
            self.model.eval()
            self.ready = True
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.ready = False
            self.error = str(exc)

    def predict(self, features: np.ndarray, bgr=None) -> tuple[float, dict[str, float]]:
        if not self.ready or self.model is None or self.scaler is None:
            raise RuntimeError("model_not_loaded")
        x = self.scaler.transform(features.reshape(1, -1))
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            if self.kind == "hybrid":
                if bgr is None:
                    raise RuntimeError("hybrid_requires_image")
                img = bgr_to_cnn_tensor(bgr).unsqueeze(0)
                score, issues = self.model(xt, img)
            else:
                score, issues = self.model(xt)
        score_v = float(score.cpu().numpy().ravel()[0])
        probs = issues.cpu().numpy().ravel()
        return score_v, {name: float(probs[i]) for i, name in enumerate(ISSUE_TYPES)}
