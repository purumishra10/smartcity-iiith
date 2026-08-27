#!/usr/bin/env python3
"""Train QualityMLP on synthetic civic degradations. CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from app.vision.features import (  # noqa: E402
    FEATURE_NAMES,
    ISSUE_TYPES,
    bgr_to_working,
    extract_global_features,
)
from app.vision.model import QualityMLP  # noqa: E402
from degrade import ISSUE_FROM_DEG, apply_named, make_civic_scene, score_from_issues  # noqa: E402

SEED = 42
ARTIFACTS = ROOT / "ml" / "artifacts"


def build_split(n: int, seed: int, strength: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = ["clean", "blur", "motion", "under", "over", "noise", "corrupt", "defect", "defect", "corrupt"]
    xs, scores, ys = [], [], []
    for i in range(n):
        img = make_civic_scene(rng, size=256)
        if rng.random() < 0.15 and strength == "train":
            # mild mix of two degradations
            a, b = rng.choice([n for n in names if n != "clean"], size=2, replace=False)
            img = apply_named(apply_named(img, str(a), rng, strength), str(b), rng, strength)
            issues = list(dict.fromkeys(ISSUE_FROM_DEG[str(a)] + ISSUE_FROM_DEG[str(b)]))
        else:
            name = names[i % len(names)]
            img = apply_named(img, name, rng, strength)
            issues = list(ISSUE_FROM_DEG[name])
        working = bgr_to_working(img, 256)
        vec, _, _ = extract_global_features(working, grid_size=16)
        xs.append(vec)
        scores.append(score_from_issues(issues, rng))
        y = np.zeros(len(ISSUE_TYPES), dtype=np.float32)
        for iss in issues:
            y[ISSUE_TYPES.index(iss)] = 1.0
        ys.append(y)
    return np.stack(xs), np.array(scores, dtype=np.float32), np.stack(ys)


def metrics_report(score_true, score_pred, y_true, y_pred, threshold=0.5) -> dict:
    mae = float(np.mean(np.abs(score_true - score_pred)))
    rmse = float(np.sqrt(np.mean((score_true - score_pred) ** 2)))
    per_issue = {}
    for i, name in enumerate(ISSUE_TYPES):
        yt = y_true[:, i] >= 0.5
        yp = y_pred[:, i] >= threshold
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        tn = int(np.sum(~yt & ~yp))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        per_issue[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    return {"score_mae": round(mae, 3), "score_rmse": round(rmse, 3), "issues": per_issue}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Building train features...")
    x_train, s_train, y_train = build_split(720, SEED, "train")
    print("Building holdout (unseen seeds + stronger degradations)...")
    x_val, s_val, y_val = build_split(160, SEED + 99, "holdout")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "names": FEATURE_NAMES}
    (ARTIFACTS / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    def z(x):
        return (x - mean) / std

    xt = torch.from_numpy(z(x_train).astype(np.float32))
    xv = torch.from_numpy(z(x_val).astype(np.float32))
    st = torch.from_numpy(s_train)
    sv = torch.from_numpy(s_val)
    yt = torch.from_numpy(y_train)
    yv = torch.from_numpy(y_val)

    loader = DataLoader(TensorDataset(xt, st, yt), batch_size=64, shuffle=True)
    model = QualityMLP(in_dim=len(FEATURE_NAMES), n_issues=len(ISSUE_TYPES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    issue_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.2, 3.0])

    best = 1e9
    best_state = None
    for epoch in range(80):
        model.train()
        total = 0.0
        for xb, sb, yb in loader:
            opt.zero_grad()
            ps, pi = model(xb)
            bce = -(yb * torch.log(pi.clamp(1e-6, 1)) + (1 - yb) * torch.log((1 - pi).clamp(1e-6, 1)))
            bce = (bce * issue_w).mean()
            loss = mse(ps, sb) / 80.0 + bce
            loss.backward()
            opt.step()
            total += float(loss.item())
        model.eval()
        with torch.no_grad():
            ps, pi = model(xv)
            val = float(mse(ps, sv) / 80.0 + ((-(yv * torch.log(pi.clamp(1e-6, 1)) + (1 - yv) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w).mean()))
        if val < best:
            best = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 79:
            print(f"epoch {epoch:02d} train_loss={total/len(loader):.4f} val={val:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ARTIFACTS / "model.pt")

    model.eval()
    with torch.no_grad():
        ps, pi = model(xv)
    report = metrics_report(s_val, ps.numpy(), y_val, pi.numpy())
    report["holdout"] = {
        "n": int(len(s_val)),
        "protocol": "Unseen RNG seed and stronger degradation ranges than train.",
    }
    report["feature_names"] = FEATURE_NAMES
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {ARTIFACTS / 'model.pt'}")


if __name__ == "__main__":
    main()
