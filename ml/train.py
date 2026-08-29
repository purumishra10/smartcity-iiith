#!/usr/bin/env python3
"""Train QualityHybrid (CNN + CV MLP) on synthetic civic degradations. CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from app.vision.cnn_input import bgr_to_cnn_tensor  # noqa: E402
from app.vision.features import (  # noqa: E402
    FEATURE_NAMES,
    ISSUE_TYPES,
    bgr_to_working,
    extract_global_features,
)
from app.vision.model import CnnOnly, QualityHybrid, QualityMLP  # noqa: E402
from degrade import ISSUE_FROM_DEG, apply_named, make_civic_scene, score_from_issues  # noqa: E402

SEED = 42
ARTIFACTS = ROOT / "ml" / "artifacts"
PUBLIC_DIR = ROOT / "ml" / "data" / "public"
TRAIN_GRID = 8
N_TRAIN = 2000
N_HOLDOUT = 400

DEG_NAMES = [
    "clean",
    "blur",
    "motion",
    "under",
    "over",
    "noise",
    "poisson",
    "corrupt",
    "defect",
    "uneven",
    "rain",
    "fog",
]


def load_public_stills() -> list[np.ndarray]:
    stills: list[np.ndarray] = []
    if not PUBLIC_DIR.exists():
        return stills
    for path in sorted(PUBLIC_DIR.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            stills.append(img)
    return stills


def download_public_stills() -> None:
    """Best-effort Kodak subset. Offline training still works without this."""
    import urllib.request

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    has_images = any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in PUBLIC_DIR.glob("*"))
    if has_images:
        return
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "dr-image/1.0")]
    urllib.request.install_opener(opener)
    for i in range(1, 9):
        url = f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
        dest = PUBLIC_DIR / f"kodim{i:02d}.png"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                dest.write_bytes(resp.read())
        except Exception:
            if dest.exists():
                dest.unlink()
            continue


def make_base(rng: np.random.Generator, public: list[np.ndarray], size: int = 256) -> np.ndarray:
    if public and rng.random() < 0.35:
        src = public[int(rng.integers(0, len(public)))]
        return cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    return make_civic_scene(rng, size=size)


def build_split(
    n: int,
    seed: int,
    strength: str,
    public: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, scores, ys, imgs = [], [], [], []
    mix_p = 0.28 if strength == "train" else 0.22
    for i in range(n):
        img = make_base(rng, public, size=256)
        if rng.random() < mix_p:
            a, b = rng.choice([name for name in DEG_NAMES if name != "clean"], size=2, replace=False)
            img = apply_named(apply_named(img, str(a), rng, strength), str(b), rng, strength)
            issues = list(dict.fromkeys(ISSUE_FROM_DEG[str(a)] + ISSUE_FROM_DEG[str(b)]))
        else:
            name = DEG_NAMES[i % len(DEG_NAMES)]
            img = apply_named(img, name, rng, strength)
            issues = list(ISSUE_FROM_DEG[name])
        working = bgr_to_working(img, 256)
        vec, _, _ = extract_global_features(working, grid_size=TRAIN_GRID)
        xs.append(vec)
        scores.append(score_from_issues(issues, rng))
        y = np.zeros(len(ISSUE_TYPES), dtype=np.float32)
        for iss in issues:
            y[ISSUE_TYPES.index(iss)] = 1.0
        ys.append(y)
        imgs.append(bgr_to_cnn_tensor(working).numpy())
        if (i + 1) % 200 == 0:
            print(f"  {strength} samples {i + 1}/{n}", flush=True)
    return (
        np.stack(xs),
        np.array(scores, dtype=np.float32),
        np.stack(ys),
        np.stack(imgs).astype(np.float32),
    )


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
        tn = int(np.sum((~yt) & (~yp)))
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


def fit_loop(model, batches, xv_args, sv, yv, epochs: int, issue_w: torch.Tensor, lr: float = 1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best = 1e9
    best_state = None
    patience = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in batches():
            opt.zero_grad()
            if len(batch) == 4:
                xb, ib, sb, yb = batch
                ps, pi = model(xb, ib)
            else:
                xb, sb, yb = batch
                ps, pi = model(xb)
            bce = -(yb * torch.log(pi.clamp(1e-6, 1)) + (1 - yb) * torch.log((1 - pi).clamp(1e-6, 1)))
            bce = (bce * issue_w).mean()
            loss = mse(ps, sb) / 80.0 + bce
            loss.backward()
            opt.step()
            total += float(loss.item())
        model.eval()
        with torch.no_grad():
            if len(xv_args) == 2:
                ps, pi = model(xv_args[0], xv_args[1])
            else:
                ps, pi = model(xv_args[0])
            val = float(
                mse(ps, sv) / 80.0
                + (
                    -(yv * torch.log(pi.clamp(1e-6, 1)) + (1 - yv) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w
                ).mean()
            )
        if val < best - 1e-4:
            best = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == epochs - 1:
            n_batches = max(len(list(batches())) if False else 1, 1)
            print(f"epoch {epoch:02d} val={val:.4f} best={best:.4f}")
        if patience >= 8:
            print(f"early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model


def calibrate_temperature(model: QualityHybrid, xv, iv, yv) -> None:
    """Fit a single temperature on holdout issue probabilities."""
    model.eval()
    temps = torch.tensor([0.6, 0.8, 1.0, 1.25, 1.5, 1.8])
    best_t, best_nll = 1.0, 1e9
    with torch.no_grad():
        fused_base_temp = float(model.issue_temperature.item())
        for t in temps:
            model.issue_temperature.copy_(torch.tensor([float(t) * fused_base_temp]))
            _, pi = model(xv, iv)
            nll = float(
                -(yv * torch.log(pi.clamp(1e-6, 1)) + (1 - yv) * torch.log((1 - pi).clamp(1e-6, 1))).mean()
            )
            if nll < best_nll:
                best_nll = nll
                best_t = float(model.issue_temperature.item())
    model.issue_temperature.data.fill_(best_t)
    print(f"calibrated issue temperature={best_t:.3f} nll={best_nll:.4f}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Trying optional public stills (Kodak)...", flush=True)
    try:
        download_public_stills()
    except Exception as exc:  # noqa: BLE001
        print(f"public download skipped: {exc}")
    public = load_public_stills()
    print(f"public stills available: {len(public)}", flush=True)

    print("Building train set...", flush=True)
    x_train, s_train, y_train, i_train = build_split(N_TRAIN, SEED, "train", public)
    print("Building holdout (unseen seed + stronger degradations)...", flush=True)
    x_val, s_val, y_val, i_val = build_split(N_HOLDOUT, SEED + 141, "holdout", public)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "names": FEATURE_NAMES}
    (ARTIFACTS / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    def z(x):
        return (x - mean) / std

    xt = torch.from_numpy(z(x_train).astype(np.float32))
    xv = torch.from_numpy(z(x_val).astype(np.float32))
    it = torch.from_numpy(i_train)
    iv = torch.from_numpy(i_val)
    st = torch.from_numpy(s_train)
    sv = torch.from_numpy(s_val)
    yt = torch.from_numpy(y_train)
    yv = torch.from_numpy(y_val)

    issue_w = torch.tensor([1.0, 1.1, 1.1, 1.2, 2.2, 3.0])
    hybrid_ds = TensorDataset(xt, it, st, yt)
    mlp_ds = TensorDataset(xt, st, yt)
    cnn_ds = TensorDataset(it, st, yt)

    def hybrid_batches():
        return DataLoader(hybrid_ds, batch_size=32, shuffle=True)

    def mlp_batches():
        loader = DataLoader(mlp_ds, batch_size=64, shuffle=True)
        for xb, sb, yb in loader:
            yield xb, sb, yb

    def cnn_batches():
        loader = DataLoader(cnn_ds, batch_size=32, shuffle=True)
        for ib, sb, yb in loader:
            yield ib, sb, yb

    print("Training fused hybrid...")
    hybrid = QualityHybrid(in_dim=len(FEATURE_NAMES), n_issues=len(ISSUE_TYPES))
    fit_loop(hybrid, hybrid_batches, (xv, iv), sv, yv, epochs=28, issue_w=issue_w)
    calibrate_temperature(hybrid, xv, iv, yv)
    torch.save({"kind": "hybrid", "state": hybrid.state_dict()}, ARTIFACTS / "model.pt")

    print("Training MLP ablation...")
    mlp = QualityMLP(in_dim=len(FEATURE_NAMES), n_issues=len(ISSUE_TYPES))

    def mlp_fit_batches():
        return DataLoader(mlp_ds, batch_size=64, shuffle=True)

    # reuse fit_loop for mlp: batch is xb, sb, yb - need to wrap
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    best, best_state, patience = 1e9, None, 0
    for epoch in range(40):
        mlp.train()
        for xb, sb, yb in DataLoader(mlp_ds, batch_size=64, shuffle=True):
            opt.zero_grad()
            ps, pi = mlp(xb)
            bce = (-(yb * torch.log(pi.clamp(1e-6, 1)) + (1 - yb) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w).mean()
            loss = mse(ps, sb) / 80.0 + bce
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            ps, pi = mlp(xv)
            val = float(mse(ps, sv) / 80.0 + ((-(yv * torch.log(pi.clamp(1e-6, 1)) + (1 - yv) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w).mean()))
        if val < best:
            best, best_state, patience = val, {k: v.cpu().clone() for k, v in mlp.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 8:
            break
    mlp.load_state_dict(best_state)

    print("Training CNN ablation...")
    cnn = CnnOnly(n_issues=len(ISSUE_TYPES))
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(20):
        cnn.train()
        for ib, sb, yb in DataLoader(cnn_ds, batch_size=32, shuffle=True):
            opt.zero_grad()
            ps, pi = cnn(ib)
            bce = (-(yb * torch.log(pi.clamp(1e-6, 1)) + (1 - yb) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w).mean()
            loss = mse(ps, sb) / 80.0 + bce
            loss.backward()
            opt.step()
        cnn.eval()
        with torch.no_grad():
            ps, pi = cnn(iv)
            val = float(mse(ps, sv) / 80.0 + ((-(yv * torch.log(pi.clamp(1e-6, 1)) + (1 - yv) * torch.log((1 - pi).clamp(1e-6, 1))) * issue_w).mean()))
        if val < best:
            best, best_state, patience = val, {k: v.cpu().clone() for k, v in cnn.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 7:
            break
    cnn.load_state_dict(best_state)

    hybrid.eval()
    mlp.eval()
    cnn.eval()
    with torch.no_grad():
        hs, hi = hybrid(xv, iv)
        ms, mi = mlp(xv)
        cs, ci = cnn(iv)

    fused = metrics_report(s_val, hs.numpy(), y_val, hi.numpy())
    fused["holdout"] = {
        "n": int(len(s_val)),
        "protocol": "Unseen RNG seed, stronger degradations, mixed issue pairs, optional Kodak stills.",
        "public_stills": len(public),
        "train_n": N_TRAIN,
    }
    fused["feature_names"] = FEATURE_NAMES
    fused["ablation"] = {
        "fused": {"score_mae": fused["score_mae"], "score_rmse": fused["score_rmse"], "issues": fused["issues"]},
        "mlp_features_only": metrics_report(s_val, ms.numpy(), y_val, mi.numpy()),
        "cnn_pixels_only": metrics_report(s_val, cs.numpy(), y_val, ci.numpy()),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(fused, indent=2), encoding="utf-8")
    print(json.dumps(fused, indent=2))
    print(f"Wrote {ARTIFACTS / 'model.pt'}")


if __name__ == "__main__":
    main()
