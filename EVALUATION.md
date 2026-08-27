# Evaluation

## Task

Multi-label issue detection (`blur`, `underexposure`, `overexposure`, `noise`, `corruption`, `defect`) plus regression of a 0–100 `quality_score`. Verdict labels `ACCEPTABLE` / `DEGRADED` / `DEFECTIVE` are **fused from score + issue confidence**, not trained as a third head.

## Model

Small CPU `QualityMLP` in PyTorch: Linear(12→64) → ReLU → Dropout → Linear(64→32) → ReLU, then a sigmoid score head (×100) and a 6-way sigmoid issue head. Inputs are engineered CV features, not pixels — this keeps explainability and matches the heatmap tiles.

## Features (same code for train and serve)

| Name | What it measures |
|------|------------------|
| `sharpness_laplacian` | Variance of Laplacian (low → blur) |
| `sharpness_tenengrad` | Mean squared Sobel energy |
| `brightness_mean` | Mean luma in [0, 1] |
| `dark_frac` / `bright_frac` | Share of near-black / near-white pixels |
| `contrast_std` | Luma standard deviation |
| `saturation_mean` | HSV saturation |
| `noise_mad` | MAD of high-frequency residual after 3×3 blur |
| `entropy` | Normalized luma histogram entropy |
| `jpeg_blockiness` | 8×8 block-boundary energy vs interior |
| `median_residual` | p95 residual after 15×15 median filter (stains/scratches) |
| `defect_peak` / `defect_frac` | Tile median-residual outliers vs the grid |

A **16×16** grid uses the identical per-tile formulas. Heatmaps are those tiles, upsampled. Global sharpness/brightness/noise reported in the API are the full-image versions of the same metrics.

## Data generation

Public DIV2K/Kodak downloads are not required for a reproducible demo. Training images are **procedural civic stills** (sky gradient, road, buildings, vehicles) from `ml/degrade.py` with seed 42.

Degradations and issue labels:

- Gaussian / motion blur → `blur`
- Additive brightness down / up → `underexposure` / `overexposure`
- Gaussian noise → `noise`
- JPEG quality 3–12 → `corruption`
- Ellipse stain, scratch, or punched hole → `defect`
- 15% of train samples receive **two** degradations

`quality_score` targets are a clipped penalty sum from the applied issues (clean ≈ 86–98).

**Holdout:** 160 images, RNG seed 141, **stronger** kernels / noise / JPEG than train. Images are not reused from the train generator seed.

## Scaling

StandardScaler mean/std fit on **train features only**, saved as `ml/artifacts/scaler.json`.

## Metrics

Holdout from `python ml/train.py` (n=160, unseen seed, stronger degradations). Also stored in `ml/artifacts/metrics.json`.

| Head | Precision | Recall | F1 |
|------|-----------|--------|-----|
| blur | 1.00 | 1.00 | 1.00 |
| underexposure | 1.00 | 1.00 | 1.00 |
| overexposure | 1.00 | 1.00 | 1.00 |
| noise | 1.00 | 1.00 | 1.00 |
| corruption | 1.00 | 1.00 | 1.00 |
| defect | 0.88 | 0.94 | 0.91 |

Quality score: MAE **5.82**, RMSE **7.47**. Perfect issue F1 on synthetic single-factor degradations is expected — the features were designed for those factors. Defect is the hardest head (4 false positives, 2 misses). Score error remains because the regression target is a heuristic penalty, not a human MOS. Re-run training on your machine to refresh `metrics.json`.

## Fusion cutoffs (serving)

- Issue listed if probability ≥ 0.42; severity low / medium / high at 0.42 / 0.58 / 0.75.
- `DEFECTIVE` if high-confidence corruption (≥ 0.65) or defect (≥ 0.70), or score < 40 with a high-severity issue.
- `ACCEPTABLE` if score ≥ 70 and no high-severity issues.
- Else `DEGRADED`.

## Heatmap alignment

Blur overlay is 1 − (tile Laplacian / p95 Laplacian). Exposure overlay is dark-fraction (blue) vs bright-fraction (amber). Noise overlay is min-max normalized tile MAD. Defect overlay is luma anomaly. There is **no corruption heatmap**; corruption is treated as a global decode/JPEG property.

## Failure cases and limits

- Procedural streets are not real CCTV. The model can overfit to synthetic statistics (uniform asphalt, cartoon buildings).
- Global blur vs a sharp subject on a blurred background can disagree with a single score.
- Night IR, rain on glass, and flare are not dedicated classes (optional extras were left out of v1).
- JPEG smash is a proxy for “corruption,” not bit-flipped files. Completely undecodable uploads return HTTP 400 instead of a label.
- Defect F1 is the weakest head; small stains on busy texture hide in the tile z-score.
- Uncertain predictions sit near 0.42–0.55 confidence; the UI still shows them as low severity so operators can override.

## Incorrect predictions you should expect

- Motion blur on a tiny moving object: global Laplacian stays high → under-called `blur`.
- Heavy underexposure + noise: noise MAD rises in crushed shadows → extra `noise` flag.
- Clean high-contrast night scene: dark_frac high → possible false `underexposure`.

These are acceptable to document; they show the feature reasoning rather than hiding it.
