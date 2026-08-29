# Evaluation

## Task

Multi-label issue detection (`blur`, `underexposure`, `overexposure`, `noise`, `corruption`, `defect`) plus regression of a 0–100 `quality_score`. Verdict labels `ACCEPTABLE` / `DEGRADED` / `DEFECTIVE` are **fused from score + issue confidence**, not trained as a third head.

## Model

**Serving model:** `QualityHybrid` (CPU PyTorch).

- Tiny CNN on 128×128 RGB (four conv stages → 48-D embedding).
- MLP on the standardized CV vector (18-D → 32-D embedding).
- Concatenate → 48-D fuse block → sigmoid score (×100) and six issue probabilities.
- A single **temperature** on issue logits is grid-searched on holdout after training.

Ablations (same holdout, stored under `ablation` in `ml/artifacts/metrics.json`):

| Variant | Role |
|---------|------|
| fused CNN+MLP | production |
| MLP on CV features only | interpretability baseline |
| CNN on pixels only | checks that pixels add signal |

Tile heatmaps stay **CV-only** so they cannot contradict the vitals panel. The CNN is not used to paint blur/exposure/noise/defect maps.

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
| `median_residual` | p95 residual after 15×15 median filter |
| `defect_peak` / `defect_frac` | Tile median-residual outliers vs the grid |
| `fft_high_ratio` | High vs low spatial-frequency energy |
| `mscn_var` | Variance of MSCN coefficients (BRISQUE-style) |
| `clahe_delta` | Mean abs difference after CLAHE (haze/flatness) |
| `color_cast` | Max channel mean deviation from grey |
| `glare_peak` | Fraction of bright, low-saturation pixels |

A **16×16** grid (8×8 during training for speed) uses the identical per-tile formulas for the first group. Heatmaps are those tiles, upsampled.

## Data generation

Training images are **procedural civic stills** plus, when the network allows, a few public **Kodak** PNGs in `ml/data/public/` (scripted download; ignored if it fails).

Degradations (`ml/degrade.py`): Gaussian and motion blur, brightness under/over, Gaussian and Poisson noise, JPEG smash, local defects, uneven illumination, rain speckle, fog. About 25% of samples receive **two** degradations.

`quality_score` targets are a clipped penalty sum from the applied issues (clean ≈ 86–98).

**Holdout:** unseen RNG seed, **stronger** kernels / noise / JPEG than train. Re-run `python ml/train.py` to refresh `metrics.json`.

## Scaling

StandardScaler mean/std fit on **train features only**, saved as `ml/artifacts/scaler.json`.

## Metrics

Numbers below are produced by the last `python ml/train.py` run and duplicated in `ml/artifacts/metrics.json`. If you retrain, trust that file over this table.

Fusion cutoffs (serving):

- Issue listed if probability ≥ 0.42; severity low / medium / high at 0.42 / 0.58 / 0.75.
- `DEFECTIVE` if high-confidence corruption (≥ 0.65) or defect (≥ 0.70), or score < 40 with a high-severity issue.
- `ACCEPTABLE` if score ≥ 70 and no high-severity issues.
- Else `DEGRADED`.

## Heatmap alignment

Blur overlay is 1 − (tile Laplacian / p95 Laplacian). Exposure overlay is dark-fraction (blue) vs bright-fraction (amber). Noise overlay is min-max normalized tile MAD. Defect overlay is luma anomaly. There is **no corruption heatmap**; corruption is treated as a global JPEG/decode property.

## Failure cases and limits

- Procedural streets and a handful of Kodak stills are not a municipal CCTV corpus. The fused model still overfits synthetic statistics compared with a large labelled field set.
- Global blur vs a sharp subject on a blurred background can disagree with a single score.
- Night IR, heavy rain on glass, and optical flare are only weakly covered (glare_peak / fog recipes).
- JPEG smash is a proxy for “corruption,” not bit-flipped files. Completely undecodable uploads return HTTP 400.
- Defect remains the hardest head: small stains on busy texture hide in the tile z-score.
- Perfect F1 on *single-factor synthetic* holdout is **not** claimed for the fused model; mixed degradations and public stills are meant to make the numbers honest. Compare the three ablation blocks in `metrics.json`.

## Incorrect predictions you should expect

- Motion blur on a tiny moving object: global Laplacian stays high → under-called `blur`.
- Heavy underexposure + noise: noise MAD rises in crushed shadows → extra `noise` flag.
- Clean high-contrast night scene: dark_frac high → possible false `underexposure`.
- Colour Kodak photos with film grain: CNN may raise `noise` while the operator would accept the grain.
