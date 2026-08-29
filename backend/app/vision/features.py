"""Shared image-quality features for training and serving.

Global statistics and tile maps use the same per-region metrics so heatmaps
cannot contradict the clinic verdict.
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = [
    "sharpness_laplacian",
    "sharpness_tenengrad",
    "brightness_mean",
    "dark_frac",
    "bright_frac",
    "contrast_std",
    "saturation_mean",
    "noise_mad",
    "entropy",
    "jpeg_blockiness",
    "median_residual",
    "defect_peak",
    "defect_frac",
    "fft_high_ratio",
    "mscn_var",
    "clahe_delta",
    "color_cast",
    "glare_peak",
]

ISSUE_TYPES = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
]

MAX_ANALYSIS_DIM = 512


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unreadable_image")
    return bgr


def bgr_to_working(bgr: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def _luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _saturation(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].astype(np.float32) / 255.0


def region_metrics(bgr: np.ndarray) -> dict[str, float]:
    """Metrics for a full image or a single tile (same formulas)."""
    gray = _luma(bgr)
    if gray.size < 16:
        return {
            "sharpness_laplacian": 0.0,
            "sharpness_tenengrad": 0.0,
            "brightness_mean": float(gray.mean() / 255.0) if gray.size else 0.0,
            "dark_frac": 0.0,
            "bright_frac": 0.0,
            "contrast_std": 0.0,
            "saturation_mean": 0.0,
            "noise_mad": 0.0,
            "entropy": 0.0,
            "jpeg_blockiness": 0.0,
            "median_residual": 0.0,
        }

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    brightness = float(gray.mean() / 255.0)
    dark_frac = float(np.mean(gray < 25.0))
    bright_frac = float(np.mean(gray > 230.0))
    contrast = float(gray.std() / 255.0)
    sat = float(_saturation(bgr).mean())

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - blurred
    noise_mad = float(np.median(np.abs(residual - np.median(residual))) / 255.0)

    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [64], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum() / 6.0)

    blockiness = _jpeg_blockiness(gray)
    med = cv2.medianBlur(gray.astype(np.uint8), 15).astype(np.float32)
    residual_p95 = float(np.percentile(np.abs(gray - med), 95) / 255.0)

    return {
        "sharpness_laplacian": float(lap.var() / (255.0**2)),
        "sharpness_tenengrad": tenengrad / (255.0**2),
        "brightness_mean": brightness,
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "contrast_std": contrast,
        "saturation_mean": sat,
        "noise_mad": noise_mad,
        "entropy": entropy,
        "jpeg_blockiness": blockiness,
        "median_residual": residual_p95,
    }


def _jpeg_blockiness(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    dh = np.abs(np.diff(gray, axis=1))
    dv = np.abs(np.diff(gray, axis=0))
    v_bounds = float(dh[:, 7::8].mean()) if dh.shape[1] > 8 else 0.0
    h_bounds = float(dv[7::8, :].mean()) if dv.shape[0] > 8 else 0.0
    # Interior (non-boundary) steps
    v_mask = np.ones(dh.shape[1], dtype=bool)
    v_mask[7::8] = False
    h_mask = np.ones(dv.shape[0], dtype=bool)
    h_mask[7::8] = False
    v_int = float(dh[:, v_mask].mean()) if v_mask.any() else 1e-6
    h_int = float(dv[h_mask, :].mean()) if h_mask.any() else 1e-6
    return float(((v_bounds / (v_int + 1e-6)) + (h_bounds / (h_int + 1e-6))) / 2.0)


def extra_global_metrics(bgr: np.ndarray) -> dict[str, float]:
    """Full-image signals (too noisy / too slow to run on every tile)."""
    gray = _luma(bgr)
    h, w = gray.shape
    if h < 16 or w < 16:
        return {
            "fft_high_ratio": 0.0,
            "mscn_var": 0.0,
            "clahe_delta": 0.0,
            "color_cast": 0.0,
            "glare_peak": 0.0,
        }

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = max(min(cy, cx), 1)
    high = float(spectrum[radius > 0.35 * rmax].mean()) if np.any(radius > 0.35 * rmax) else 0.0
    low = float(spectrum[radius <= 0.15 * rmax].mean()) if np.any(radius <= 0.15 * rmax) else 1e-6
    fft_high_ratio = float(np.clip(high / (low + 1e-6), 0.0, 20.0) / 20.0)

    mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
    sigma = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (7, 7), 1.166) - mu * mu, 0))
    mscn = (gray - mu) / (sigma + 1.0)
    mscn_var = float(np.clip(mscn.var() / 8.0, 0.0, 1.0))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray.astype(np.uint8)).astype(np.float32)
    clahe_delta = float(np.mean(np.abs(equalized - gray)) / 255.0)

    channels = bgr.astype(np.float32)
    means = channels.reshape(-1, 3).mean(axis=0)
    gray_mean = float(means.mean())
    color_cast = float(np.max(np.abs(means - gray_mean)) / 255.0)

    sat = _saturation(bgr)
    glare_peak = float(np.mean((gray > 240) & (sat < 0.18)))

    return {
        "fft_high_ratio": fft_high_ratio,
        "mscn_var": mscn_var,
        "clahe_delta": clahe_delta,
        "color_cast": color_cast,
        "glare_peak": glare_peak,
    }


def extract_tile_maps(bgr: np.ndarray, grid_size: int = 16) -> dict:
    h, w = bgr.shape[:2]
    rows, cols = grid_size, grid_size
    tile_h = h / rows
    tile_w = w / cols

    sharpness = np.zeros((rows, cols), dtype=np.float32)
    dark = np.zeros((rows, cols), dtype=np.float32)
    bright = np.zeros((rows, cols), dtype=np.float32)
    noise = np.zeros((rows, cols), dtype=np.float32)
    residual = np.zeros((rows, cols), dtype=np.float32)
    tiles: list[dict[str, float]] = []

    for r in range(rows):
        for c in range(cols):
            y0 = int(r * tile_h)
            y1 = int((r + 1) * tile_h) if r < rows - 1 else h
            x0 = int(c * tile_w)
            x1 = int((c + 1) * tile_w) if c < cols - 1 else w
            patch = bgr[y0:y1, x0:x1]
            m = region_metrics(patch)
            sharpness[r, c] = m["sharpness_laplacian"]
            dark[r, c] = m["dark_frac"]
            bright[r, c] = m["bright_frac"]
            noise[r, c] = m["noise_mad"]
            residual[r, c] = m["median_residual"]
            tiles.append(
                {
                    "blur": 0.0,
                    "underexposure": float(m["dark_frac"]),
                    "overexposure": float(m["bright_frac"]),
                    "noise": float(m["noise_mad"]),
                    "defect": 0.0,
                    "sharpness": float(m["sharpness_laplacian"]),
                    "brightness": float(m["brightness_mean"]),
                }
            )

    # Blur = inverted sharpness relative to this image (aligned with global low sharpness).
    p95 = float(np.percentile(sharpness, 95) + 1e-8)
    blur = np.clip(1.0 - sharpness / p95, 0.0, 1.0)

    # Defect = tiles whose median-filter residual is an outlier vs the grid.
    r_med = float(np.median(residual) + 1e-6)
    defect = np.clip((residual - r_med) / (np.percentile(residual, 75) + 1e-6), 0.0, 1.0)

    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        t["blur"] = float(blur[r, c])
        t["defect"] = float(defect[r, c])

    return {
        "rows": rows,
        "cols": cols,
        "blur": blur,
        "underexposure": np.clip(dark, 0, 1),
        "overexposure": np.clip(bright, 0, 1),
        "noise": _normalize_map(noise),
        "defect": defect,
        "tiles": tiles,
    }


def _normalize_map(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-8:
        return np.zeros_like(m)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def extract_global_features(bgr: np.ndarray, grid_size: int = 16) -> tuple[np.ndarray, dict, dict]:
    maps = extract_tile_maps(bgr, grid_size=grid_size)
    m = region_metrics(bgr)
    extra = extra_global_metrics(bgr)

    defect_peak = float(maps["defect"].max())
    defect_frac = float(np.mean(maps["defect"] > 0.45))

    named = {
        **m,
        "defect_peak": defect_peak,
        "defect_frac": defect_frac,
        **extra,
    }
    vec = np.array([named[k] for k in FEATURE_NAMES], dtype=np.float32)

    stats = {
        "sharpness": round(m["sharpness_laplacian"], 4),
        "brightness": round(m["brightness_mean"], 4),
        "contrast": round(m["contrast_std"], 4),
        "noise_estimate": round(m["noise_mad"], 4),
        "saturation": round(m["saturation_mean"], 4),
        "blockiness": round(m["jpeg_blockiness"], 4),
        "glare": round(extra["glare_peak"], 4),
        "color_cast": round(extra["color_cast"], 4),
    }
    return vec, stats, maps
