from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _upsample(tile_map: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(tile_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _overlay(intensity: np.ndarray, color_bgr: tuple[int, int, int], alpha_scale: float = 0.72) -> np.ndarray:
    h, w = intensity.shape
    a = np.clip(intensity, 0, 1)
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, 0] = color_bgr[0]
    bgra[:, :, 1] = color_bgr[1]
    bgra[:, :, 2] = color_bgr[2]
    bgra[:, :, 3] = (a * 255 * alpha_scale).astype(np.uint8)
    return bgra


def exposure_overlay(under: np.ndarray, over: np.ndarray) -> np.ndarray:
    h, w = under.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    u = np.clip(under, 0, 1)
    o = np.clip(over, 0, 1)
    # BGRA: blue = underexposure, amber = overexposure
    bgra[:, :, 0] = (u * 255).astype(np.uint8)
    bgra[:, :, 1] = (o * 170).astype(np.uint8)
    bgra[:, :, 2] = (o * 255).astype(np.uint8)
    bgra[:, :, 3] = (np.maximum(u, o) * 255 * 0.7).astype(np.uint8)
    return bgra


def save_heatmap_overlays(
    maps: dict,
    image_shape: tuple[int, int],
    dest_dir: Path,
) -> dict[str, str]:
    h, w = image_shape[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)

    blur = _upsample(maps["blur"], h, w)
    under = _upsample(maps["underexposure"], h, w)
    over = _upsample(maps["overexposure"], h, w)
    noise = _upsample(maps["noise"], h, w)
    defect = _upsample(maps["defect"], h, w)

    files = {
        "blur": _overlay(blur, (196, 64, 220)),
        "exposure": exposure_overlay(under, over),
        "noise": _overlay(noise, (32, 180, 120)),
        "defect": _overlay(defect, (48, 48, 220)),
    }
    paths: dict[str, str] = {}
    for kind, bgra in files.items():
        path = dest_dir / f"{kind}.png"
        cv2.imwrite(str(path), bgra)
        paths[kind] = str(path)
    return paths
