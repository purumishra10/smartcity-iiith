from __future__ import annotations

import io
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def make_civic_scene(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Procedural street-like still (sky, road, buildings, vehicles)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sky_top = np.array([210, 160, 90], dtype=np.float32)
    sky_bot = np.array([245, 210, 170], dtype=np.float32)
    for y in range(size // 2):
        t = y / max(size // 2 - 1, 1)
        img[y, :] = (sky_top * (1 - t) + sky_bot * t).astype(np.uint8)

    horizon = size // 2
    img[horizon:, :] = (42, 42, 48)
    cv2.fillConvexPoly(
        img,
        np.array(
            [[0, size - 1], [size - 1, size - 1], [int(size * 0.62), horizon + 8], [int(size * 0.38), horizon + 8]],
            dtype=np.int32,
        ),
        (70, 70, 76),
    )
    cv2.line(img, (size // 2, horizon + 10), (size // 2, size - 1), (220, 220, 230), 2)

    for i in range(4):
        x = 8 + i * (size // 5)
        w = rng.integers(28, 50)
        h = rng.integers(40, 90)
        color = (
            int(rng.integers(40, 90)),
            int(rng.integers(50, 100)),
            int(rng.integers(70, 130)),
        )
        cv2.rectangle(img, (x, horizon - h), (x + w, horizon + 4), color, -1)
        for wy in range(horizon - h + 8, horizon - 8, 14):
            for wx in range(x + 4, x + w - 6, 10):
                cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), (180, 200, 220), -1)

    for _ in range(int(rng.integers(2, 5))):
        cx = int(rng.integers(40, size - 40))
        cy = int(rng.integers(horizon + 20, size - 20))
        color = (
            int(rng.integers(20, 80)),
            int(rng.integers(40, 160)),
            int(rng.integers(80, 220)),
        )
        cv2.rectangle(img, (cx, cy), (cx + 28, cy + 14), color, -1)

    return img


def gaussian_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img: np.ndarray, length: int) -> np.ndarray:
    length = max(5, length | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, kernel)


def adjust_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_smash(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_defect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    kind = int(rng.integers(0, 3))
    if kind == 0:
        x0, y0 = int(rng.integers(w // 8, w // 2)), int(rng.integers(h // 8, h // 2))
        axes = (int(rng.integers(28, 70)), int(rng.integers(22, 55)))
        stain = (int(rng.integers(0, 40)), int(rng.integers(0, 70)), int(rng.integers(180, 255)))
        cv2.ellipse(out, (x0, y0), axes, 0, 0, 360, stain, -1)
    elif kind == 1:
        pt1 = (int(rng.integers(0, w // 3)), int(rng.integers(0, h)))
        pt2 = (int(rng.integers(2 * w // 3, w)), int(rng.integers(0, h)))
        cv2.line(out, pt1, pt2, (8, 8, 8), int(rng.integers(6, 12)))
    else:
        x, y = int(rng.integers(20, w - 90)), int(rng.integers(20, h - 70))
        out[y : y + 55, x : x + 80] = (int(rng.integers(0, 15)),) * 3
    return out


def add_poisson_noise(img: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    lam = np.clip(x * scale, 1e-4, None)
    noisy = rng.poisson(lam).astype(np.float32) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def uneven_illumination(img: np.ndarray, rng: np.random.Generator, strong: bool) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = float(rng.uniform(0.2, 0.8) * h)
    cx = float(rng.uniform(0.2, 0.8) * w)
    sigma = float(rng.uniform(0.25, 0.55) * max(h, w))
    falloff = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    gain = 0.35 if strong else 0.55
    shade = (gain + (1.0 - gain) * falloff)[:, :, None]
    return np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)


def rain_speckle(img: np.ndarray, rng: np.random.Generator, strong: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    n = int(rng.integers(80 if not strong else 140, 220 if strong else 160))
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(6, 18))
        cv2.line(out, (x, y), (min(w - 1, x + 2), min(h - 1, y + length)), (210, 210, 220), 1)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return add_gaussian_noise(out, 8 if not strong else 14, rng)


def add_fog(img: np.ndarray, rng: np.random.Generator, strong: bool) -> np.ndarray:
    beta = float(rng.uniform(0.55, 0.78 if strong else 0.68))
    haze = np.array([210, 205, 200], dtype=np.float32)
    blended = img.astype(np.float32) * (1.0 - beta) + haze * beta
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_named(img: np.ndarray, name: str, rng: np.random.Generator, strength: str = "train") -> np.ndarray:
    strong = strength == "holdout"
    if name == "clean":
        return img
    if name == "blur":
        return gaussian_blur(img, int(rng.integers(15, 31 if strong else 23)))
    if name == "motion":
        return motion_blur(img, int(rng.integers(11, 25 if strong else 19)))
    if name == "under":
        return adjust_brightness(img, float(rng.uniform(-140, -90 if strong else -70)))
    if name == "over":
        return adjust_brightness(img, float(rng.uniform(90 if not strong else 110, 160)))
    if name == "noise":
        return add_gaussian_noise(img, float(rng.uniform(28, 55 if strong else 42)), rng)
    if name == "poisson":
        return add_poisson_noise(img, rng, float(rng.uniform(18, 32 if strong else 26)))
    if name == "corrupt":
        return jpeg_smash(img, int(rng.integers(3, 8 if strong else 14)))
    if name == "defect":
        return add_defect(img, rng)
    if name == "uneven":
        return uneven_illumination(img, rng, strong)
    if name == "rain":
        return rain_speckle(img, rng, strong)
    if name == "fog":
        return add_fog(img, rng, strong)
    raise ValueError(name)


ISSUE_FROM_DEG = {
    "clean": [],
    "blur": ["blur"],
    "motion": ["blur"],
    "under": ["underexposure"],
    "over": ["overexposure"],
    "noise": ["noise"],
    "poisson": ["noise"],
    "corrupt": ["corruption"],
    "defect": ["defect"],
    "uneven": ["underexposure"],
    "rain": ["noise"],
    "fog": ["blur"],
}


def score_from_issues(issues: Sequence[str], rng: np.random.Generator) -> float:
    if not issues:
        return float(rng.uniform(86, 98))
    penalty = {
        "blur": rng.uniform(28, 42),
        "underexposure": rng.uniform(22, 36),
        "overexposure": rng.uniform(22, 36),
        "noise": rng.uniform(18, 32),
        "corruption": rng.uniform(40, 55),
        "defect": rng.uniform(38, 52),
    }
    s = 100.0 - sum(penalty[i] for i in issues)
    return float(np.clip(s, 5, 95))


def to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
