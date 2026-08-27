#!/usr/bin/env python3
"""Write civic sample stills for each required quality condition."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from degrade import apply_named, make_civic_scene, to_jpeg_bytes  # noqa: E402

OUT = ROOT / "sample_images"


def main() -> None:
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("00_clean.jpg", "clean"),
        ("01_blur.jpg", "blur"),
        ("02_underexposure.jpg", "under"),
        ("03_overexposure.jpg", "over"),
        ("04_noise.jpg", "noise"),
        ("05_corrupt.jpg", "corrupt"),
        ("06_defect.jpg", "defect"),
        ("07_motion_blur.jpg", "motion"),
    ]
    for filename, name in mapping:
        img = make_civic_scene(rng, size=384)
        img = apply_named(img, name, rng, strength="holdout")
        path = OUT / filename
        if name == "corrupt":
            path.write_bytes(to_jpeg_bytes(img, quality=8))
        else:
            cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(path)


if __name__ == "__main__":
    main()
