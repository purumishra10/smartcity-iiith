from __future__ import annotations

import numpy as np

from app.vision.features import FEATURE_NAMES

ISSUE_TYPES = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
]

ACCEPTABLE_MIN = 70.0
DEGRADED_MIN = 40.0
ISSUE_CONF_MIN = 0.42

HEATMAP_FOR_ISSUE = {
    "blur": "blur",
    "underexposure": "exposure",
    "overexposure": "exposure",
    "noise": "noise",
    "defect": "defect",
    "corruption": None,
}

ISSUE_EVIDENCE = {
    "blur": "Low Laplacian / Tenengrad sharpness and reduced high-frequency FFT energy.",
    "underexposure": "Low mean luma and a large dark-pixel fraction (and/or uneven illumination).",
    "overexposure": "High mean luma, clipped highlights, and/or glare on low-saturation regions.",
    "noise": "High MAD of the high-frequency residual and elevated MSCN variance.",
    "corruption": "JPEG 8×8 block-boundary energy much higher than interior gradients.",
    "defect": "Local median-filter residual outliers on the 16×16 tile grid (stain, scratch, hole).",
}


def _severity(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.58:
        return "medium"
    return "low"


def _label(score: float, issues: list[dict]) -> str:
    types = {i["type"] for i in issues}
    high = [i for i in issues if i["severity"] == "high"]
    if "corruption" in types and any(i["type"] == "corruption" and i["confidence"] >= 0.65 for i in issues):
        return "DEFECTIVE"
    if "defect" in types and any(i["type"] == "defect" and i["confidence"] >= 0.7 for i in issues):
        return "DEFECTIVE"
    if any(i["type"] in ("corruption", "defect") and i["severity"] == "high" for i in high):
        return "DEFECTIVE"
    if score >= ACCEPTABLE_MIN and not high:
        return "ACCEPTABLE"
    if score < DEGRADED_MIN and high:
        return "DEFECTIVE"
    return "DEGRADED"


def _pct(value: float, lo: float, hi: float, invert: bool = False) -> int:
    t = (float(value) - lo) / (hi - lo + 1e-8)
    t = max(0.0, min(1.0, t))
    if invert:
        t = 1.0 - t
    return int(round(t * 100))


def vital_explanations(stats: dict) -> list[dict]:
    sharp = float(stats.get("sharpness", 0))
    bright = float(stats.get("brightness", 0))
    contrast = float(stats.get("contrast", 0))
    noise = float(stats.get("noise_estimate", 0))
    sat = float(stats.get("saturation", 0))

    items = [
        {
            "id": "sharpness",
            "label": "Sharpness",
            "raw": sharp,
            "display": _pct(sharp, 0.0, 0.045),
            "higher_is_better": True,
            "meaning": "Variance of the Laplacian on the working copy. Soft / motion-blurred frames score low.",
            "why": (
                f"This still measured Laplacian variance {sharp:.4f}. "
                "Values near 0.03–0.05 are typical of focused civic daylight stills; "
                "below ~0.008 usually means the plate, face, or signage will not read."
            ),
            "action": "Refocus, raise shutter speed, or rest the camera. Avoid uploading interpolated zooms.",
        },
        {
            "id": "brightness",
            "label": "Brightness",
            "raw": bright,
            "display": int(round(bright * 100)),
            "higher_is_better": None,
            "meaning": "Mean luma in [0, 1]. Mid-tones (~0.35–0.65) are most usable for review.",
            "why": (
                f"Mean luma is {bright:.2f}. "
                + (
                    "The frame sits in crushed shadow — under-called detail and noisy blacks are likely."
                    if bright < 0.28
                    else "Highlights are dominating; clipped regions cannot be recovered."
                    if bright > 0.78
                    else "Exposure is in an operator-usable band."
                )
            ),
            "action": "Wait for light, disable auto-gain spikes, or avoid shooting into headlights/sun.",
        },
        {
            "id": "contrast",
            "label": "Contrast",
            "raw": contrast,
            "display": _pct(contrast, 0.0, 0.42),
            "higher_is_better": True,
            "meaning": "Standard deviation of luma. Fog, haze, and heavy compression flatten this.",
            "why": (
                f"Luma std is {contrast:.3f}. "
                + (
                    "Low contrast hides kerbs, lane paint, and clothing edges."
                    if contrast < 0.12
                    else "Contrast is adequate to separate foreground structure."
                )
            ),
            "action": "Avoid fogged glass and over-processed HDR that collapses local contrast.",
        },
        {
            "id": "noise",
            "label": "Noise load",
            "raw": noise,
            "display": _pct(noise, 0.0, 0.08, invert=False),
            "higher_is_better": False,
            "meaning": "Median absolute deviation of the high-frequency residual after a 3×3 blur.",
            "why": (
                f"Noise MAD is {noise:.4f}. "
                + (
                    "Grain is high enough to mimic texture and to confuse defect tiles."
                    if noise > 0.035
                    else "Residual energy is in a normal range for a compressed still."
                )
            ),
            "action": "Use a slower ISO or a longer exposure with support rather than boosting gain.",
        },
        {
            "id": "saturation",
            "label": "Saturation",
            "raw": sat,
            "display": int(round(sat * 100)),
            "higher_is_better": None,
            "meaning": "Mean HSV saturation. Near-zero often means IR, grayscale, or colour-cast night video.",
            "why": (
                f"Mean saturation is {sat:.2f}. "
                + (
                    "The palette is almost grey; colour cues (signals, vests) may be missing."
                    if sat < 0.12
                    else "Colour is present enough for typical civic review."
                )
            ),
            "action": "Prefer colour sensors when clothing or signal lamps matter; check white balance.",
        },
    ]
    return items


def issue_explanations(issues: list[dict], stats: dict) -> list[dict]:
    out = []
    for iss in issues:
        kind = iss["type"]
        out.append(
            {
                "type": kind,
                "severity": iss["severity"],
                "confidence": iss["confidence"],
                "heatmap": HEATMAP_FOR_ISSUE.get(kind),
                "evidence": ISSUE_EVIDENCE.get(kind, ""),
                "why": (
                    f"The fused model assigned {int(round(iss['confidence'] * 100))}% confidence to {kind.replace('_', ' ')}. "
                    f"{ISSUE_EVIDENCE.get(kind, '')} "
                    f"Supporting vitals: sharpness {stats.get('sharpness', 0):.4f}, "
                    f"brightness {stats.get('brightness', 0):.2f}, "
                    f"noise {stats.get('noise_estimate', 0):.4f}, "
                    f"blockiness {stats.get('blockiness', 0):.3f}."
                ),
                "operator_note": {
                    "blur": "Do not treat unreadable plates or faces as evidence.",
                    "underexposure": "Shadow regions may hide people and debris.",
                    "overexposure": "Headlights and sky may be empty pixels, not objects.",
                    "noise": "Speckle can look like rain or dust; confirm on the noise map.",
                    "corruption": "Re-export from the original DVR file if possible.",
                    "defect": "Check the defect map for stains, scratches, or punched holes.",
                }.get(kind, ""),
            }
        )
    return out


def _diagnosis(label: str, issues: list[dict], stats: dict) -> str:
    lead = {
        "DEFECTIVE": "Treat this frame as defective for civic use.",
        "DEGRADED": "This frame is degraded and may be unreliable for review.",
        "ACCEPTABLE": "Usable overall for civic review.",
    }[label]
    if not issues:
        return (
            f"{lead} Sharpness, exposure, and noise sit in an operator-acceptable band. "
            f"Mean luma {stats.get('brightness', 0):.2f}, Laplacian sharpness {stats.get('sharpness', 0):.4f}. "
            "No issue head cleared the 0.42 confidence gate."
        )
    ranked = sorted(issues, key=lambda x: x["confidence"], reverse=True)
    parts = []
    for item in ranked[:3]:
        parts.append(
            f"{item['type'].replace('_', ' ')} at {item['severity']} severity "
            f"({int(round(item['confidence'] * 100))}% confidence)"
        )
    follow = " Primary findings: " + "; ".join(parts) + "."
    extra = ""
    if len(ranked) > 1:
        extra = " Secondary issues can compound — inspect each heatmap before filing the still."
    return lead + follow + extra


FEATURE_CATALOG = {
    "sharpness_laplacian": {
        "label": "Laplacian sharpness",
        "feeds": "blur score, blur heatmap",
        "meaning": "Variance of the 3×3 Laplacian on luma. Low values mean the frame is soft or motion-blurred.",
    },
    "sharpness_tenengrad": {
        "label": "Tenengrad (Sobel energy)",
        "feeds": "blur score",
        "meaning": "Mean squared horizontal and vertical Sobel gradients. Confirms focus independently of Laplacian.",
    },
    "brightness_mean": {
        "label": "Mean luma",
        "feeds": "underexposure / overexposure",
        "meaning": "Average pixel brightness in [0, 1]. Mid-tones (~0.35–0.65) are most usable.",
    },
    "dark_frac": {
        "label": "Near-black fraction",
        "feeds": "underexposure, exposure heatmap (blue)",
        "meaning": "Share of pixels with luma < 25/255. High values crush shadow detail.",
    },
    "bright_frac": {
        "label": "Near-white fraction",
        "feeds": "overexposure, exposure heatmap (amber)",
        "meaning": "Share of pixels with luma > 230/255. High values mean clipped highlights.",
    },
    "contrast_std": {
        "label": "Luma contrast (std)",
        "feeds": "fog / flatness, overall usability",
        "meaning": "Standard deviation of luma. Fog, haze, and heavy compression flatten this.",
    },
    "saturation_mean": {
        "label": "Mean HSV saturation",
        "feeds": "colour usability, glare check",
        "meaning": "Average colour saturation. Near-zero often means IR, greyscale, or a heavy colour cast.",
    },
    "noise_mad": {
        "label": "Noise MAD",
        "feeds": "noise score, noise heatmap",
        "meaning": "Median absolute deviation of luma minus a 3×3 blur. High values are grain or sensor noise.",
    },
    "entropy": {
        "label": "Luma entropy",
        "feeds": "corruption / collapse check",
        "meaning": "Normalized histogram entropy. Very low entropy means posterization or a smashed file.",
    },
    "jpeg_blockiness": {
        "label": "JPEG blockiness",
        "feeds": "corruption",
        "meaning": "8×8 block-boundary gradient energy versus interior. High values indicate heavy JPEG smash.",
    },
    "median_residual": {
        "label": "Median-filter residual (p95)",
        "feeds": "defect",
        "meaning": "95th percentile of |luma − 15×15 median|. Spikes on stains, scratches, and punched holes.",
    },
    "defect_peak": {
        "label": "Peak tile defect",
        "feeds": "defect score, defect heatmap",
        "meaning": "Maximum tile-level residual outlier on the 16×16 grid.",
    },
    "defect_frac": {
        "label": "Defect tile fraction",
        "feeds": "defect score",
        "meaning": "Share of tiles whose defect score is above 0.45.",
    },
    "fft_high_ratio": {
        "label": "FFT high-frequency ratio",
        "feeds": "blur / softness (CNN+MLP)",
        "meaning": "High spatial frequencies versus low frequencies after FFT. Soft frames lose high-frequency energy.",
    },
    "mscn_var": {
        "label": "MSCN variance",
        "feeds": "noise / naturalness (BRISQUE-style)",
        "meaning": "Variance of mean-subtracted contrast-normalized luma. Grain and unnatural residuals raise it.",
    },
    "clahe_delta": {
        "label": "CLAHE residual",
        "feeds": "haze / low local contrast",
        "meaning": "Mean absolute change after CLAHE. Large change means the original was flat or hazy.",
    },
    "color_cast": {
        "label": "Colour-cast strength",
        "feeds": "white-balance / sensor bias",
        "meaning": "Max deviation of B/G/R channel means from grey. Strong casts distort civic colour cues.",
    },
    "glare_peak": {
        "label": "Glare / bloom fraction",
        "feeds": "overexposure / headlights",
        "meaning": "Share of pixels that are both very bright and low-saturation (bloom, flare, headlights).",
    },
}

FUSION_RULES = [
    "Unreadable or invalid files never receive a quality label; they return HTTP 400.",
    f"An issue is listed only if its fused probability is ≥ {ISSUE_CONF_MIN:.2f}.",
    "Severity is low at the gate, medium at 0.58, high at 0.75.",
    "DEFECTIVE if corruption confidence ≥ 0.65, or defect confidence ≥ 0.70, or score < 40 with a high-severity issue.",
    f"ACCEPTABLE if score ≥ {ACCEPTABLE_MIN:.0f} and there is no high-severity issue.",
    "Otherwise the label is DEGRADED.",
    "The 16×16 heatmaps use the same tile formulas as the global vitals (blur, exposure, noise, defect).",
    "There is no corruption heatmap; JPEG smash is treated as a global file property.",
    "The serving model is a CPU hybrid: 18 CV features (standardized) concatenated with a 128×128 TinyCNN embedding.",
]


def _round_num(v) -> float:
    return round(float(v), 6)


def measurement_ledger(features, zscores=None) -> list[dict]:
    vec = list(features) if features is not None else []
    zs = list(zscores) if zscores is not None else []
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        meta = FEATURE_CATALOG.get(name, {})
        rows.append(
            {
                "id": name,
                "label": meta.get("label", name),
                "raw": _round_num(vec[i]) if i < len(vec) else None,
                "zscore": _round_num(zs[i]) if i < len(zs) else None,
                "feeds": meta.get("feeds", ""),
                "meaning": meta.get("meaning", ""),
            }
        )
    return rows


def heatmap_summaries(maps: dict | None) -> list[dict]:
    if not maps:
        return []
    keys = [
        ("blur", "Blur overlay (inverted Laplacian vs image p95)"),
        ("underexposure", "Underexposure tiles (dark-pixel fraction)"),
        ("overexposure", "Overexposure tiles (bright-pixel fraction)"),
        ("noise", "Noise overlay (normalized MAD)"),
        ("defect", "Defect overlay (median-residual outliers)"),
    ]
    out = []
    for key, meaning in keys:
        arr = maps.get(key)
        if arr is None:
            continue
        flat = arr.ravel()
        out.append(
            {
                "id": key,
                "meaning": meaning,
                "min": _round_num(flat.min()),
                "mean": _round_num(flat.mean()),
                "max": _round_num(flat.max()),
                "p95": _round_num(float(np.percentile(flat, 95))),
            }
        )
    return out


def all_issue_heads(issue_probs: dict[str, float]) -> list[dict]:
    rows = []
    for name in ISSUE_TYPES:
        p = float(issue_probs.get(name, 0.0))
        listed = p >= ISSUE_CONF_MIN
        rows.append(
            {
                "type": name,
                "confidence": round(p, 4),
                "listed": listed,
                "severity": _severity(p) if listed else None,
                "heatmap": HEATMAP_FOR_ISSUE.get(name),
                "evidence": ISSUE_EVIDENCE.get(name, ""),
                "gate": ISSUE_CONF_MIN,
            }
        )
    return rows


def _region_name(row: int, col: int, rows: int, cols: int) -> str:
    vert = "the top" if row < rows / 3 else "the bottom" if row >= 2 * rows / 3 else "the middle"
    horiz = "left" if col < cols / 3 else "right" if col >= 2 * cols / 3 else "centre"
    if vert == "the middle" and horiz == "centre":
        return "the centre of the frame"
    if horiz == "centre":
        return f"{vert} of the frame"
    return f"{vert} {horiz}"


def _hotspot(maps: dict | None, key: str) -> str | None:
    if not maps or key not in maps:
        return None
    arr = np.asarray(maps[key])
    if arr.size == 0 or float(arr.max()) < 0.28:
        return None
    r, c = np.unravel_index(int(arr.argmax()), arr.shape)
    return _region_name(int(r), int(c), arr.shape[0], arr.shape[1])


def frame_description(
    score: float,
    label: str,
    issues: list[dict],
    stats: dict,
    maps: dict | None,
    context: str | None,
    bgr=None,
) -> dict:
    if issues:
        names = ", ".join(i["type"].replace("_", " ") for i in issues[:3])
        quality = (
            f"Quality note: score {score:.1f} ({label}). Main listed issues: {names}."
        )
    else:
        quality = f"Quality note: score {score:.1f} ({label}). No issue cleared the confidence gate."

    if bgr is not None:
        from app.vision.scene import describe_scene

        try:
            scene = describe_scene(bgr, context)
        except Exception:
            scene = {
                "full": "The still could not be described automatically. Inspect the photo on the left.",
                "people": 0,
                "vehicles": 0,
            }
        return {
            "appearance": scene["full"],
            "maps": "",
            "usefulness": quality,
            "full": f"{scene['full']} {quality}",
            "people": scene.get("people", 0),
            "vehicles": scene.get("vehicles", 0),
        }

    source = {
        "street": "a street still",
        "camera": "a CCTV grab",
        "other": "an uploaded still",
    }.get((context or "").lower(), "an uploaded still")
    return {
        "appearance": f"This looks like {source}. A full object readout needs the original pixels.",
        "maps": "",
        "usefulness": quality,
        "full": quality,
    }


def fuse_report(
    quality_score: float,
    issue_probs: dict[str, float],
    stats: dict,
    features=None,
    zscores=None,
    maps: dict | None = None,
    context: str | None = None,
    bgr=None,
) -> dict:
    score = float(max(0.0, min(100.0, quality_score)))
    issues = []
    for name in ISSUE_TYPES:
        p = float(issue_probs.get(name, 0.0))
        if p >= ISSUE_CONF_MIN:
            issues.append(
                {
                    "type": name,
                    "severity": _severity(p),
                    "confidence": round(p, 4),
                }
            )
    issues.sort(key=lambda x: x["confidence"], reverse=True)
    label = _label(score, issues)
    vitals = vital_explanations(stats)
    heads = all_issue_heads(issue_probs)
    return {
        "quality_score": round(score, 1),
        "quality_label": label,
        "issues": issues,
        "diagnosis": _diagnosis(label, issues, stats),
        "frame_description": frame_description(
            score, label, issues, stats, maps, context, bgr=bgr
        ),
        "statistics": stats,
        "issue_probabilities": {k: round(float(v), 4) for k, v in issue_probs.items()},
        "issue_heads": heads,
        "vital_explanations": vitals,
        "issue_explanations": issue_explanations(issues, stats),
        "measurements": measurement_ledger(features, zscores),
        "heatmap_summaries": heatmap_summaries(maps),
        "fusion": {
            "acceptable_min_score": ACCEPTABLE_MIN,
            "defective_score_floor": DEGRADED_MIN,
            "issue_confidence_gate": ISSUE_CONF_MIN,
            "severity_medium_at": 0.58,
            "severity_high_at": 0.75,
            "corruption_defective_at": 0.65,
            "defect_defective_at": 0.70,
            "rules": FUSION_RULES,
        },
        "model": {
            "name": "quality_hybrid_cnn_mlp",
            "device": "cpu",
            "inputs": "18 standardized CV features + 128×128 RGB TinyCNN",
            "outputs": "quality_score 0–100 and six issue probabilities",
            "heatmaps": "CV tiles only (not Grad-CAM); aligned with vitals",
        },
    }
