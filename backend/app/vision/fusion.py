from __future__ import annotations

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


def _diagnosis(label: str, issues: list[dict], stats: dict) -> str:
    if not issues:
        return (
            "No material quality issues on this still. Sharpness, exposure, and noise "
            "are within operator-acceptable range for civic review."
        )
    ranked = sorted(issues, key=lambda x: x["confidence"], reverse=True)
    primary = ranked[0]
    extra = ranked[1]["type"].replace("_", " ") if len(ranked) > 1 else None
    name = primary["type"].replace("_", " ")
    conf = int(round(primary["confidence"] * 100))
    lead = {
        "DEFECTIVE": "Treat this frame as defective for civic use.",
        "DEGRADED": "This frame is degraded and may be unreliable for review.",
        "ACCEPTABLE": "Usable overall, with a mild finding.",
    }[label]
    second = f" Secondary note: {extra}." if extra else ""
    vitals = (
        f" Vitals — sharpness {stats.get('sharpness', 0):.3f}, "
        f"brightness {stats.get('brightness', 0):.2f}, "
        f"noise {stats.get('noise_estimate', 0):.3f}."
    )
    return f"{lead} Primary finding: {name} ({primary['severity']}, {conf}% confidence).{second}{vitals}"


def fuse_report(
    quality_score: float,
    issue_probs: dict[str, float],
    stats: dict,
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
    return {
        "quality_score": round(score, 1),
        "quality_label": label,
        "issues": issues,
        "diagnosis": _diagnosis(label, issues, stats),
        "statistics": stats,
    }
