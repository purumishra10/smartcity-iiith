from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


INK = colors.HexColor("#1c2430")
LINE = colors.HexColor("#c8c0b4")


def _composite(original_bgr: np.ndarray, overlay_path: Path) -> Image.Image:
    base = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    base_img = Image.fromarray(base).convert("RGBA")
    if overlay_path.exists():
        overlay = Image.open(overlay_path).convert("RGBA")
        overlay = overlay.resize(base_img.size, Image.Resampling.BILINEAR)
        base_img = Image.alpha_composite(base_img, overlay)
    return base_img.convert("RGB")


def _png_bytes(pil: Image.Image, max_w: int = 720) -> io.BytesIO:
    w, h = pil.size
    if w > max_w:
        h = int(h * max_w / w)
        w = max_w
        pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    buf.name = "frame.png"
    pil.save(buf, format="PNG")
    buf.seek(0)
    buf._w, buf._h = w, h  # type: ignore[attr-defined]
    return buf


def _styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=styles["Title"], fontSize=16, spaceAfter=6),
        "h": ParagraphStyle("H", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("B", parent=styles["BodyText"], fontSize=9, leading=12),
        "small": ParagraphStyle(
            "S", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#444")
        ),
    }


def _table(data, col_widths):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def _img(buf, max_w=160 * mm):
    width = min(max_w, 160 * mm)
    height = buf._h * (width / buf._w)  # type: ignore[attr-defined]
    return RLImage(buf, width=width, height=height)


def build_report_pdf(row, payload: dict, original_path: Path, heatmap_dir: Path) -> bytes:
    data = original_path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unreadable_original")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Dr. Image report {row.id}",
    )
    s = _styles()
    small, body, h, title = s["small"], s["body"], s["h"], s["title"]

    story = []
    story.append(Paragraph("Dr. Image — complete exam report", title))
    created = row.created_at
    created_s = created.isoformat() if isinstance(created, datetime) else str(created)
    story.append(
        Paragraph(
            f"Exam <b>{row.id}</b> · label <b>{payload.get('quality_label')}</b> · "
            f"score <b>{payload.get('quality_score')}</b> · {created_s}",
            body,
        )
    )
    story.append(Paragraph(payload.get("diagnosis") or "", body))

    desc = payload.get("frame_description") or {}
    story.append(Paragraph("What's in the frame", h))
    story.append(Paragraph(desc.get("appearance") or payload.get("diagnosis") or "", body))
    if desc.get("usefulness"):
        story.append(Paragraph(desc["usefulness"], small))

    intake = payload.get("intake") or {}
    story.append(Paragraph("Intake and file facts", h))
    story.append(
        Paragraph(
            f"Filename: {intake.get('filename') or 'upload'} · "
            f"{intake.get('bytes', row.file_size)} bytes · context: {payload.get('context') or 'unspecified'} · "
            f"original {intake.get('original_width')}×{intake.get('original_height')} px · "
            f"working copy {intake.get('working_width')}×{intake.get('working_height')} px "
            f"(capped at {intake.get('max_analysis_dim', 512)} px) · "
            f"heatmap grid {intake.get('grid_size') or (payload.get('grid') or {}).get('rows')}×"
            f"{intake.get('grid_size') or (payload.get('grid') or {}).get('cols')}.",
            small,
        )
    )

    orig_buf = _png_bytes(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    story.append(Paragraph("Original still", h))
    story.append(_img(orig_buf))

    model = payload.get("model") or {}
    story.append(Paragraph("Model", h))
    story.append(
        Paragraph(
            f"Name: {model.get('name')} · device: {model.get('device')} · "
            f"inputs: {model.get('inputs')} · outputs: {model.get('outputs')} · "
            f"heatmaps: {model.get('heatmaps')}",
            small,
        )
    )

    fusion = payload.get("fusion") or {}
    story.append(Paragraph("Fusion rules (how the label is decided)", h))
    for rule in fusion.get("rules") or []:
        story.append(Paragraph(f"• {rule}", small))
    story.append(
        Paragraph(
            f"Numeric gates — list issue if confidence ≥ {fusion.get('issue_confidence_gate')}; "
            f"medium ≥ {fusion.get('severity_medium_at')}; high ≥ {fusion.get('severity_high_at')}; "
            f"corruption DEFECTIVE ≥ {fusion.get('corruption_defective_at')}; "
            f"defect DEFECTIVE ≥ {fusion.get('defect_defective_at')}; "
            f"ACCEPTABLE if score ≥ {fusion.get('acceptable_min_score')} and no high-severity issue; "
            f"DEFECTIVE also if score &lt; {fusion.get('defective_score_floor')} with a high-severity issue.",
            small,
        )
    )

    heads = payload.get("issue_heads") or []
    if heads:
        story.append(Paragraph("Every issue head (listed or not)", h))
        rows = [["Issue", "Confidence", "Listed", "Severity", "Heatmap", "What the model used"]]
        for item in heads:
            rows.append(
                [
                    item.get("type", "").replace("_", " "),
                    f"{100 * float(item.get('confidence') or 0):.1f}%",
                    "yes" if item.get("listed") else "no (below gate)",
                    item.get("severity") or "—",
                    item.get("heatmap") or "none (global)",
                    Paragraph(item.get("evidence") or "", small),
                ]
            )
        story.append(_table(rows, [28 * mm, 22 * mm, 28 * mm, 20 * mm, 24 * mm, 50 * mm]))

    vitals = payload.get("vital_explanations") or []
    if vitals:
        story.append(Paragraph("Operator vitals (why each display score)", h))
        table_data = [["Vital", "Display", "Raw", "Meaning", "Why this image", "Operator action"]]
        for v in vitals:
            table_data.append(
                [
                    v.get("label", ""),
                    str(v.get("display", "")),
                    f"{v.get('raw', 0):.4f}" if isinstance(v.get("raw"), (int, float)) else str(v.get("raw")),
                    Paragraph(v.get("meaning") or "", small),
                    Paragraph(v.get("why") or "", small),
                    Paragraph(v.get("action") or "", small),
                ]
            )
        story.append(_table(table_data, [22 * mm, 16 * mm, 18 * mm, 38 * mm, 42 * mm, 36 * mm]))

    measurements = payload.get("measurements") or []
    if measurements:
        story.append(PageBreak())
        story.append(Paragraph("Complete measurement ledger (every model input)", h))
        story.append(
            Paragraph(
                "These 18 numbers are standardized with the training-set mean/std (z-score) and concatenated "
                "with the CNN embedding. Z-score 0 is typical of the train set; large |z| is unusual.",
                small,
            )
        )
        mrows = [["Feature", "Raw value", "Z-score", "Used for", "Meaning"]]
        for m in measurements:
            raw = m.get("raw")
            z = m.get("zscore")
            mrows.append(
                [
                    Paragraph(f"<b>{m.get('label')}</b><br/>{m.get('id')}", small),
                    f"{raw:.6f}" if isinstance(raw, (int, float)) else "—",
                    f"{z:.4f}" if isinstance(z, (int, float)) else "—",
                    Paragraph(m.get("feeds") or "", small),
                    Paragraph(m.get("meaning") or "", small),
                ]
            )
        story.append(_table(mrows, [38 * mm, 22 * mm, 18 * mm, 40 * mm, 54 * mm]))

    summaries = payload.get("heatmap_summaries") or []
    if summaries:
        story.append(Paragraph("Heatmap tile statistics", h))
        srows = [["Map", "Min", "Mean", "P95", "Max", "How it is computed"]]
        for item in summaries:
            srows.append(
                [
                    item.get("id", ""),
                    f"{item.get('min'):.4f}",
                    f"{item.get('mean'):.4f}",
                    f"{item.get('p95'):.4f}",
                    f"{item.get('max'):.4f}",
                    Paragraph(item.get("meaning") or "", small),
                ]
            )
        story.append(_table(srows, [28 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 72 * mm]))

    notes = payload.get("issue_explanations") or []
    if notes:
        story.append(Paragraph("Flagged-issue write-ups", h))
        for note in notes:
            story.append(Paragraph(f"<b>{note.get('type', '').replace('_', ' ')}</b>", body))
            story.append(Paragraph(note.get("why") or "", small))
            story.append(Paragraph(note.get("operator_note") or "", small))

    kinds = [
        ("blur", "Blur", "Magenta = soft tiles. Computed as 1 − (tile Laplacian / image p95 Laplacian)."),
        ("exposure", "Exposure", "Blue = underexposure (dark fraction). Amber = overexposure (bright fraction)."),
        ("noise", "Noise", "Green = high-frequency residual MAD, min–max normalized on this frame."),
        ("defect", "Defect", "Red = tiles whose median-filter residual is an outlier versus the grid."),
    ]
    explanations = {e["type"]: e for e in notes}
    issue_by_heat = {
        "blur": explanations.get("blur"),
        "exposure": explanations.get("underexposure") or explanations.get("overexposure"),
        "noise": explanations.get("noise"),
        "defect": explanations.get("defect"),
    }

    for kind, title_s, caption in kinds:
        story.append(PageBreak())
        story.append(Paragraph(f"{title_s} heatmap on the original still", h))
        story.append(Paragraph(caption, body))
        detail = issue_by_heat.get(kind)
        if detail:
            story.append(Paragraph(detail.get("why") or "", body))
            if detail.get("operator_note"):
                story.append(Paragraph(detail["operator_note"], small))
        else:
            story.append(
                Paragraph(
                    f"No {title_s.lower()} issue cleared the {fusion.get('issue_confidence_gate', 0.42)} "
                    "confidence gate. The overlay is still printed so residual heat can be inspected.",
                    body,
                )
            )
        overlay = heatmap_dir / f"{kind}.png"
        story.append(Spacer(1, 6))
        story.append(_img(_png_bytes(_composite(bgr, overlay)), max_w=170 * mm))

    doc.build(story)
    return buffer.getvalue()
