from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response as RawResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import (
    can_access,
    clear_auth_cookie,
    current_user,
    make_token,
    migrate_guest_exams,
    new_user,
    set_auth_cookie,
    validate_credentials,
    verify_password,
)
from app.config import settings
from app.database import Analysis, SessionLocal, User, init_db
from app.inference import QualityEngine
from app.report import build_report_pdf
from app.vision.features import bgr_to_working, decode_image, extract_global_features
from app.vision.fusion import fuse_report
from app.vision.heatmaps import save_heatmap_overlays

log = logging.getLogger("clinic")
logging.basicConfig(level=logging.INFO)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
HEATMAP_KINDS = {"blur", "exposure", "noise", "defect"}

engine_ml = QualityEngine(settings.model_path, settings.scaler_path)

app = FastAPI(title="Dr. Image", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AuthBody(BaseModel):
    email: str
    password: str


@app.middleware("http")
async def ensure_guest_cookie(request: Request, call_next):
    gid = request.cookies.get(settings.guest_cookie)
    minted = False
    if not gid:
        gid = str(uuid.uuid4())
        minted = True
    request.state.guest_id = gid
    response = await call_next(request)
    if minted:
        response.set_cookie(
            settings.guest_cookie,
            gid,
            httponly=True,
            samesite="lax",
            max_age=7 * 24 * 3600,
            path="/",
        )
    return response


@app.on_event("startup")
def startup() -> None:
    init_db()
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    engine_ml.load()


def _analysis_dir(analysis_id: str) -> Path:
    path = Path(settings.storage_dir) / analysis_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _identity(request: Request, db: Session):
    token = request.cookies.get(settings.auth_cookie)
    user = current_user(db, token)
    guest_id = getattr(request.state, "guest_id", None) or request.cookies.get(settings.guest_cookie)
    return user, guest_id


def _to_public(row: Analysis, saved: bool) -> dict:
    payload = json.loads(row.payload_json)
    payload["id"] = row.id
    payload["created_at"] = row.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    payload["heatmaps"] = {kind: f"/api/analyses/{row.id}/heatmaps/{kind}" for kind in HEATMAP_KINDS}
    payload["image_url"] = f"/api/analyses/{row.id}/image"
    payload["thumbnail_url"] = f"/api/analyses/{row.id}/image?thumb=1"
    payload["report_url"] = f"/api/analyses/{row.id}/report.pdf"
    payload["saved_to_history"] = saved
    payload["guest"] = row.user_id is None
    return payload


def _get_row_or_404(db: Session, analysis_id: str, user, guest_id: str | None) -> Analysis:
    row = db.get(Analysis, analysis_id)
    if row is None or not can_access(row, user, guest_id):
        raise HTTPException(status_code=404, detail="not_found")
    return row


@app.get("/health")
def health() -> dict:
    if not engine_ml.ready:
        raise HTTPException(status_code=503, detail=engine_ml.error or "model_not_loaded")
    return {"status": "ok", "model": engine_ml.kind, "device": "cpu"}


@app.get("/api/me")
def me(
    request: Request,
    db: Session = Depends(get_db),
):
    user, _ = _identity(request, db)
    if user is None:
        return {"authenticated": False, "email": None}
    return {"authenticated": True, "email": user.email, "id": user.id}


@app.post("/api/auth/signup")
def signup(
    body: AuthBody,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    email, password = validate_credentials(body.email, body.password)
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="email_taken")
    user = new_user(email, password)
    db.add(user)
    db.commit()
    guest_id = getattr(request.state, "guest_id", None) or request.cookies.get(settings.guest_cookie)
    migrated = migrate_guest_exams(db, user, guest_id)
    set_auth_cookie(response, make_token(user))
    return {"email": user.email, "migrated": migrated}


@app.post("/api/auth/login")
def login(
    body: AuthBody,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    email, password = validate_credentials(body.email, body.password)
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    guest_id = getattr(request.state, "guest_id", None) or request.cookies.get(settings.guest_cookie)
    migrated = migrate_guest_exams(db, user, guest_id)
    set_auth_cookie(response, make_token(user))
    return {"email": user.email, "migrated": migrated}


@app.post("/api/auth/logout")
def logout(response: Response):
    clear_auth_cookie(response)
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    context: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict:
    if not engine_ml.ready:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    user, guest_id = _identity(request, db)

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES and not (file.filename or "").lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ):
        raise HTTPException(status_code=400, detail="unsupported_file_type")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty_file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail="file_too_large")

    try:
        bgr = decode_image(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="unreadable_image") from None

    analysis_id = str(uuid.uuid4())
    dest = _analysis_dir(analysis_id)
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    original_path = dest / f"original{suffix}"
    original_path.write_bytes(data)

    working = bgr_to_working(bgr, settings.max_analysis_dim)
    features, stats, maps = extract_global_features(working, grid_size=settings.grid_size)
    try:
        score, probs = engine_ml.predict(features, working)
    except Exception:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference_failed") from None

    zscores = None
    if engine_ml.scaler is not None:
        zscores = engine_ml.scaler.transform(features.reshape(1, -1)).ravel()
    report = fuse_report(score, probs, stats, features=features, zscores=zscores, maps=maps)
    save_heatmap_overlays(maps, working.shape, dest / "heatmaps")

    thumb = cv2.resize(working, (160, 120), interpolation=cv2.INTER_AREA)
    thumb_path = dest / "thumb.jpg"
    cv2.imwrite(str(thumb_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    payload = {
        **report,
        "context": context,
        "grid": {"rows": maps["rows"], "cols": maps["cols"], "tiles": maps["tiles"]},
        "intake": {
            "filename": file.filename,
            "bytes": len(data),
            "original_width": int(bgr.shape[1]),
            "original_height": int(bgr.shape[0]),
            "working_width": int(working.shape[1]),
            "working_height": int(working.shape[0]),
            "grid_size": settings.grid_size,
            "max_analysis_dim": settings.max_analysis_dim,
        },
    }
    now = datetime.now(timezone.utc)
    try:
        row = Analysis(
            id=analysis_id,
            created_at=now,
            quality_score=report["quality_score"],
            quality_label=report["quality_label"],
            context=context,
            original_path=str(original_path),
            thumbnail_path=str(thumb_path),
            payload_json=json.dumps(payload),
            file_size=len(data),
            user_id=user.id if user else None,
            guest_session_id=None if user else guest_id,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _to_public(row, saved=True)
    except Exception:
        db.rollback()
        log.exception("persist failed")
        raise HTTPException(status_code=500, detail="persist_failed") from None


@app.get("/api/analyses")
def list_analyses(
    request: Request,
    db: Session = Depends(get_db),
) -> list[dict]:
    user, guest_id = _identity(request, db)
    q = db.query(Analysis)
    if user is not None:
        q = q.filter(Analysis.user_id == user.id)
    elif guest_id:
        q = q.filter(Analysis.guest_session_id == guest_id, Analysis.user_id.is_(None))
    else:
        return []
    rows = q.order_by(Analysis.created_at.desc()).limit(100).all()
    return [
        {
            "id": row.id,
            "created_at": row.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "quality_score": row.quality_score,
            "quality_label": row.quality_label,
            "context": row.context,
            "thumbnail_url": f"/api/analyses/{row.id}/image?thumb=1",
            "image_url": f"/api/analyses/{row.id}/image",
        }
        for row in rows
    ]


@app.get("/api/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    user, guest_id = _identity(request, db)
    row = _get_row_or_404(db, analysis_id, user, guest_id)
    return _to_public(row, saved=True)


@app.get("/api/analyses/{analysis_id}/image")
def get_image(
    analysis_id: str,
    request: Request,
    thumb: int = 0,
    db: Session = Depends(get_db),
):
    user, guest_id = _identity(request, db)
    row = _get_row_or_404(db, analysis_id, user, guest_id)
    path = Path(row.thumbnail_path or row.original_path) if thumb else Path(row.original_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="file_missing")
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media)


@app.get("/api/analyses/{analysis_id}/heatmaps/{kind}")
def get_heatmap(
    analysis_id: str,
    kind: str,
    request: Request,
    db: Session = Depends(get_db),
):
    if kind not in HEATMAP_KINDS:
        raise HTTPException(status_code=400, detail="unknown_heatmap")
    user, guest_id = _identity(request, db)
    _get_row_or_404(db, analysis_id, user, guest_id)
    path = Path(settings.storage_dir) / analysis_id / "heatmaps" / f"{kind}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/analyses/{analysis_id}/report.pdf")
def get_report(
    analysis_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    user, guest_id = _identity(request, db)
    row = _get_row_or_404(db, analysis_id, user, guest_id)
    payload = json.loads(row.payload_json)
    heatmap_dir = Path(settings.storage_dir) / analysis_id / "heatmaps"
    try:
        pdf = build_report_pdf(row, payload, Path(row.original_path), heatmap_dir)
    except Exception:
        log.exception("report failed")
        raise HTTPException(status_code=500, detail="report_failed") from None
    return RawResponse(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="dr-image-{analysis_id[:8]}.pdf"'},
    )
