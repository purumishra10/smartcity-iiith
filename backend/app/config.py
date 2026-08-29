from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    database_url: str = f"sqlite:///{(ROOT / 'data' / 'clinic.db').as_posix()}"
    storage_dir: str = str(ROOT / "data" / "storage")
    model_path: str = str((ROOT / "ml" / "artifacts" / "model.pt").as_posix())
    scaler_path: str = str((ROOT / "ml" / "artifacts" / "scaler.json").as_posix())
    yolo_model_path: str = str((ROOT / "ml" / "artifacts" / "yolov8n.pt").as_posix())
    max_upload_bytes: int = 10 * 1024 * 1024
    grid_size: int = 16
    max_analysis_dim: int = 512
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    jwt_secret: str = "dev-change-me-clinic-jwt"
    jwt_expire_hours: int = 72
    guest_cookie: str = "clinic_guest"
    auth_cookie: str = "clinic_token"


settings = Settings()
