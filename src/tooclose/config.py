"""Load and validate ``config.yaml`` — the single source of TooClose configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    max_messages_per_second: int = 15
    max_message_bytes: int = 65536
    log_landmarks: bool = False


class CameraConfig(BaseModel):
    default_type: Literal["webcam", "phone"] = "webcam"
    default_hfov_deg: float = 70.0
    phone_hfov_deg: float = 75.0
    default_width: int = 1280
    default_height: int = 720


class AnatomyConfig(BaseModel):
    palm_width_cm: float = 8.5
    fingertip_width_mm: float = 15.0


class MediaPipeConfig(BaseModel):
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_landmarks: int = 21
    selfie_mirror: bool = True
    model_path: str = "models/hand_landmarker.task"


class SharpnessConfig(BaseModel):
    crop_scale: float = 1.5
    resize: int = 64
    laplacian_ksize: int = 3


class RiskConfig(BaseModel):
    ppi_low: float = 150.0
    ppi_high: float = 400.0
    warning_threshold: float = 0.4
    danger_threshold: float = 0.7
    ema_alpha: float = 0.35
    hysteresis_enter: float = 0.05
    hysteresis_exit: float = 0.08


class LabelingConfig(BaseModel):
    source: Literal["physics", "nfiq"] = "physics"
    nfiq_binary: str | None = None


class TrainingConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.08
    min_samples_leaf: int = 20
    group_by: str = "subject_id"
    synthetic_subjects: int = 40
    synthetic_sessions_per_subject: int = 4
    synthetic_frames_per_session: int = 40


class PathsConfig(BaseModel):
    data_dir: str = "data"
    models_dir: str = "models"
    model_file: str = "models/risk_model.joblib"
    metrics_file: str = "models/metrics.json"
    shap_global_plot: str = "models/shap_global.png"


class FrontendConfig(BaseModel):
    ws_url: str = "ws://127.0.0.1:8000/ws"
    send_hz: int = 15
    sparkline_seconds: int = 10


class AppConfig(BaseModel):
    seed: int = 42
    server: ServerConfig = Field(default_factory=ServerConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    anatomy: AnatomyConfig = Field(default_factory=AnatomyConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    sharpness: SharpnessConfig = Field(default_factory=SharpnessConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a config-relative path against the repository root."""
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path | None = None) -> AppConfig:
    """Parse YAML config from disk and return a validated ``AppConfig``."""
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)


def public_config(config: AppConfig) -> dict[str, Any]:
    """Subset of config that is safe to expose over GET /config (no paths to binaries)."""
    return {
        "seed": config.seed,
        "camera": config.camera.model_dump(),
        "anatomy": config.anatomy.model_dump(),
        "risk": {
            "ppi_low": config.risk.ppi_low,
            "ppi_high": config.risk.ppi_high,
            "warning_threshold": config.risk.warning_threshold,
            "danger_threshold": config.risk.danger_threshold,
        },
        "sharpness": config.sharpness.model_dump(),
        "frontend": config.frontend.model_dump(),
        "mediapipe": {
            "max_hands": config.mediapipe.max_hands,
            "selfie_mirror": config.mediapipe.selfie_mirror,
            "num_landmarks": config.mediapipe.num_landmarks,
        },
    }
