import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    return int(raw_value) if raw_value else default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    return float(raw_value) if raw_value else default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default[:]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _data_path(relative_path: str) -> str:
    data_root = os.getenv("SOUND_DASHBOARD_DATA_ROOT", "").strip()
    if not data_root:
        return relative_path
    return str(Path(data_root) / relative_path)


class AppConfig(BaseModel):
    sample_rate_hz: int = _env_int("SOUND_DASHBOARD_SAMPLE_RATE_HZ", 16000)
    chunk_duration_ms: int = _env_int("SOUND_DASHBOARD_CHUNK_DURATION_MS", 1000)
    chunk_overlap_ms: int = _env_int("SOUND_DASHBOARD_CHUNK_OVERLAP_MS", 250)
    classifier_name: str = os.getenv("SOUND_DASHBOARD_CLASSIFIER_NAME", "baseline_rules_v1")
    classifier_confidence_threshold: float = _env_float(
        "SOUND_DASHBOARD_CLASSIFIER_CONFIDENCE_THRESHOLD",
        0.45,
    )
    class_confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "speech": 0.55,
            "keyboard": 0.4,
            "dog_bark": 0.55,
            "traffic": 0.45,
            "siren": 0.5,
            "vacuum": 0.45,
            "music": 0.5,
        }
    )
    fallback_dominant_label_share_threshold: float = _env_float(
        "SOUND_DASHBOARD_FALLBACK_DOMINANT_LABEL_SHARE_THRESHOLD",
        0.5,
    )
    fallback_secondary_label_share_threshold: float = _env_float(
        "SOUND_DASHBOARD_FALLBACK_SECONDARY_LABEL_SHARE_THRESHOLD",
        0.3,
    )
    default_suppression_factor: float = _env_float(
        "SOUND_DASHBOARD_DEFAULT_SUPPRESSION_FACTOR",
        0.2,
    )
    trained_model_manifest_path: str = os.getenv(
        "SOUND_DASHBOARD_TRAINED_MODEL_MANIFEST_PATH",
        "training/artifacts/versions/20260423T000854Z/manifest.json",
    )
    real_recordings_dir: str = _data_path("training/real_recordings")
    real_recordings_manifest_path: str = _data_path("training/real_recordings/manifest.jsonl")
    real_recordings_validation_ratio: float = _env_float(
        "SOUND_DASHBOARD_REAL_RECORDINGS_VALIDATION_RATIO",
        0.2,
    )
    real_recordings_test_ratio: float = _env_float(
        "SOUND_DASHBOARD_REAL_RECORDINGS_TEST_RATIO",
        0.1,
    )
    real_recordings_split_seed: int = _env_int("SOUND_DASHBOARD_REAL_RECORDINGS_SPLIT_SEED", 7)
    training_runs_dir: str = _data_path("training/artifacts/runs")
    training_output_dir: str = _data_path("training/artifacts/latest-auto")
    training_versions_dir: str = _data_path("training/artifacts/versions")
    active_model_state_path: str = _data_path("training/artifacts/active-model.json")
    training_epochs: int = _env_int("SOUND_DASHBOARD_TRAINING_EPOCHS", 8)
    training_batch_size: int = _env_int("SOUND_DASHBOARD_TRAINING_BATCH_SIZE", 8)
    training_learning_rate: float = _env_float("SOUND_DASHBOARD_TRAINING_LEARNING_RATE", 0.001)
    session_store_dir: str = _data_path("backend/data/sessions")
    session_list_limit: int = _env_int("SOUND_DASHBOARD_SESSION_LIST_LIMIT", 20)
    supported_classes: list[str] = Field(
        default_factory=lambda: [
            "speech",
            "keyboard",
            "dog_bark",
            "traffic",
            "siren",
            "vacuum",
            "music",
        ]
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: _env_list(
            "SOUND_DASHBOARD_CORS_ALLOWED_ORIGINS",
            ["http://127.0.0.1:3000", "http://localhost:3000"],
        )
    )


settings = AppConfig()
