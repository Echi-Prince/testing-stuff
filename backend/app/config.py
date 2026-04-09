from pydantic import BaseModel


class AppConfig(BaseModel):
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 1000
    chunk_overlap_ms: int = 500
    classifier_name: str = "baseline_rules_v1"
    classifier_confidence_threshold: float = 0.45
    default_suppression_factor: float = 0.2
    trained_model_manifest_path: str = "training/artifacts/latest/manifest.json"
    supported_classes: list[str] = [
        "speech",
        "keyboard",
        "dog_bark",
        "traffic",
        "siren",
        "vacuum",
        "music",
    ]


settings = AppConfig()
