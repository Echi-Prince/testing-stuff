from pydantic import BaseModel


class AppConfig(BaseModel):
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 1000
    chunk_overlap_ms: int = 500
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
