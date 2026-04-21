from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from .config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_SPLITS = {"train", "val", "test"}


def save_training_recording(
    *,
    file_bytes: bytes,
    label: str,
    split: str = "",
    source_name: str = "",
) -> dict[str, int | str]:
    normalized_label = label.strip().lower()
    if normalized_label not in settings.supported_classes:
        raise ValueError("label must be one of the supported classes.")

    normalized_split = split.strip().lower()
    if normalized_split and normalized_split not in _VALID_SPLITS:
        raise ValueError("split must be blank or one of: train, val, test.")

    safe_source = _slugify(source_name) or "browser"
    recording_id = uuid4().hex
    filename = f"{normalized_label}-{safe_source}-{recording_id[:8]}.wav"

    target_dir = _recordings_dir()
    if normalized_split:
        target_dir = target_dir / normalized_split
    target_dir = target_dir / normalized_label
    target_dir.mkdir(parents=True, exist_ok=True)

    absolute_path = target_dir / filename
    absolute_path.write_bytes(file_bytes)

    relative_path = absolute_path.relative_to(_REPO_ROOT).as_posix()
    return {
        "recording_id": recording_id,
        "label": normalized_label,
        "split": normalized_split,
        "filename": filename,
        "relative_path": relative_path,
        "byte_count": len(file_bytes),
    }


def _recordings_dir() -> Path:
    configured_path = Path(settings.real_recordings_dir)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")
