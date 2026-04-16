from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def create_analysis_session(
    *,
    filename: str,
    analysis_response: dict,
    original_audio_base64: str,
) -> dict:
    timestamp = _utc_now()
    session_id = uuid4().hex
    session_record = {
        "session_id": session_id,
        "filename": filename,
        "created_at": timestamp,
        "updated_at": timestamp,
        "analysis": analysis_response,
        "processed_response": None,
        "original_audio_base64": original_audio_base64,
    }
    session_record["analysis"]["session_id"] = session_id
    _write_session(session_id, session_record)
    return session_record


def update_processed_session(*, session_id: str, process_response: dict) -> dict:
    session_record = load_session(session_id)
    session_record["processed_response"] = process_response
    session_record["processed_response"]["session_id"] = session_id
    session_record["updated_at"] = _utc_now()
    _write_session(session_id, session_record)
    return session_record


def list_sessions(limit: int | None = None) -> list[dict]:
    sessions: list[dict] = []
    session_limit = limit or settings.session_list_limit
    session_dir = _session_dir()
    if not session_dir.exists():
        return []

    for session_file in session_dir.glob("*.json"):
        with session_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        sessions.append(
            {
                "session_id": record["session_id"],
                "filename": record["filename"],
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
                "status": record["analysis"]["status"],
                "detection_count": len(record["analysis"]["detections"]),
                "has_processed_audio": bool(record.get("processed_response")),
            }
        )

    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions[:session_limit]


def load_session(session_id: str) -> dict:
    session_path = _session_path(session_id)
    if not session_path.exists():
        raise FileNotFoundError(session_id)
    with session_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def delete_session(session_id: str) -> None:
    session_path = _session_path(session_id)
    if session_path.exists():
        session_path.unlink()


def _write_session(session_id: str, session_record: dict) -> None:
    session_dir = _session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = _session_path(session_id)
    with session_path.open("w", encoding="utf-8") as handle:
        json.dump(session_record, handle, indent=2)


def _session_dir() -> Path:
    configured_path = Path(settings.session_store_dir)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _session_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
