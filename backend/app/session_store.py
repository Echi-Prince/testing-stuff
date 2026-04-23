from __future__ import annotations

import base64
import binascii
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
    original_audio_bytes: bytes,
) -> dict:
    timestamp = _utc_now()
    session_id = uuid4().hex
    original_audio_path = _session_audio_path(session_id)
    original_audio_path.parent.mkdir(parents=True, exist_ok=True)
    original_audio_path.write_bytes(original_audio_bytes)
    session_record = {
        "session_id": session_id,
        "filename": filename,
        "created_at": timestamp,
        "updated_at": timestamp,
        "analysis": analysis_response,
        "processed_response": None,
        "original_audio_path": original_audio_path.name,
    }
    session_record["analysis"]["session_id"] = session_id
    _write_session(session_id, session_record)
    return _attach_audio_payloads(session_record)


def update_processed_session(*, session_id: str, process_response: dict) -> dict:
    session_record = _load_raw_session(session_id)
    stored_process_response = json.loads(json.dumps(process_response))
    processed_audio = stored_process_response.get("processed_audio")
    if isinstance(processed_audio, dict):
        wav_base64 = str(processed_audio.pop("wav_base64", "") or "")
        if wav_base64:
            processed_audio_bytes = _decode_base64_audio(wav_base64)
            processed_audio_path = _session_processed_audio_path(session_id)
            processed_audio_path.write_bytes(processed_audio_bytes)
            session_record["processed_audio_path"] = processed_audio_path.name

    session_record["processed_response"] = stored_process_response
    session_record["processed_response"]["session_id"] = session_id
    session_record["updated_at"] = _utc_now()
    _write_session(session_id, session_record)
    return _attach_audio_payloads(session_record)


def list_sessions(limit: int | None = None) -> list[dict]:
    sessions: list[dict] = []
    session_limit = limit or settings.session_list_limit
    session_dir = _session_dir()
    if not session_dir.exists():
        return []

    summary_by_id: dict[str, dict] = {}
    for summary_file in session_dir.glob("*.summary.json"):
        with summary_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        summary_by_id[str(record["session_id"])] = record

    for session_file in session_dir.glob("*.json"):
        if session_file.name.endswith(".summary.json"):
            continue
        session_id = session_file.stem
        if session_id in summary_by_id:
            continue
        with session_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        summary = _build_session_summary(record)
        summary_by_id[str(summary["session_id"])] = summary
        _write_session_summary(summary)

    sessions = list(summary_by_id.values())
    sessions.sort(key=lambda item: (str(item["updated_at"]), str(item["session_id"])), reverse=True)
    return sessions[:session_limit]


def load_session(session_id: str) -> dict:
    return _attach_audio_payloads(_load_raw_session(session_id))


def delete_session(session_id: str) -> None:
    session_path = _session_path(session_id)
    if session_path.exists():
        session_path.unlink()
    summary_path = _session_summary_path(session_id)
    if summary_path.exists():
        summary_path.unlink()
    original_audio_path = _session_audio_path(session_id)
    if original_audio_path.exists():
        original_audio_path.unlink()
    processed_audio_path = _session_processed_audio_path(session_id)
    if processed_audio_path.exists():
        processed_audio_path.unlink()


def _write_session(session_id: str, session_record: dict) -> None:
    session_dir = _session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = _session_path(session_id)
    with session_path.open("w", encoding="utf-8") as handle:
        json.dump(session_record, handle, separators=(",", ":"))
    _write_session_summary(_build_session_summary(session_record))


def _load_raw_session(session_id: str) -> dict:
    session_path = _session_path(session_id)
    if not session_path.exists():
        raise FileNotFoundError(session_id)
    with session_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _session_dir() -> Path:
    configured_path = Path(settings.session_store_dir)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _session_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.json"


def _session_summary_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.summary.json"


def _session_audio_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.original.wav"


def _session_processed_audio_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.processed.wav"


def _write_session_summary(summary: dict) -> None:
    summary_path = _session_summary_path(str(summary["session_id"]))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, separators=(",", ":"))


def _build_session_summary(session_record: dict) -> dict[str, object]:
    processed_response = session_record.get("processed_response")
    return {
        "session_id": str(session_record["session_id"]),
        "filename": str(session_record["filename"]),
        "created_at": str(session_record["created_at"]),
        "updated_at": str(session_record["updated_at"]),
        "status": str(session_record["analysis"]["status"]),
        "detection_count": len(session_record["analysis"]["detections"]),
        "has_processed_audio": bool(
            session_record.get("processed_audio_path")
            or (
                isinstance(processed_response, dict)
                and processed_response.get("processed_audio")
            )
        ),
    }


def _attach_audio_payloads(session_record: dict) -> dict:
    hydrated_record = json.loads(json.dumps(session_record))
    hydrated_record["original_audio_base64"] = _load_original_audio_base64(session_record)

    processed_response = hydrated_record.get("processed_response")
    if isinstance(processed_response, dict):
        processed_audio = processed_response.get("processed_audio")
        if isinstance(processed_audio, dict) and "wav_base64" not in processed_audio:
            processed_audio["wav_base64"] = _load_processed_audio_base64(session_record)

    hydrated_record.pop("original_audio_path", None)
    hydrated_record.pop("processed_audio_path", None)
    return hydrated_record


def _load_original_audio_base64(session_record: dict) -> str:
    if session_record.get("original_audio_base64"):
        return str(session_record["original_audio_base64"])
    relative_audio_path = str(session_record.get("original_audio_path") or "")
    if not relative_audio_path:
        return ""
    audio_path = _session_dir() / relative_audio_path
    if not audio_path.exists():
        return ""
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def _load_processed_audio_base64(session_record: dict) -> str:
    processed_response = session_record.get("processed_response")
    if (
        isinstance(processed_response, dict)
        and isinstance(processed_response.get("processed_audio"), dict)
        and processed_response["processed_audio"].get("wav_base64")
    ):
        return str(processed_response["processed_audio"]["wav_base64"])
    relative_audio_path = str(session_record.get("processed_audio_path") or "")
    if not relative_audio_path:
        return ""
    audio_path = _session_dir() / relative_audio_path
    if not audio_path.exists():
        return ""
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def _decode_base64_audio(value: str) -> bytes:
    try:
        return base64.b64decode(value)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 audio payload.") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
