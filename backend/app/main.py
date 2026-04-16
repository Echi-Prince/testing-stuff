import wave
import base64
import json
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio import (
    decode_wav,
    encode_wav_mono,
    extract_features,
    extract_log_mel_features,
    preprocess_audio,
    suppress_detected_classes,
)
from .classifier import build_classifier_detections
from .config import settings
from .model_loader import build_inference_backend
from .session_store import create_analysis_session, list_sessions, load_session, update_processed_session
from .schemas import (
    AnalysisResponse,
    AudioFeatures,
    AudioMetadata,
    DetectionResult,
    HealthResponse,
    ProcessResponse,
    ProcessedAudio,
    SessionDetailResponse,
    SessionListResponse,
    SpectralFeatures,
)


app = FastAPI(
    title="Sound Dashboard API",
    version="0.1.0",
    description=(
        "Backend scaffold for sound event detection and selective suppression."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_backend = build_inference_backend(
    supported_classes=settings.supported_classes,
    confidence_threshold=settings.classifier_confidence_threshold,
    baseline_name=settings.classifier_name,
    manifest_path=settings.trained_model_manifest_path,
)


@dataclass
class PreparedAnalysis:
    filename: str
    file_bytes: bytes
    decoded_audio: object
    processed_audio: object
    features: object
    spectral_features: object
    classifier_source: str
    used_fallback: bool
    detections: list[DetectionResult]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/config")
async def config() -> dict:
    return settings.model_dump()


@app.get("/sessions", response_model=SessionListResponse)
async def get_sessions() -> SessionListResponse:
    return SessionListResponse(sessions=list_sessions())


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    try:
        session_record = load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc

    return SessionDetailResponse(**session_record)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    prepared = await _prepare_analysis(file)
    analysis_response = AnalysisResponse(
        session_id="",
        filename=prepared.filename,
        status=inference_backend.name,
        classifier_source=prepared.classifier_source,
        used_fallback=prepared.used_fallback,
        message=(
            "Decoded WAV audio, normalized and resampled it to the shared target rate, "
            "computed prototype features, generated a log-mel summary for model-ready inputs, "
            "and ran the active classifier backend over the processed audio."
        ),
        metadata=_build_audio_metadata(prepared),
        features=AudioFeatures(
            rms=prepared.features.rms,
            peak_amplitude=prepared.features.peak_amplitude,
            zero_crossing_rate=prepared.features.zero_crossing_rate,
            dominant_activity_ratio=prepared.features.dominant_activity_ratio,
        ),
        spectral_features=SpectralFeatures(
            frame_count=prepared.spectral_features.frame_count,
            mel_bin_count=prepared.spectral_features.mel_bin_count,
            frame_size=prepared.spectral_features.frame_size,
            hop_size=prepared.spectral_features.hop_size,
            fft_size=prepared.spectral_features.fft_size,
            min_db=prepared.spectral_features.min_db,
            max_db=prepared.spectral_features.max_db,
            mean_db=prepared.spectral_features.mean_db,
            dynamic_range_db=prepared.spectral_features.dynamic_range_db,
            low_band_mean_db=prepared.spectral_features.low_band_mean_db,
            mid_band_mean_db=prepared.spectral_features.mid_band_mean_db,
            high_band_mean_db=prepared.spectral_features.high_band_mean_db,
        ),
        detections=prepared.detections,
    )
    session_record = create_analysis_session(
        filename=prepared.filename,
        analysis_response=analysis_response.model_dump(),
        original_audio_base64=base64.b64encode(prepared.file_bytes).decode("ascii"),
    )
    return AnalysisResponse(**session_record["analysis"])


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    suppressed_classes: str = Form(""),
    attenuation_factor: float = Form(settings.default_suppression_factor),
    suppression_profile: str = Form(""),
    session_id: str = Form(""),
) -> ProcessResponse:
    prepared = await _prepare_analysis(file)
    normalized_session_id = session_id.strip() if isinstance(session_id, str) else ""
    class_attenuation_factors = _build_suppression_profile(
        suppressed_classes=suppressed_classes,
        attenuation_factor=attenuation_factor,
        suppression_profile=suppression_profile,
    )

    suppressed_labels = list(class_attenuation_factors.keys())

    processed_samples = suppress_detected_classes(
        samples=prepared.processed_audio.samples,
        detections=[detection.model_dump() for detection in prepared.detections],
        class_attenuation_factors=class_attenuation_factors,
        sample_rate_hz=prepared.processed_audio.sample_rate_hz,
    )
    encoded_wav = encode_wav_mono(
        samples=processed_samples,
        sample_rate_hz=prepared.processed_audio.sample_rate_hz,
    )

    process_response = ProcessResponse(
        session_id=normalized_session_id,
        filename=prepared.filename,
        status="suppression_prototype_v1",
        classifier_source=prepared.classifier_source,
        used_fallback=prepared.used_fallback,
        message=(
            "Analyzed the uploaded WAV, applied prototype class-based attenuation over "
            "matching detected spans, and returned a processed mono WAV preview."
        ),
        metadata=_build_audio_metadata(prepared),
        detections=prepared.detections,
        processed_audio=ProcessedAudio(
            sample_rate_hz=prepared.processed_audio.sample_rate_hz,
            duration_ms=prepared.decoded_audio.duration_ms,
            sample_width_bytes=2,
            wav_byte_count=len(encoded_wav),
            attenuation_factor=attenuation_factor,
            suppressed_classes=suppressed_labels,
            class_attenuation_factors=class_attenuation_factors,
            wav_base64=base64.b64encode(encoded_wav).decode("ascii"),
        ),
    )
    if process_response.session_id:
        try:
            update_processed_session(
                session_id=process_response.session_id,
                process_response=process_response.model_dump(),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc

    return process_response


async def _prepare_analysis(file: UploadFile) -> PreparedAnalysis:
    if not file.filename:
        raise HTTPException(status_code=400, detail="An uploaded file is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        decoded_audio = decode_wav(file_bytes)
    except (wave.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processed_audio = preprocess_audio(
        samples=decoded_audio.samples,
        sample_rate_hz=decoded_audio.sample_rate_hz,
        target_sample_rate_hz=settings.sample_rate_hz,
    )
    features = extract_features(
        samples=processed_audio.samples,
        sample_rate_hz=processed_audio.sample_rate_hz,
    )
    spectral_features = extract_log_mel_features(
        samples=processed_audio.samples,
        sample_rate_hz=processed_audio.sample_rate_hz,
    )
    raw_detections, classifier_source, used_fallback = build_classifier_detections(
        classifier=inference_backend,
        samples=processed_audio.samples,
        sample_rate_hz=processed_audio.sample_rate_hz,
        features=features,
        spectral_features=spectral_features,
        duration_ms=decoded_audio.duration_ms,
    )
    detections = [
        DetectionResult(**detection)
        for detection in raw_detections
    ]

    return PreparedAnalysis(
        filename=file.filename,
        file_bytes=file_bytes,
        decoded_audio=decoded_audio,
        processed_audio=processed_audio,
        features=features,
        spectral_features=spectral_features,
        classifier_source=classifier_source,
        used_fallback=used_fallback,
        detections=detections,
    )


def _build_audio_metadata(prepared: PreparedAnalysis) -> AudioMetadata:
    return AudioMetadata(
        sample_rate_hz=prepared.decoded_audio.sample_rate_hz,
        processed_sample_rate_hz=prepared.processed_audio.sample_rate_hz,
        num_channels=prepared.decoded_audio.num_channels,
        sample_width_bytes=prepared.decoded_audio.sample_width_bytes,
        duration_ms=prepared.decoded_audio.duration_ms,
        frame_count=prepared.decoded_audio.frame_count,
        processed_sample_count=len(prepared.processed_audio.samples),
        was_resampled=prepared.processed_audio.was_resampled,
        original_peak_amplitude=prepared.processed_audio.original_peak_amplitude,
        normalized_peak_amplitude=prepared.processed_audio.normalized_peak_amplitude,
        normalization_gain=prepared.processed_audio.normalization_gain,
    )


def _parse_suppressed_classes(raw_value: str) -> list[str]:
    return [label.strip() for label in raw_value.split(",") if label.strip()]


def _build_suppression_profile(
    suppressed_classes: str,
    attenuation_factor: float,
    suppression_profile: str,
) -> dict[str, float]:
    if suppression_profile.strip():
        try:
            parsed_profile = json.loads(suppression_profile)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="suppression_profile must be valid JSON.",
            ) from exc

        if not isinstance(parsed_profile, dict):
            raise HTTPException(
                status_code=400,
                detail="suppression_profile must be a JSON object.",
            )

        normalized_profile: dict[str, float] = {}
        for raw_label, raw_factor in parsed_profile.items():
            label = str(raw_label).strip()
            factor = float(raw_factor)
            if factor < 0.0 or factor > 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="All suppression_profile values must be between 0.0 and 1.0.",
                )
            if label:
                normalized_profile[label] = factor

        return normalized_profile

    if attenuation_factor < 0.0 or attenuation_factor > 1.0:
        raise HTTPException(
            status_code=400,
            detail="attenuation_factor must be between 0.0 and 1.0.",
        )

    return {
        label: attenuation_factor for label in _parse_suppressed_classes(suppressed_classes)
    }
