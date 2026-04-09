import wave

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio import (
    decode_wav,
    extract_features,
    extract_log_mel_features,
    preprocess_audio,
)
from .classifier import BaselineSoundClassifier, build_classifier_detections
from .config import settings
from .schemas import (
    AnalysisResponse,
    AudioFeatures,
    AudioMetadata,
    DetectionResult,
    HealthResponse,
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

classifier = BaselineSoundClassifier(
    supported_classes=settings.supported_classes,
    confidence_threshold=settings.classifier_confidence_threshold,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/config")
async def config() -> dict:
    return settings.model_dump()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
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
    detections = [
        DetectionResult(**detection)
        for detection in build_classifier_detections(
            classifier=classifier,
            features=features,
            spectral_features=spectral_features,
            duration_ms=decoded_audio.duration_ms,
        )
    ]

    return AnalysisResponse(
        filename=file.filename,
        status=settings.classifier_name,
        message=(
            "Decoded WAV audio, normalized and resampled it to the shared target rate, "
            "computed prototype features, generated a log-mel summary for model-ready inputs, "
            "and ran the baseline classifier over the processed audio."
        ),
        metadata=AudioMetadata(
            sample_rate_hz=decoded_audio.sample_rate_hz,
            processed_sample_rate_hz=processed_audio.sample_rate_hz,
            num_channels=decoded_audio.num_channels,
            sample_width_bytes=decoded_audio.sample_width_bytes,
            duration_ms=decoded_audio.duration_ms,
            frame_count=decoded_audio.frame_count,
            processed_sample_count=len(processed_audio.samples),
            was_resampled=processed_audio.was_resampled,
            original_peak_amplitude=processed_audio.original_peak_amplitude,
            normalized_peak_amplitude=processed_audio.normalized_peak_amplitude,
            normalization_gain=processed_audio.normalization_gain,
        ),
        features=AudioFeatures(
            rms=features.rms,
            peak_amplitude=features.peak_amplitude,
            zero_crossing_rate=features.zero_crossing_rate,
            dominant_activity_ratio=features.dominant_activity_ratio,
        ),
        spectral_features=SpectralFeatures(
            frame_count=spectral_features.frame_count,
            mel_bin_count=spectral_features.mel_bin_count,
            frame_size=spectral_features.frame_size,
            hop_size=spectral_features.hop_size,
            fft_size=spectral_features.fft_size,
            min_db=spectral_features.min_db,
            max_db=spectral_features.max_db,
            mean_db=spectral_features.mean_db,
            dynamic_range_db=spectral_features.dynamic_range_db,
            low_band_mean_db=spectral_features.low_band_mean_db,
            mid_band_mean_db=spectral_features.mid_band_mean_db,
            high_band_mean_db=spectral_features.high_band_mean_db,
        ),
        detections=detections,
    )
