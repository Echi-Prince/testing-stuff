from __future__ import annotations

import io
import math
import os
import base64
import json
import unittest
import wave
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import HTTPException
from fastapi import UploadFile

from backend.app.audio import extract_log_mel_features, preprocess_audio
from backend.app.classifier import (
    BaselineSoundClassifier,
    ClassPrediction,
    apply_prediction_thresholds,
)
from backend.app.config import settings
from backend.app.model_loader import (
    InferenceBackend,
    ModelArtifactManifest,
    TorchscriptWaveformClassifier,
    build_inference_backend,
    load_model_manifest,
    prepare_waveform_input,
)
from backend.app.main import (
    _filter_merged_detections,
    analyze_audio,
    build_windowed_detections,
    get_session,
    get_sessions,
    health,
    inference_backend,
    save_recording,
    process_audio,
)
from backend.app.model_loader import PredictionResult


def build_test_wav_bytes(
    sample_rate_hz: int = 16000,
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
    amplitude: int = 12000,
) -> bytes:
    frame_count = int(sample_rate_hz * duration_seconds)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)

        frames = bytearray()
        for index in range(frame_count):
            sample = int(
                amplitude * math.sin((2.0 * math.pi * frequency_hz * index) / sample_rate_hz)
            )
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))

        wav_file.writeframes(bytes(frames))

    return buffer.getvalue()


class AudioPipelineTests(unittest.TestCase):
    def test_apply_prediction_thresholds_uses_per_class_overrides(self) -> None:
        predictions = [
            ClassPrediction(label="speech", confidence=0.62),
            ClassPrediction(label="keyboard", confidence=0.41),
            ClassPrediction(label="music", confidence=0.49),
        ]

        filtered_predictions = apply_prediction_thresholds(
            predictions=predictions,
            default_threshold=0.45,
            class_confidence_thresholds={"speech": 0.7, "keyboard": 0.4},
        )

        self.assertEqual([prediction.label for prediction in filtered_predictions], ["keyboard", "music"])

    def test_inference_backend_falls_back_to_baseline_when_trained_model_returns_no_classes(self) -> None:
        class EmptyTorchscriptClassifier(TorchscriptWaveformClassifier):
            def __init__(self) -> None:
                self.manifest = ModelArtifactManifest(
                    model_name="waveform_cnn_v1",
                    model_type="torchscript_waveform_cnn",
                    class_names=settings.supported_classes,
                    sample_rate_hz=settings.sample_rate_hz,
                    input_sample_count=settings.sample_rate_hz,
                    confidence_threshold=0.45,
                    weights_path="training/artifacts/real-v1/model.ts",
                    normalization_target_peak=0.95,
                )

            def predict(self, samples: list[float]) -> list:
                return []

            def predict_ranked(self, samples: list[float]) -> list:
                return []

        baseline = BaselineSoundClassifier(
            supported_classes=settings.supported_classes,
            confidence_threshold=settings.classifier_confidence_threshold,
        )
        backend = InferenceBackend(
            name="trained_model:waveform_cnn_v1",
            predictor=EmptyTorchscriptClassifier(),
            fallback_predictor=baseline,
            manifest=EmptyTorchscriptClassifier().manifest,
            confidence_threshold=0.99,
            class_confidence_thresholds={
                label: 0.99 for label in settings.supported_classes
            },
        )
        samples = [0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667
        spectral = extract_log_mel_features(samples=samples, sample_rate_hz=16000)
        from backend.app.audio import extract_features

        features = extract_features(samples=samples, sample_rate_hz=16000)

        predictions = backend.predict(
            samples=samples,
            sample_rate_hz=16000,
            features=features,
            spectral_features=spectral,
        )

        self.assertGreaterEqual(len(predictions), 1)
        self.assertTrue(all(prediction.label in settings.supported_classes for prediction in predictions))

    def test_preprocess_audio_normalizes_and_resamples(self) -> None:
        samples = [0.0, 0.1, -0.2, 0.25, -0.25] * 20

        processed = preprocess_audio(
            samples=samples,
            sample_rate_hz=8000,
            target_sample_rate_hz=settings.sample_rate_hz,
        )

        self.assertEqual(processed.sample_rate_hz, settings.sample_rate_hz)
        self.assertTrue(processed.was_resampled)
        self.assertGreater(len(processed.samples), len(samples))
        self.assertAlmostEqual(processed.normalized_peak_amplitude, 0.95, places=6)
        self.assertGreater(processed.normalization_gain, 1.0)

    def test_extract_log_mel_features_returns_expected_shape(self) -> None:
        samples = [0.0, 0.3, -0.15, 0.5, -0.4, 0.2] * 2667

        spectral = extract_log_mel_features(samples=samples, sample_rate_hz=16000)

        self.assertEqual(spectral.mel_bin_count, 40)
        self.assertGreater(spectral.frame_count, 50)
        self.assertGreater(spectral.dynamic_range_db, 0.0)
        self.assertLessEqual(spectral.min_db, spectral.mean_db)
        self.assertLessEqual(spectral.mean_db, spectral.max_db)

    def test_baseline_classifier_returns_ranked_predictions(self) -> None:
        classifier = BaselineSoundClassifier(
            supported_classes=settings.supported_classes,
            confidence_threshold=settings.classifier_confidence_threshold,
        )
        spectral = extract_log_mel_features(
            samples=[0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667,
            sample_rate_hz=16000,
        )
        from backend.app.audio import extract_features

        features = extract_features(
            samples=[0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667,
            sample_rate_hz=16000,
        )

        predictions = classifier.predict(features=features, spectral_features=spectral)

        self.assertGreaterEqual(len(predictions), 1)
        self.assertLessEqual(len(predictions), 3)
        self.assertGreaterEqual(predictions[0].confidence, predictions[-1].confidence)
        self.assertTrue(all(prediction.label in settings.supported_classes for prediction in predictions))

    def test_baseline_classifier_predict_ranked_returns_best_match_below_threshold(self) -> None:
        classifier = BaselineSoundClassifier(
            supported_classes=settings.supported_classes,
            confidence_threshold=0.99,
        )
        spectral = extract_log_mel_features(
            samples=[0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667,
            sample_rate_hz=16000,
        )
        from backend.app.audio import extract_features

        features = extract_features(
            samples=[0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667,
            sample_rate_hz=16000,
        )

        thresholded_predictions = classifier.predict(features=features, spectral_features=spectral)
        ranked_predictions = classifier.predict_ranked(features=features, spectral_features=spectral)

        self.assertEqual(thresholded_predictions, [])
        self.assertGreaterEqual(len(ranked_predictions), 1)
        self.assertGreaterEqual(ranked_predictions[0].confidence, ranked_predictions[-1].confidence)

    def test_prepare_waveform_input_pads_to_target_length(self) -> None:
        prepared = prepare_waveform_input(samples=[0.1, -0.2, 0.3], input_sample_count=6)

        self.assertEqual(prepared, [0.1, -0.2, 0.3, 0.0, 0.0, 0.0])

    def test_model_manifest_loader_reads_export_metadata(self) -> None:
        manifest_path = "backend/tests/tmp-model-manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model_name": "waveform_cnn_v1",
                        "model_type": "torchscript_waveform_cnn",
                        "class_names": ["speech", "music"],
                        "sample_rate_hz": 16000,
                        "input_sample_count": 16000,
                        "confidence_threshold": 0.45,
                        "weights_path": "model.ts",
                        "normalization_target_peak": 0.95,
                    }
                )
            )

        try:
            manifest = load_model_manifest(manifest_path)
        finally:
            if os.path.exists(manifest_path):
                os.remove(manifest_path)

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.model_name, "waveform_cnn_v1")
        self.assertEqual(manifest.class_names, ["speech", "music"])
        self.assertTrue(str(manifest.weights_path_obj).endswith("model.ts"))

    def test_inference_backend_falls_back_to_baseline_when_manifest_missing(self) -> None:
        backend = build_inference_backend(
            supported_classes=settings.supported_classes,
            confidence_threshold=settings.classifier_confidence_threshold,
            class_confidence_thresholds=settings.class_confidence_thresholds,
            baseline_name=settings.classifier_name,
            manifest_path="",
        )

        self.assertEqual(backend.name, settings.classifier_name)
        self.assertIsNone(backend.manifest)

    def test_inference_backend_promotes_top_baseline_match_when_threshold_filters_everything(self) -> None:
        class EmptyTorchscriptClassifier(TorchscriptWaveformClassifier):
            def __init__(self) -> None:
                self.manifest = ModelArtifactManifest(
                    model_name="waveform_cnn_v1",
                    model_type="torchscript_waveform_cnn",
                    class_names=settings.supported_classes,
                    sample_rate_hz=settings.sample_rate_hz,
                    input_sample_count=settings.sample_rate_hz,
                    confidence_threshold=0.45,
                    weights_path="training/artifacts/real-v1/model.ts",
                    normalization_target_peak=0.95,
                )

            def predict(self, samples: list[float]) -> list:
                return []

            def predict_ranked(self, samples: list[float]) -> list:
                return []

        baseline = BaselineSoundClassifier(
            supported_classes=settings.supported_classes,
            confidence_threshold=0.99,
        )
        backend = InferenceBackend(
            name="trained_model:waveform_cnn_v1",
            predictor=EmptyTorchscriptClassifier(),
            fallback_predictor=baseline,
            manifest=EmptyTorchscriptClassifier().manifest,
            confidence_threshold=0.99,
            class_confidence_thresholds={
                label: 0.99 for label in settings.supported_classes
            },
        )
        samples = [0.0, 0.3, -0.1, 0.5, -0.4, 0.2] * 2667
        spectral = extract_log_mel_features(samples=samples, sample_rate_hz=16000)
        from backend.app.audio import extract_features

        features = extract_features(samples=samples, sample_rate_hz=16000)

        prediction_result = backend.predict_with_metadata(
            samples=samples,
            sample_rate_hz=16000,
            features=features,
            spectral_features=spectral,
        )

        self.assertEqual(len(prediction_result.predictions), 1)
        self.assertEqual(prediction_result.source_name, "baseline_rules_v1")
        self.assertTrue(prediction_result.used_fallback)

    def test_build_windowed_detections_merges_overlapping_chunk_predictions(self) -> None:
        class FakeWindowClassifier:
            def __init__(self) -> None:
                self._call_index = 0
                self._responses = [
                    PredictionResult(
                        predictions=[ClassPrediction(label="keyboard", confidence=0.82)],
                        source_name="trained_model:waveform_cnn_v1",
                        used_fallback=False,
                    ),
                    PredictionResult(
                        predictions=[ClassPrediction(label="keyboard", confidence=0.75)],
                        source_name="trained_model:waveform_cnn_v1",
                        used_fallback=False,
                    ),
                    PredictionResult(
                        predictions=[ClassPrediction(label="speech", confidence=0.64)],
                        source_name="baseline_rules_v1",
                        used_fallback=True,
                    ),
                ]

            def predict_with_metadata(self, **_: object) -> PredictionResult:
                response = self._responses[self._call_index]
                self._call_index += 1
                return response

        detections, classifier_sources, used_fallback = build_windowed_detections(
            classifier=FakeWindowClassifier(),
            samples=[0.0] * 32000,
            sample_rate_hz=16000,
            chunk_duration_ms=1000,
            chunk_overlap_ms=500,
        )

        self.assertEqual(
            detections,
            [
                {
                    "label": "keyboard",
                    "confidence": 0.82,
                    "start_ms": 0,
                    "end_ms": 1500,
                },
            ],
        )
        self.assertEqual(
            classifier_sources,
            ["baseline_rules_v1", "trained_model:waveform_cnn_v1"],
        )
        self.assertTrue(used_fallback)

    def test_filter_merged_detections_keeps_only_dominant_fallback_label(self) -> None:
        detections = [
            {"label": "keyboard", "confidence": 0.84, "start_ms": 0, "end_ms": 4644},
            {"label": "speech", "confidence": 0.71, "start_ms": 500, "end_ms": 4644},
            {"label": "dog_bark", "confidence": 0.69, "start_ms": 0, "end_ms": 1500},
        ]

        filtered_detections = _filter_merged_detections(
            detections=detections,
            top_window_labels=[
                "keyboard",
                "keyboard",
                "speech",
                "speech",
                "keyboard",
                "keyboard",
                "keyboard",
                "keyboard",
                "keyboard",
            ],
            used_fallback=True,
        )

        self.assertEqual(
            filtered_detections,
            [{"label": "keyboard", "confidence": 0.84, "start_ms": 0, "end_ms": 4644}],
        )


class ApiRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        temp_root = Path(__file__).resolve().parents[2] / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        self._temp_dir = temp_root / f"session-store-{uuid4().hex}"
        self._temp_dir.mkdir()
        self._recordings_dir = temp_root / f"real-recordings-{uuid4().hex}"
        self._recordings_dir.mkdir()
        self._original_session_store_dir = settings.session_store_dir
        self._original_real_recordings_dir = settings.real_recordings_dir
        settings.session_store_dir = str(self._temp_dir)
        settings.real_recordings_dir = str(self._recordings_dir)

    def tearDown(self) -> None:
        settings.session_store_dir = self._original_session_store_dir
        settings.real_recordings_dir = self._original_real_recordings_dir
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        shutil.rmtree(self._recordings_dir, ignore_errors=True)
        super().tearDown()

    async def test_health_route_returns_ok(self) -> None:
        response = await health()
        self.assertEqual(response.status, "ok")

    async def test_analyze_audio_returns_classifier_backed_response(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=1.0)),
        )

        response = await analyze_audio(file=upload)

        self.assertEqual(response.status, inference_backend.name)
        self.assertTrue(response.session_id)
        self.assertTrue(response.classifier_source)
        self.assertIsInstance(response.used_fallback, bool)
        self.assertEqual(response.metadata.sample_rate_hz, 16000)
        self.assertEqual(response.metadata.processed_sample_rate_hz, settings.sample_rate_hz)
        self.assertEqual(response.metadata.duration_ms, 1000)
        self.assertGreater(response.spectral_features.frame_count, 50)
        self.assertIsInstance(response.detections, list)
        self.assertTrue(
            all(detection.label in settings.supported_classes for detection in response.detections)
        )

    async def test_saved_session_can_be_listed_and_loaded(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.5)),
        )

        analysis = await analyze_audio(file=upload)
        session_list = await get_sessions()
        session_detail = await get_session(analysis.session_id)

        self.assertEqual(len(session_list.sessions), 1)
        self.assertEqual(session_list.sessions[0].session_id, analysis.session_id)
        self.assertEqual(session_detail.analysis.session_id, analysis.session_id)
        self.assertEqual(session_detail.filename, "sample.wav")
        self.assertTrue(session_detail.original_audio_base64)
        self.assertTrue(session_detail.analysis.classifier_source)
        self.assertIsNone(session_detail.processed_response)

    async def test_analyze_audio_rejects_missing_filename(self) -> None:
        upload = UploadFile(
            filename="",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.2)),
        )

        with self.assertRaises(HTTPException) as context:
            await analyze_audio(file=upload)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("uploaded file is required", context.exception.detail.lower())

    async def test_analyze_audio_rejects_empty_file(self) -> None:
        upload = UploadFile(filename="empty.wav", file=io.BytesIO(b""))

        with self.assertRaises(HTTPException) as context:
            await analyze_audio(file=upload)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("uploaded file is empty", context.exception.detail.lower())

    async def test_analyze_audio_rejects_invalid_wav_payload(self) -> None:
        upload = UploadFile(filename="invalid.wav", file=io.BytesIO(b"not a wav file"))

        with self.assertRaises(HTTPException) as context:
            await analyze_audio(file=upload)

        self.assertEqual(context.exception.status_code, 400)
        self.assertTrue(context.exception.detail)

    async def test_process_audio_returns_processed_wav_payload(self) -> None:
        analysis = await analyze_audio(
            file=UploadFile(
                filename="sample.wav",
                file=io.BytesIO(build_test_wav_bytes(duration_seconds=1.0)),
            )
        )
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=1.0)),
        )

        response = await process_audio(
            file=upload,
            suppressed_classes="traffic,music",
            attenuation_factor=0.2,
            suppression_profile="",
            session_id=analysis.session_id,
        )

        self.assertEqual(response.status, "suppression_prototype_v1")
        self.assertEqual(response.session_id, analysis.session_id)
        self.assertEqual(response.classifier_source, analysis.classifier_source)
        self.assertEqual(response.used_fallback, analysis.used_fallback)
        self.assertEqual(response.processed_audio.sample_rate_hz, settings.sample_rate_hz)
        self.assertEqual(response.processed_audio.suppressed_classes, ["traffic", "music"])
        self.assertEqual(
            response.processed_audio.class_attenuation_factors,
            {"traffic": 0.2, "music": 0.2},
        )
        self.assertGreater(response.processed_audio.wav_byte_count, 100)
        decoded_bytes = base64.b64decode(response.processed_audio.wav_base64)
        self.assertTrue(decoded_bytes.startswith(b"RIFF"))
        session_detail = await get_session(analysis.session_id)
        self.assertIsNotNone(session_detail.processed_response)
        assert session_detail.processed_response is not None
        self.assertEqual(
            session_detail.processed_response.processed_audio.suppressed_classes,
            ["traffic", "music"],
        )

    async def test_process_audio_accepts_per_class_suppression_profile(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=1.0)),
        )

        response = await process_audio(
            file=upload,
            attenuation_factor=0.2,
            suppression_profile=json.dumps({"traffic": 0.1, "music": 0.65}),
        )

        self.assertEqual(
            response.processed_audio.class_attenuation_factors,
            {"traffic": 0.1, "music": 0.65},
        )
        self.assertEqual(response.processed_audio.suppressed_classes, ["traffic", "music"])

    async def test_process_audio_rejects_invalid_attenuation_factor(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.2)),
        )

        with self.assertRaises(HTTPException) as context:
            await process_audio(
                file=upload,
                suppressed_classes="traffic",
                attenuation_factor=1.5,
                suppression_profile="",
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("attenuation_factor", context.exception.detail)

    async def test_process_audio_rejects_invalid_suppression_profile(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.2)),
        )

        with self.assertRaises(HTTPException) as context:
            await process_audio(
                file=upload,
                suppression_profile='{"traffic": 1.4}',
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("suppression_profile", context.exception.detail.lower())

    async def test_save_recording_persists_wav_into_real_recordings_layout(self) -> None:
        response = await save_recording(
            file=UploadFile(
                filename="recorded.wav",
                file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.25)),
            ),
            label="speech",
            split="train",
            source_name="browser mic",
        )

        self.assertEqual(response.label, "speech")
        self.assertEqual(response.split, "train")
        self.assertGreater(response.byte_count, 100)
        self.assertEqual(response.sample_rate_hz, 16000)
        self.assertEqual(response.duration_ms, 250)
        saved_path = Path(settings.real_recordings_dir) / "train" / "speech" / response.filename
        self.assertTrue(saved_path.exists())
        self.assertTrue(response.relative_path.endswith(f"train/speech/{response.filename}"))

    async def test_save_recording_rejects_unsupported_label(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await save_recording(
                file=UploadFile(
                    filename="recorded.wav",
                    file=io.BytesIO(build_test_wav_bytes(duration_seconds=0.25)),
                ),
                label="car_horn",
                split="",
                source_name="browser",
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("supported classes", context.exception.detail.lower())


if __name__ == "__main__":
    unittest.main()
