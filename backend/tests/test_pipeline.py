from __future__ import annotations

import io
import math
import unittest
import wave

from fastapi import HTTPException
from fastapi import UploadFile

from backend.app.audio import extract_log_mel_features, preprocess_audio
from backend.app.classifier import BaselineSoundClassifier
from backend.app.config import settings
from backend.app.main import analyze_audio, health


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


class ApiRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_route_returns_ok(self) -> None:
        response = await health()
        self.assertEqual(response.status, "ok")

    async def test_analyze_audio_returns_classifier_backed_response(self) -> None:
        upload = UploadFile(
            filename="sample.wav",
            file=io.BytesIO(build_test_wav_bytes(duration_seconds=1.0)),
        )

        response = await analyze_audio(file=upload)

        self.assertEqual(response.status, settings.classifier_name)
        self.assertEqual(response.metadata.sample_rate_hz, 16000)
        self.assertEqual(response.metadata.processed_sample_rate_hz, settings.sample_rate_hz)
        self.assertEqual(response.metadata.duration_ms, 1000)
        self.assertGreater(response.spectral_features.frame_count, 50)
        self.assertGreaterEqual(len(response.detections), 1)
        self.assertTrue(
            all(detection.label in settings.supported_classes for detection in response.detections)
        )

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


if __name__ == "__main__":
    unittest.main()
