from __future__ import annotations

from dataclasses import dataclass

from .audio import ComputedFeatures, SpectralFeatures


@dataclass
class ClassPrediction:
    label: str
    confidence: float


class BaselineSoundClassifier:
    def __init__(
        self,
        supported_classes: list[str],
        confidence_threshold: float = 0.45,
        class_confidence_thresholds: dict[str, float] | None = None,
    ):
        self.supported_classes = supported_classes
        self.confidence_threshold = confidence_threshold
        self.class_confidence_thresholds = class_confidence_thresholds or {}

    def predict(
        self,
        features: ComputedFeatures,
        spectral_features: SpectralFeatures,
    ) -> list[ClassPrediction]:
        ranked_predictions = self.predict_ranked(
            features=features,
            spectral_features=spectral_features,
        )
        return apply_prediction_thresholds(
            predictions=ranked_predictions,
            default_threshold=self.confidence_threshold,
            class_confidence_thresholds=self.class_confidence_thresholds,
        )

    def predict_ranked(
        self,
        features: ComputedFeatures,
        spectral_features: SpectralFeatures,
    ) -> list[ClassPrediction]:
        scorers = {
            "speech": self._score_speech,
            "keyboard": self._score_keyboard,
            "dog_bark": self._score_dog_bark,
            "traffic": self._score_traffic,
            "siren": self._score_siren,
            "vacuum": self._score_vacuum,
            "music": self._score_music,
        }

        predictions: list[ClassPrediction] = []
        for label in self.supported_classes:
            scorer = scorers.get(label)
            if scorer is None:
                continue
            confidence = round(scorer(features, spectral_features), 3)
            predictions.append(ClassPrediction(label=label, confidence=confidence))

        predictions.sort(key=lambda item: item.confidence, reverse=True)
        return predictions[:3]

    def _score_speech(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.18
        score += self._scale(features.dominant_activity_ratio, 0.18, 0.9, 0.0, 0.28)
        score += self._scale(features.zero_crossing_rate, 0.035, 0.18, 0.0, 0.22)
        score += self._scale(features.rms, 0.02, 0.2, 0.0, 0.18)
        score += self._band_preference(
            spectral_features.mid_band_mean_db,
            spectral_features.low_band_mean_db,
            spectral_features.high_band_mean_db,
            weight=0.22,
        )
        score += self._scale(
            spectral_features.mid_band_mean_db - spectral_features.high_band_mean_db,
            2.0,
            14.0,
            0.0,
            0.14,
        )
        return self._clamp(score)

    def _score_keyboard(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        crest_factor = self._safe_ratio(features.peak_amplitude, max(features.rms, 1e-6))
        score = 0.08
        score += self._scale(features.zero_crossing_rate, 0.08, 0.2, 0.0, 0.22)
        score += self._scale(features.peak_amplitude, 0.45, 1.0, 0.0, 0.12)
        score += self._scale(features.dominant_activity_ratio, 0.05, 0.6, 0.08, 0.0)
        score += self._scale(spectral_features.dynamic_range_db, 18.0, 90.0, 0.0, 0.2)
        score += self._scale(crest_factor, 6.0, 28.0, 0.0, 0.24)
        score += self._scale(
            spectral_features.mid_band_mean_db - spectral_features.high_band_mean_db,
            -8.0,
            8.0,
            0.0,
            0.12,
        )
        return self._clamp(score)

    def _score_dog_bark(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.06
        score += self._scale(features.peak_amplitude, 0.55, 1.0, 0.0, 0.2)
        score += self._scale(features.zero_crossing_rate, 0.025, 0.12, 0.0, 0.14)
        score += self._scale(features.dominant_activity_ratio, 0.02, 0.35, 0.14, 0.0)
        score += self._scale(spectral_features.dynamic_range_db, 20.0, 75.0, 0.0, 0.22)
        score += self._band_preference(
            spectral_features.mid_band_mean_db,
            spectral_features.high_band_mean_db,
            spectral_features.low_band_mean_db,
            weight=0.16,
        )
        score += self._scale(features.rms, 0.04, 0.2, 0.0, 0.12)
        score -= self._scale(features.dominant_activity_ratio, 0.35, 0.75, 0.0, 0.18)
        score -= self._scale(features.rms, 0.1, 0.22, 0.0, 0.14)
        return self._clamp(score)

    def _score_traffic(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.08
        score += self._scale(features.dominant_activity_ratio, 0.4, 1.0, 0.0, 0.26)
        score += self._scale(features.rms, 0.02, 0.28, 0.0, 0.14)
        score += self._scale(features.zero_crossing_rate, 0.0, 0.06, 0.12, 0.0)
        score += self._band_preference(
            spectral_features.low_band_mean_db,
            spectral_features.mid_band_mean_db,
            spectral_features.high_band_mean_db,
            weight=0.24,
        )
        score += self._scale(spectral_features.dynamic_range_db, 6.0, 30.0, 0.16, 0.0)
        return self._clamp(score)

    def _score_siren(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.05
        score += self._scale(features.peak_amplitude, 0.75, 1.0, 0.0, 0.24)
        score += self._scale(features.zero_crossing_rate, 0.02, 0.16, 0.0, 0.1)
        score += self._scale(spectral_features.dynamic_range_db, 18.0, 80.0, 0.0, 0.24)
        score += self._band_preference(
            spectral_features.high_band_mean_db,
            spectral_features.mid_band_mean_db,
            spectral_features.low_band_mean_db,
            weight=0.22,
        )
        score += self._scale(spectral_features.max_db, 10.0, 45.0, 0.0, 0.12)
        score += self._scale(features.rms, 0.05, 0.25, 0.0, 0.1)
        score -= self._scale(
            spectral_features.mid_band_mean_db - spectral_features.high_band_mean_db,
            4.0,
            16.0,
            0.0,
            0.14,
        )
        return self._clamp(score)

    def _score_vacuum(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.08
        score += self._scale(features.dominant_activity_ratio, 0.55, 1.0, 0.0, 0.24)
        score += self._scale(features.rms, 0.025, 0.2, 0.0, 0.14)
        score += self._scale(features.zero_crossing_rate, 0.015, 0.09, 0.0, 0.08)
        score += self._band_preference(
            spectral_features.mid_band_mean_db,
            spectral_features.low_band_mean_db,
            spectral_features.high_band_mean_db,
            weight=0.18,
        )
        score += self._scale(spectral_features.dynamic_range_db, 8.0, 26.0, 0.2, 0.0)
        return self._clamp(score)

    def _score_music(
        self, features: ComputedFeatures, spectral_features: SpectralFeatures
    ) -> float:
        score = 0.12
        score += self._scale(features.dominant_activity_ratio, 0.2, 1.0, 0.0, 0.16)
        score += self._scale(features.rms, 0.015, 0.22, 0.0, 0.12)
        score += self._scale(spectral_features.dynamic_range_db, 12.0, 55.0, 0.0, 0.18)
        score += self._balanced_band_score(
            spectral_features.low_band_mean_db,
            spectral_features.mid_band_mean_db,
            spectral_features.high_band_mean_db,
            weight=0.26,
        )
        score += self._scale(spectral_features.max_db, 8.0, 40.0, 0.0, 0.12)
        return self._clamp(score)

    def _scale(
        self, value: float, minimum: float, maximum: float, low_score: float, high_score: float
    ) -> float:
        if maximum <= minimum:
            return low_score
        bounded = min(max(value, minimum), maximum)
        ratio = (bounded - minimum) / (maximum - minimum)
        return low_score + ((high_score - low_score) * ratio)

    def _band_preference(
        self,
        preferred_db: float,
        secondary_db: float,
        tertiary_db: float,
        weight: float,
    ) -> float:
        margin = preferred_db - max(secondary_db, tertiary_db)
        return self._scale(margin, -15.0, 12.0, 0.0, weight)

    def _balanced_band_score(
        self, low_db: float, mid_db: float, high_db: float, weight: float
    ) -> float:
        spread = max(low_db, mid_db, high_db) - min(low_db, mid_db, high_db)
        if spread <= 0.0:
            return weight
        return self._scale(max(0.0, 18.0 - spread), 0.0, 18.0, 0.0, weight)

    def _clamp(self, value: float) -> float:
        return min(0.99, max(0.0, value))

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator


def apply_prediction_thresholds(
    *,
    predictions: list[ClassPrediction],
    default_threshold: float,
    class_confidence_thresholds: dict[str, float] | None = None,
    max_results: int = 3,
) -> list[ClassPrediction]:
    threshold_lookup = class_confidence_thresholds or {}
    filtered_predictions = [
        prediction
        for prediction in predictions
        if prediction.confidence >= threshold_lookup.get(prediction.label, default_threshold)
    ]
    return filtered_predictions[:max_results]


def build_classifier_detections(
    classifier: object,
    samples: list[float],
    sample_rate_hz: int,
    features: ComputedFeatures,
    spectral_features: SpectralFeatures,
    duration_ms: int,
) -> tuple[list[dict[str, float | int | str]], str, bool]:
    source_name = getattr(classifier, "name", classifier.__class__.__name__)
    used_fallback = False
    if hasattr(classifier, "predict_with_metadata"):
        prediction_result = classifier.predict_with_metadata(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            features=features,
            spectral_features=spectral_features,
        )
        predictions = prediction_result.predictions
        source_name = prediction_result.source_name
        used_fallback = prediction_result.used_fallback
    else:
        predictions = classifier.predict(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            features=features,
            spectral_features=spectral_features,
        )
    end_ms = max(duration_ms, 1)
    return (
        [
        {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "start_ms": 0,
            "end_ms": end_ms,
        }
        for prediction in predictions
        ],
        source_name,
        used_fallback,
    )
