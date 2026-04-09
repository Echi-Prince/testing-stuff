from __future__ import annotations

import io
import cmath
import math
import wave
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for minimal environments
    np = None


@dataclass
class DecodedAudio:
    samples: list[float]
    sample_rate_hz: int
    num_channels: int
    sample_width_bytes: int
    frame_count: int

    @property
    def duration_ms(self) -> int:
        if self.sample_rate_hz == 0:
            return 0
        return int((self.frame_count / self.sample_rate_hz) * 1000)


@dataclass
class ComputedFeatures:
    rms: float
    peak_amplitude: float
    zero_crossing_rate: float
    dominant_activity_ratio: float


@dataclass
class PreprocessedAudio:
    samples: list[float]
    sample_rate_hz: int
    original_peak_amplitude: float
    normalized_peak_amplitude: float
    was_resampled: bool
    normalization_gain: float


@dataclass
class SpectralFeatures:
    frame_count: int
    mel_bin_count: int
    frame_size: int
    hop_size: int
    fft_size: int
    min_db: float
    max_db: float
    mean_db: float
    dynamic_range_db: float
    low_band_mean_db: float
    mid_band_mean_db: float
    high_band_mean_db: float


def decode_wav(file_bytes: bytes) -> DecodedAudio:
    with wave.open(io.BytesIO(file_bytes), "rb") as wav_file:
        sample_rate_hz = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width_bytes = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()

        if sample_width_bytes not in (1, 2, 4):
            raise ValueError("Only 8-bit, 16-bit, and 32-bit PCM WAV files are supported.")

        raw_frames = wav_file.readframes(frame_count)

    samples = _pcm_to_mono_samples(
        raw_frames=raw_frames,
        num_channels=num_channels,
        sample_width_bytes=sample_width_bytes,
    )

    return DecodedAudio(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
    )


def extract_features(samples: list[float], sample_rate_hz: int) -> ComputedFeatures:
    if not samples:
        return ComputedFeatures(
            rms=0.0,
            peak_amplitude=0.0,
            zero_crossing_rate=0.0,
            dominant_activity_ratio=0.0,
        )

    peak_amplitude = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    zero_crossings = 0
    for previous_sample, current_sample in zip(samples, samples[1:]):
        if (previous_sample < 0.0 and current_sample >= 0.0) or (
            previous_sample >= 0.0 and current_sample < 0.0
        ):
            zero_crossings += 1

    zero_crossing_rate = zero_crossings / max(1, len(samples) - 1)

    window_size = max(1, int(sample_rate_hz * 0.05))
    activity_threshold = max(0.02, rms * 0.75)
    active_windows = 0
    total_windows = 0

    for start in range(0, len(samples), window_size):
        window = samples[start : start + window_size]
        if not window:
            continue

        total_windows += 1
        window_rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        if window_rms >= activity_threshold:
            active_windows += 1

    dominant_activity_ratio = active_windows / max(1, total_windows)

    return ComputedFeatures(
        rms=round(rms, 6),
        peak_amplitude=round(peak_amplitude, 6),
        zero_crossing_rate=round(zero_crossing_rate, 6),
        dominant_activity_ratio=round(dominant_activity_ratio, 6),
    )


def preprocess_audio(
    samples: list[float], sample_rate_hz: int, target_sample_rate_hz: int
) -> PreprocessedAudio:
    original_peak_amplitude = max((abs(sample) for sample in samples), default=0.0)

    normalized_samples = normalize_waveform(samples)
    normalized_peak_amplitude = max(
        (abs(sample) for sample in normalized_samples),
        default=0.0,
    )
    normalization_gain = (
        normalized_peak_amplitude / original_peak_amplitude
        if original_peak_amplitude > 0.0
        else 1.0
    )

    was_resampled = sample_rate_hz != target_sample_rate_hz
    processed_samples = (
        resample_waveform(normalized_samples, sample_rate_hz, target_sample_rate_hz)
        if was_resampled
        else normalized_samples
    )

    return PreprocessedAudio(
        samples=processed_samples,
        sample_rate_hz=target_sample_rate_hz,
        original_peak_amplitude=round(original_peak_amplitude, 6),
        normalized_peak_amplitude=round(normalized_peak_amplitude, 6),
        was_resampled=was_resampled,
        normalization_gain=round(normalization_gain, 6),
    )


def extract_log_mel_features(
    samples: list[float],
    sample_rate_hz: int,
    frame_size: int = 400,
    hop_size: int = 160,
    fft_size: int = 512,
    mel_bin_count: int = 40,
    min_frequency_hz: float = 20.0,
) -> SpectralFeatures:
    if np is not None:
        return _extract_log_mel_features_numpy(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            frame_size=frame_size,
            hop_size=hop_size,
            fft_size=fft_size,
            mel_bin_count=mel_bin_count,
            min_frequency_hz=min_frequency_hz,
        )

    return _extract_log_mel_features_python(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        frame_size=frame_size,
        hop_size=hop_size,
        fft_size=fft_size,
        mel_bin_count=mel_bin_count,
        min_frequency_hz=min_frequency_hz,
    )


def _extract_log_mel_features_numpy(
    samples: list[float],
    sample_rate_hz: int,
    frame_size: int,
    hop_size: int,
    fft_size: int,
    mel_bin_count: int,
    min_frequency_hz: float,
) -> SpectralFeatures:
    if not samples or sample_rate_hz <= 0:
        return _empty_spectral_features(
            mel_bin_count=mel_bin_count,
            frame_size=frame_size,
            hop_size=hop_size,
            fft_size=fft_size,
        )

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.size < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.size))

    if waveform.size <= frame_size:
        frames = waveform[:frame_size].reshape(1, frame_size)
    else:
        frame_count = 1 + math.ceil((waveform.size - frame_size) / hop_size)
        padded_length = frame_size + ((frame_count - 1) * hop_size)
        if padded_length > waveform.size:
            waveform = np.pad(waveform, (0, padded_length - waveform.size))
        shape = (frame_count, frame_size)
        strides = (waveform.strides[0] * hop_size, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            waveform, shape=shape, strides=strides
        ).copy()

    window = np.hanning(frame_size).astype(np.float32)
    windowed_frames = frames * window
    power_spectrum = np.abs(np.fft.rfft(windowed_frames, n=fft_size, axis=1)) ** 2
    mel_filter_bank = _build_mel_filter_bank_numpy(
        sample_rate_hz=sample_rate_hz,
        fft_size=fft_size,
        mel_bin_count=mel_bin_count,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=sample_rate_hz / 2.0,
    )
    mel_energies = np.maximum(power_spectrum @ mel_filter_bank.T, 1e-10)
    log_mel = 10.0 * np.log10(mel_energies)
    return _spectral_summary_from_array(
        log_mel=log_mel,
        frame_size=frame_size,
        hop_size=hop_size,
        fft_size=fft_size,
    )


def _extract_log_mel_features_python(
    samples: list[float],
    sample_rate_hz: int,
    frame_size: int,
    hop_size: int,
    fft_size: int,
    mel_bin_count: int,
    min_frequency_hz: float,
) -> SpectralFeatures:
    if not samples or sample_rate_hz <= 0:
        return _empty_spectral_features(
            mel_bin_count=mel_bin_count,
            frame_size=frame_size,
            hop_size=hop_size,
            fft_size=fft_size,
        )

    waveform = samples[:]
    if len(waveform) < frame_size:
        waveform.extend([0.0] * (frame_size - len(waveform)))

    window = _hann_window(frame_size)
    mel_filter_bank = _build_mel_filter_bank(
        sample_rate_hz=sample_rate_hz,
        fft_size=fft_size,
        mel_bin_count=mel_bin_count,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=sample_rate_hz / 2.0,
    )

    mel_frames: list[list[float]] = []
    for start in range(0, max(1, len(waveform) - frame_size + 1), hop_size):
        frame = waveform[start : start + frame_size]
        if len(frame) < frame_size:
            frame = frame + ([0.0] * (frame_size - len(frame)))
        windowed_frame = [sample * weight for sample, weight in zip(frame, window)]
        power_spectrum = _power_spectrum(windowed_frame, fft_size)
        mel_energies = _apply_mel_filter_bank(mel_filter_bank, power_spectrum)
        mel_frames.append(
            [10.0 * math.log10(max(energy, 1e-10)) for energy in mel_energies]
        )

    if not mel_frames:
        mel_frames.append([0.0] * mel_bin_count)

    flattened_values = [value for frame in mel_frames for value in frame]
    low_band_mean_db, mid_band_mean_db, high_band_mean_db = _band_mean_levels(
        mel_frames
    )
    min_db = round(min(flattened_values), 6)
    max_db = round(max(flattened_values), 6)
    return SpectralFeatures(
        frame_count=len(mel_frames),
        mel_bin_count=mel_bin_count,
        frame_size=frame_size,
        hop_size=hop_size,
        fft_size=fft_size,
        min_db=min_db,
        max_db=max_db,
        mean_db=round(sum(flattened_values) / len(flattened_values), 6),
        dynamic_range_db=round(max_db - min_db, 6),
        low_band_mean_db=round(low_band_mean_db, 6),
        mid_band_mean_db=round(mid_band_mean_db, 6),
        high_band_mean_db=round(high_band_mean_db, 6),
    )


def normalize_waveform(samples: list[float], target_peak: float = 0.95) -> list[float]:
    if not samples:
        return []

    peak_amplitude = max(abs(sample) for sample in samples)
    if peak_amplitude <= 0.0:
        return [0.0 for _ in samples]

    gain = target_peak / peak_amplitude
    return [max(-1.0, min(1.0, sample * gain)) for sample in samples]


def resample_waveform(
    samples: list[float], source_sample_rate_hz: int, target_sample_rate_hz: int
) -> list[float]:
    if not samples or source_sample_rate_hz <= 0 or target_sample_rate_hz <= 0:
        return samples[:]

    if source_sample_rate_hz == target_sample_rate_hz or len(samples) == 1:
        return samples[:]

    target_length = max(
        1,
        int(round(len(samples) * target_sample_rate_hz / source_sample_rate_hz)),
    )
    resampled_samples: list[float] = []
    source_last_index = len(samples) - 1

    for target_index in range(target_length):
        source_position = (
            target_index * (len(samples) - 1) / max(1, target_length - 1)
        )
        lower_index = int(source_position)
        upper_index = min(lower_index + 1, source_last_index)
        blend = source_position - lower_index
        sample = (samples[lower_index] * (1.0 - blend)) + (
            samples[upper_index] * blend
        )
        resampled_samples.append(sample)

    return resampled_samples


def _pcm_to_mono_samples(
    raw_frames: bytes, num_channels: int, sample_width_bytes: int
) -> list[float]:
    bytes_per_frame = num_channels * sample_width_bytes
    if bytes_per_frame == 0:
        return []

    frame_total = len(raw_frames) // bytes_per_frame
    samples: list[float] = []

    for frame_index in range(frame_total):
        frame_offset = frame_index * bytes_per_frame
        channel_values: list[float] = []

        for channel_index in range(num_channels):
            start = frame_offset + (channel_index * sample_width_bytes)
            end = start + sample_width_bytes
            chunk = raw_frames[start:end]
            channel_values.append(_decode_sample(chunk, sample_width_bytes))

        samples.append(sum(channel_values) / len(channel_values))

    return samples


def _decode_sample(sample_bytes: bytes, sample_width_bytes: int) -> float:
    if sample_width_bytes == 1:
        raw_value = sample_bytes[0] - 128
        return raw_value / 128.0

    raw_value = int.from_bytes(sample_bytes, byteorder="little", signed=True)
    max_value = float(2 ** ((sample_width_bytes * 8) - 1))
    return raw_value / max_value


def _build_mel_filter_bank(
    sample_rate_hz: int,
    fft_size: int,
    mel_bin_count: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> list[list[float]]:
    mel_min = _hz_to_mel(min_frequency_hz)
    mel_max = _hz_to_mel(max_frequency_hz)
    mel_points = [
        mel_min + ((mel_max - mel_min) * index / (mel_bin_count + 1))
        for index in range(mel_bin_count + 2)
    ]
    hz_points = [_mel_to_hz(mel_value) for mel_value in mel_points]
    fft_bin_frequencies = [
        int(math.floor(((fft_size + 1) * hz_point) / sample_rate_hz))
        for hz_point in hz_points
    ]

    filter_bank = [
        [0.0 for _ in range((fft_size // 2) + 1)] for _ in range(mel_bin_count)
    ]

    for mel_index in range(mel_bin_count):
        left = fft_bin_frequencies[mel_index]
        center = fft_bin_frequencies[mel_index + 1]
        right = fft_bin_frequencies[mel_index + 2]

        if center <= left:
            center = min(left + 1, len(filter_bank[mel_index]) - 1)
        if right <= center:
            right = min(center + 1, len(filter_bank[mel_index]))

        for frequency_bin in range(left, center):
            filter_bank[mel_index][frequency_bin] = (frequency_bin - left) / max(
                1, center - left
            )

        for frequency_bin in range(center, right):
            filter_bank[mel_index][frequency_bin] = (right - frequency_bin) / max(
                1, right - center
            )

    return filter_bank


def _build_mel_filter_bank_numpy(
    sample_rate_hz: int,
    fft_size: int,
    mel_bin_count: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> np.ndarray:
    mel_min = _hz_to_mel(min_frequency_hz)
    mel_max = _hz_to_mel(max_frequency_hz)
    mel_points = np.linspace(mel_min, mel_max, mel_bin_count + 2, dtype=np.float32)
    hz_points = _mel_to_hz_numpy(mel_points)
    fft_bin_frequencies = np.floor(((fft_size + 1) * hz_points) / sample_rate_hz).astype(
        np.int32
    )

    filter_bank = np.zeros((mel_bin_count, (fft_size // 2) + 1), dtype=np.float32)

    for mel_index in range(mel_bin_count):
        left = int(fft_bin_frequencies[mel_index])
        center = int(fft_bin_frequencies[mel_index + 1])
        right = int(fft_bin_frequencies[mel_index + 2])

        if center <= left:
            center = min(left + 1, filter_bank.shape[1] - 1)
        if right <= center:
            right = min(center + 1, filter_bank.shape[1])

        if center > left:
            filter_bank[mel_index, left:center] = np.linspace(
                0.0,
                1.0,
                center - left,
                endpoint=False,
                dtype=np.float32,
            )

        if right > center:
            filter_bank[mel_index, center:right] = np.linspace(
                1.0,
                0.0,
                right - center,
                endpoint=False,
                dtype=np.float32,
            )

    return filter_bank


def _hz_to_mel(frequency_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + (frequency_hz / 700.0))


def _mel_to_hz(mel_value: float) -> float:
    return 700.0 * ((10.0 ** (mel_value / 2595.0)) - 1.0)


def _mel_to_hz_numpy(mel_values: np.ndarray) -> np.ndarray:
    return 700.0 * ((10.0 ** (mel_values / 2595.0)) - 1.0)


def _hann_window(frame_size: int) -> list[float]:
    if frame_size <= 1:
        return [1.0] * max(1, frame_size)
    return [
        0.5 - (0.5 * math.cos((2.0 * math.pi * index) / (frame_size - 1)))
        for index in range(frame_size)
    ]


def _power_spectrum(frame: list[float], fft_size: int) -> list[float]:
    padded_frame = frame[:fft_size] + ([0.0] * max(0, fft_size - len(frame)))
    spectrum: list[float] = []

    for frequency_bin in range((fft_size // 2) + 1):
        frequency_value = 0j
        for sample_index, sample in enumerate(padded_frame):
            angle = -2j * cmath.pi * frequency_bin * sample_index / fft_size
            frequency_value += sample * cmath.exp(angle)
        spectrum.append((frequency_value.real**2) + (frequency_value.imag**2))

    return spectrum


def _apply_mel_filter_bank(
    filter_bank: list[list[float]], power_spectrum: list[float]
) -> list[float]:
    mel_energies: list[float] = []

    for filter_row in filter_bank:
        energy = 0.0
        for weight, power in zip(filter_row, power_spectrum):
            energy += weight * power
        mel_energies.append(energy)

    return mel_energies


def _band_mean_levels(mel_frames: list[list[float]]) -> tuple[float, float, float]:
    if not mel_frames or not mel_frames[0]:
        return 0.0, 0.0, 0.0

    mel_bin_count = len(mel_frames[0])
    low_end = max(1, mel_bin_count // 3)
    mid_end = max(low_end + 1, (mel_bin_count * 2) // 3)

    def average_band(start: int, end: int) -> float:
        values = [value for frame in mel_frames for value in frame[start:end]]
        if not values:
            return 0.0
        return sum(values) / len(values)

    return (
        average_band(0, low_end),
        average_band(low_end, mid_end),
        average_band(mid_end, mel_bin_count),
    )


def _spectral_summary_from_array(
    log_mel: np.ndarray, frame_size: int, hop_size: int, fft_size: int
) -> SpectralFeatures:
    mel_bin_count = int(log_mel.shape[1])
    low_end = max(1, mel_bin_count // 3)
    mid_end = max(low_end + 1, (mel_bin_count * 2) // 3)
    min_db = round(float(log_mel.min()), 6)
    max_db = round(float(log_mel.max()), 6)
    return SpectralFeatures(
        frame_count=int(log_mel.shape[0]),
        mel_bin_count=mel_bin_count,
        frame_size=frame_size,
        hop_size=hop_size,
        fft_size=fft_size,
        min_db=min_db,
        max_db=max_db,
        mean_db=round(float(log_mel.mean()), 6),
        dynamic_range_db=round(max_db - min_db, 6),
        low_band_mean_db=round(float(log_mel[:, :low_end].mean()), 6),
        mid_band_mean_db=round(float(log_mel[:, low_end:mid_end].mean()), 6),
        high_band_mean_db=round(float(log_mel[:, mid_end:].mean()), 6),
    )


def _empty_spectral_features(
    mel_bin_count: int, frame_size: int, hop_size: int, fft_size: int
) -> SpectralFeatures:
    return SpectralFeatures(
        frame_count=0,
        mel_bin_count=mel_bin_count,
        frame_size=frame_size,
        hop_size=hop_size,
        fft_size=fft_size,
        min_db=0.0,
        max_db=0.0,
        mean_db=0.0,
        dynamic_range_db=0.0,
        low_band_mean_db=0.0,
        mid_band_mean_db=0.0,
        high_band_mean_db=0.0,
    )
