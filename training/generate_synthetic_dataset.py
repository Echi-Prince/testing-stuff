from __future__ import annotations

import argparse
import json
import math
import random
import wave
from pathlib import Path


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_DURATION_SECONDS = 1.0
SUPPORTED_CLASSES = [
    "speech",
    "keyboard",
    "dog_bark",
    "traffic",
    "siren",
    "vacuum",
    "music",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic WAV dataset and JSONL manifest for local training."
    )
    parser.add_argument(
        "--output-dir",
        default="training/synthetic_data",
        help="Directory that will receive the WAV files and manifest.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=6,
        help="Number of examples to generate for each class.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Output sample rate in Hz.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Clip duration for each generated sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_lines: list[str] = []

    for label in SUPPORTED_CLASSES:
        for example_index in range(args.examples_per_class):
            split = choose_split(example_index)
            relative_path = Path(split) / label / f"{label}-{example_index + 1:03d}.wav"
            absolute_path = output_dir / relative_path
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            samples = generate_class_waveform(
                label=label,
                sample_rate_hz=args.sample_rate,
                duration_seconds=args.duration_seconds,
            )
            write_wav(
                path=absolute_path,
                sample_rate_hz=args.sample_rate,
                samples=samples,
            )
            manifest_lines.append(
                json.dumps(
                    {
                        "audio_path": relative_path.as_posix(),
                        "label": label,
                        "split": split,
                    }
                )
            )

    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Generated {len(manifest_lines)} examples at {output_dir}")
    print(f"Manifest: {manifest_path}")


def choose_split(example_index: int) -> str:
    if example_index == 0:
        return "test"
    if example_index == 1:
        return "val"
    return "train"


def generate_class_waveform(
    *,
    label: str,
    sample_rate_hz: int,
    duration_seconds: float,
) -> list[float]:
    frame_count = max(1, int(sample_rate_hz * duration_seconds))
    time_axis = [index / sample_rate_hz for index in range(frame_count)]
    noise = [random.uniform(-1.0, 1.0) for _ in range(frame_count)]

    if label == "speech":
        base = sine_mix(time_axis, [(180.0, 0.45), (320.0, 0.22), (650.0, 0.1)])
        envelope = pulse_envelope(time_axis, rate_hz=4.5, floor=0.15)
        return scale_waveform(mix([base, multiply(base, envelope), scale(noise, 0.03)]))

    if label == "keyboard":
        waveform = [0.0] * frame_count
        burst_count = max(4, int(duration_seconds * 8))
        for _ in range(burst_count):
            start = random.randint(0, max(0, frame_count - 400))
            for offset in range(320):
                index = start + offset
                if index >= frame_count:
                    break
                decay = math.exp(-offset / 38.0)
                waveform[index] += random.choice((-1.0, 1.0)) * decay * (0.6 + random.random() * 0.3)
        return scale_waveform(mix([waveform, scale(noise, 0.015)]))

    if label == "dog_bark":
        waveform = [0.0] * frame_count
        bark_count = max(2, int(duration_seconds * 2.5))
        for bark_index in range(bark_count):
            start = int((bark_index + 0.2) * frame_count / (bark_count + 0.6))
            for offset in range(int(sample_rate_hz * 0.12)):
                index = start + offset
                if index >= frame_count:
                    break
                tone = math.sin((2.0 * math.pi * (420.0 + (offset * 0.7)) * index) / sample_rate_hz)
                envelope = math.exp(-offset / 250.0)
                waveform[index] += tone * envelope * 0.95
        return scale_waveform(mix([waveform, scale(noise, 0.025)]))

    if label == "traffic":
        low = sine_mix(time_axis, [(55.0, 0.4), (85.0, 0.28), (120.0, 0.18)])
        rumble = moving_average(noise, 90)
        return scale_waveform(mix([low, scale(rumble, 0.32)]))

    if label == "siren":
        waveform = []
        for t in time_axis:
            sweep = 720.0 + (math.sin(2.0 * math.pi * 1.4 * t) * 260.0)
            waveform.append(
                (0.55 * math.sin(2.0 * math.pi * sweep * t))
                + (0.2 * math.sin(2.0 * math.pi * (sweep * 1.5) * t))
            )
        return scale_waveform(waveform)

    if label == "vacuum":
        drone = sine_mix(time_axis, [(140.0, 0.35), (280.0, 0.18), (420.0, 0.12)])
        air = moving_average(noise, 18)
        return scale_waveform(mix([drone, scale(air, 0.25)]))

    if label == "music":
        melody = sine_mix(time_axis, [(261.63, 0.22), (329.63, 0.2), (392.0, 0.18), (523.25, 0.12)])
        rhythm = pulse_envelope(time_axis, rate_hz=2.2, floor=0.3)
        return scale_waveform(mix([multiply(melody, rhythm), scale(noise, 0.015)]))

    raise ValueError(f"Unsupported label: {label}")


def write_wav(path: Path, sample_rate_hz: int, samples: list[float]) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        frames = bytearray()
        for sample in samples:
            pcm_value = int(max(-1.0, min(1.0, sample)) * 32767.0)
            frames.extend(pcm_value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))


def sine_mix(time_axis: list[float], components: list[tuple[float, float]]) -> list[float]:
    return [
        sum(amplitude * math.sin(2.0 * math.pi * frequency_hz * time_value) for frequency_hz, amplitude in components)
        for time_value in time_axis
    ]


def pulse_envelope(time_axis: list[float], rate_hz: float, floor: float) -> list[float]:
    return [
        max(floor, 0.5 + (0.5 * math.sin(2.0 * math.pi * rate_hz * time_value)))
        for time_value in time_axis
    ]


def moving_average(samples: list[float], window_size: int) -> list[float]:
    if window_size <= 1:
        return samples[:]

    averaged: list[float] = []
    accumulator = 0.0
    window: list[float] = []
    for sample in samples:
        window.append(sample)
        accumulator += sample
        if len(window) > window_size:
            accumulator -= window.pop(0)
        averaged.append(accumulator / len(window))
    return averaged


def scale(samples: list[float], factor: float) -> list[float]:
    return [sample * factor for sample in samples]


def multiply(samples: list[float], envelope: list[float]) -> list[float]:
    return [sample * level for sample, level in zip(samples, envelope)]


def mix(signals: list[list[float]]) -> list[float]:
    return [sum(values) for values in zip(*signals)]


def scale_waveform(samples: list[float], target_peak: float = 0.95) -> list[float]:
    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak <= 0.0:
        return samples[:]
    gain = target_peak / peak
    return [sample * gain for sample in samples]


if __name__ == "__main__":
    main()
