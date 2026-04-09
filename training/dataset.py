from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from backend.app.audio import decode_wav, preprocess_audio


@dataclass
class AudioExample:
    audio_path: Path
    label: str
    split: str


SUPPORTED_SPLITS = {"train", "val", "test"}


def load_manifest(manifest_path: str) -> list[AudioExample]:
    path = Path(manifest_path)
    examples: list[AudioExample] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        payload = json.loads(line)
        audio_path = (path.parent / payload["audio_path"]).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Manifest line {line_number} references missing audio file: {audio_path}"
            )
        examples.append(
            AudioExample(
                audio_path=audio_path,
                label=str(payload["label"]),
                split=str(payload.get("split", "train")),
            )
        )
    return examples


def build_examples_from_labeled_directory(
    source_dir: str,
    *,
    supported_classes: list[str],
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 7,
) -> list[AudioExample]:
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    examples: list[AudioExample] = []
    split_root_names = SUPPORTED_SPLITS
    top_level_names = {child.name for child in source_path.iterdir() if child.is_dir()}

    if top_level_names & split_root_names:
        for split_name in sorted(top_level_names & split_root_names):
            split_root = source_path / split_name
            for label_dir in sorted(child for child in split_root.iterdir() if child.is_dir()):
                label = label_dir.name
                _validate_label(label=label, supported_classes=supported_classes)
                for audio_path in _iter_audio_files(label_dir):
                    examples.append(
                        AudioExample(
                            audio_path=audio_path.resolve(),
                            label=label,
                            split=split_name,
                        )
                    )
        return examples

    rng = random.Random(seed)
    for label_dir in sorted(child for child in source_path.iterdir() if child.is_dir()):
        label = label_dir.name
        _validate_label(label=label, supported_classes=supported_classes)
        label_files = [path.resolve() for path in _iter_audio_files(label_dir)]
        rng.shuffle(label_files)
        for index, audio_path in enumerate(label_files):
            examples.append(
                AudioExample(
                    audio_path=audio_path,
                    label=label,
                    split=_choose_split_for_index(
                        index=index,
                        total=len(label_files),
                        validation_ratio=validation_ratio,
                        test_ratio=test_ratio,
                    ),
                )
            )

    return examples


def write_manifest(examples: list[AudioExample], manifest_path: str, base_dir: str) -> None:
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_dir).resolve()
    lines = []
    for example in examples:
        lines.append(
            json.dumps(
                {
                    "audio_path": example.audio_path.resolve().relative_to(base_path).as_posix(),
                    "label": example.label,
                    "split": example.split,
                }
            )
        )
    manifest_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_preprocessed_waveform(
    audio_path: Path,
    sample_rate_hz: int,
    input_sample_count: int,
) -> list[float]:
    decoded = decode_wav(audio_path.read_bytes())
    processed = preprocess_audio(
        samples=decoded.samples,
        sample_rate_hz=decoded.sample_rate_hz,
        target_sample_rate_hz=sample_rate_hz,
    )
    waveform = processed.samples[:input_sample_count]
    if len(waveform) < input_sample_count:
        waveform.extend([0.0] * (input_sample_count - len(waveform)))
    return waveform


def _iter_audio_files(root: Path) -> list[Path]:
    audio_paths = {path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"}
    return sorted(audio_paths)


def _validate_label(label: str, supported_classes: list[str]) -> None:
    if label not in supported_classes:
        supported = ", ".join(supported_classes)
        raise ValueError(f"Unsupported label '{label}'. Supported classes: {supported}")


def _choose_split_for_index(
    *,
    index: int,
    total: int,
    validation_ratio: float,
    test_ratio: float,
) -> str:
    if total <= 1:
        return "train"

    test_cutoff = max(1, round(total * test_ratio)) if test_ratio > 0.0 else 0
    val_cutoff = max(1, round(total * validation_ratio)) if validation_ratio > 0.0 else 0

    if index < test_cutoff:
        return "test"
    if index < test_cutoff + val_cutoff:
        return "val"
    return "train"
