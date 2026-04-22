from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite WAV files as PCM 16-bit WAV files in place.",
    )
    parser.add_argument(
        "--source-dir",
        default="training/real_recordings",
        help="Root directory containing WAV files to rewrite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    converted = 0
    skipped = 0
    for wav_path in sorted(source_dir.rglob("*.wav")):
        try:
            with sf.SoundFile(str(wav_path)) as handle:
                subtype = str(handle.subtype or "")
                format_name = str(handle.format or "")
            if format_name.upper() == "WAV" and subtype.upper() == "PCM_16":
                skipped += 1
                continue

            data, sample_rate = sf.read(str(wav_path))
            sf.write(str(wav_path), data, sample_rate, subtype="PCM_16")
            converted += 1
            print(f"Converted {wav_path}")
        except Exception as exc:
            print(f"Skipped {wav_path}: {exc}")

    print(
        f"PCM conversion complete. converted={converted} skipped_already_pcm16={skipped}"
    )


if __name__ == "__main__":
    main()
