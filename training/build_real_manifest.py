from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from backend.app.config import settings
from training.dataset import build_examples_from_labeled_directory, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest from real labeled WAV recordings.",
    )
    parser.add_argument(
        "--source-dir",
        default="training/real_recordings",
        help=(
            "Directory containing either <label>/*.wav or "
            "<split>/<label>/*.wav recordings."
        ),
    )
    parser.add_argument(
        "--output-manifest",
        default="training/real_recordings/manifest.jsonl",
        help="Destination JSONL manifest path.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio when source-dir is not already pre-split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio when source-dir is not already pre-split.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples_from_labeled_directory(
        args.source_dir,
        supported_classes=settings.supported_classes,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if not examples:
        raise SystemExit(
            "No WAV files were found. Place recordings in training/real_recordings/<label>/"
            " or training/real_recordings/<split>/<label>/ and rerun."
        )

    write_manifest(
        examples=examples,
        manifest_path=args.output_manifest,
        base_dir=args.source_dir,
    )

    by_split = Counter(example.split for example in examples)
    by_label = Counter(example.label for example in examples)
    label_split_table: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        label_split_table[example.label][example.split] += 1

    print(f"Wrote manifest: {args.output_manifest}")
    print(f"Total examples: {len(examples)}")
    print(f"By split: {dict(sorted(by_split.items()))}")
    print(f"By label: {dict(sorted(by_label.items()))}")
    print("Per-label splits:")
    for label in settings.supported_classes:
        counts = label_split_table.get(label)
        if not counts:
            continue
        print(
            f"  {label}: "
            f"train={counts.get('train', 0)} "
            f"val={counts.get('val', 0)} "
            f"test={counts.get('test', 0)}"
        )


if __name__ == "__main__":
    main()
