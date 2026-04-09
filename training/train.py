from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.app.config import settings
from training.dataset import load_manifest, load_preprocessed_waveform
from training.model import build_waveform_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a waveform classifier for the sound dashboard.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the JSONL dataset manifest.")
    parser.add_argument(
        "--output-dir",
        default="training/artifacts/latest",
        help="Directory to write the exported model and manifest.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install training/requirements.txt first."
        ) from exc

    args = parse_args()
    examples = load_manifest(args.manifest)
    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split == "val"]
    if not train_examples:
        raise SystemExit("The manifest must contain at least one training example.")

    class_names = sorted({example.label for example in examples})
    label_to_index = {label: index for index, label in enumerate(class_names)}
    input_sample_count = int(
        settings.sample_rate_hz * (settings.chunk_duration_ms / 1000.0)
    )

    class WaveformDataset(Dataset):
        def __init__(self, subset):
            self.subset = subset

        def __len__(self) -> int:
            return len(self.subset)

        def __getitem__(self, index: int):
            example = self.subset[index]
            waveform = load_preprocessed_waveform(
                audio_path=example.audio_path,
                sample_rate_hz=settings.sample_rate_hz,
                input_sample_count=input_sample_count,
            )
            return (
                torch.tensor(waveform, dtype=torch.float32).unsqueeze(0),
                torch.tensor(label_to_index[example.label], dtype=torch.long),
            )

    train_loader = DataLoader(
        WaveformDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WaveformDataset(val_examples or train_examples),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_waveform_cnn(class_count=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch_index in range(args.epochs):
        model.train()
        running_loss = 0.0
        for waveforms, targets in train_loader:
            optimizer.zero_grad()
            logits = model(waveforms)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        validation_accuracy = evaluate(model, val_loader)
        print(
            f"epoch={epoch_index + 1} "
            f"loss={running_loss / max(1, len(train_loader)):.4f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    example_input = torch.zeros(1, 1, input_sample_count, dtype=torch.float32)
    scripted_model = torch.jit.trace(model.eval(), example_input)
    weights_path = output_dir / "model.ts"
    scripted_model.save(str(weights_path))

    manifest = {
        "model_name": "waveform_cnn_v1",
        "model_type": "torchscript_waveform_cnn",
        "class_names": class_names,
        "sample_rate_hz": settings.sample_rate_hz,
        "input_sample_count": input_sample_count,
        "confidence_threshold": settings.classifier_confidence_threshold,
        "weights_path": weights_path.name,
        "normalization_target_peak": 0.95,
        "training_example_count": len(train_examples),
        "validation_example_count": len(val_examples),
        "class_distribution": dict(Counter(example.label for example in train_examples)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model artifact to {output_dir}")


def evaluate(model, loader) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, targets in loader:
            predictions = model(waveforms).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct / max(1, total)


if __name__ == "__main__":
    main()
