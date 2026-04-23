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
    args = parse_args()
    training_result = train_waveform_model(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        progress_callback=_print_progress_update,
    )
    print(f"Saved model artifact to {training_result['output_dir']}")


def train_waveform_model(
    *,
    manifest_path: str,
    output_dir: str,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    progress_callback=None,
) -> dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install training/requirements.txt first."
        ) from exc

    examples = load_manifest(manifest_path)
    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split == "val"]
    if not train_examples:
        raise SystemExit("The manifest must contain at least one training example.")

    class_names = sorted({example.label for example in examples})
    label_to_index = {label: index for index, label in enumerate(class_names)}
    input_sample_count = int(
        settings.sample_rate_hz * (settings.chunk_duration_ms / 1000.0)
    )

    def _preload_dataset(subset, subset_name: str):
        waveforms = []
        targets = []
        total = len(subset)
        for index, example in enumerate(subset, start=1):
            waveform = load_preprocessed_waveform(
                audio_path=example.audio_path,
                sample_rate_hz=settings.sample_rate_hz,
                input_sample_count=input_sample_count,
            )
            waveforms.append(waveform)
            targets.append(label_to_index[example.label])
            if total and (index == total or index % 250 == 0):
                print(f"preload subset={subset_name} {index}/{total}")
        return torch.tensor(waveforms, dtype=torch.float32).unsqueeze(1), torch.tensor(
            targets,
            dtype=torch.long,
        )

    class WaveformDataset(Dataset):
        def __init__(self, waveforms, targets):
            self.waveforms = waveforms
            self.targets = targets

        def __len__(self) -> int:
            return int(self.targets.shape[0])

        def __getitem__(self, index: int):
            return self.waveforms[index], self.targets[index]

    train_waveforms, train_targets = _preload_dataset(train_examples, "train")
    validation_source = val_examples or train_examples
    validation_name = "val" if val_examples else "train_fallback"
    val_waveforms, val_targets = _preload_dataset(validation_source, validation_name)

    train_loader = DataLoader(
        WaveformDataset(train_waveforms, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WaveformDataset(val_waveforms, val_targets),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_waveform_cnn(class_count=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_metrics: list[dict[str, float | int]] = []

    for epoch_index in range(epochs):
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
        epoch_result = {
            "epoch": epoch_index + 1,
            "loss": running_loss / max(1, len(train_loader)),
            "val_accuracy": validation_accuracy,
        }
        epoch_metrics.append(epoch_result)
        if progress_callback is not None:
            progress_callback(epoch_result)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    example_input = torch.zeros(1, 1, input_sample_count, dtype=torch.float32)
    scripted_model = torch.jit.trace(model.eval(), example_input)
    weights_path = output_dir_path / "model.ts"
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
    (output_dir_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir_path),
        "weights_path": str(weights_path),
        "manifest_path": str((output_dir_path / "manifest.json").resolve()),
        "class_names": class_names,
        "training_example_count": len(train_examples),
        "validation_example_count": len(val_examples),
        "epoch_metrics": epoch_metrics,
        "final_val_accuracy": epoch_metrics[-1]["val_accuracy"] if epoch_metrics else 0.0,
    }


def _print_progress_update(epoch_result: dict[str, float | int]) -> None:
    print(
        f"epoch={epoch_result['epoch']} "
        f"loss={float(epoch_result['loss']):.4f} "
        f"val_accuracy={float(epoch_result['val_accuracy']):.4f}"
    )


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
