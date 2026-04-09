from __future__ import annotations


def build_waveform_cnn(class_count: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(32),
        nn.Flatten(),
        nn.Linear(64 * 32, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, class_count),
    )
