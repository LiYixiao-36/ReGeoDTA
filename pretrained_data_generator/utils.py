"""Shared runtime utilities for pretraining-data generation."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def should_skip(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.is_file()
