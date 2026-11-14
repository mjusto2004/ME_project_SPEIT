"""Compute-device selection."""

from __future__ import annotations

import torch


def select_device(name: str) -> torch.device:
    """Resolve ``"auto" | "cuda" | "cpu"`` to a :class:`torch.device`."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Request a GPU through Slurm.")
    use_cuda = name == "cuda" or (name == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")
