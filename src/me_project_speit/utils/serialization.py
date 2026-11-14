"""Helpers for saving plain-Python data."""

from __future__ import annotations


def to_plain(obj):
    """Convert tuples to lists so checkpoints load with ``weights_only=True``."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
