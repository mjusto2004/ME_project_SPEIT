"""Helpers for naming and locating run folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def timestamp() -> str:
    """Current local time as ``YYYYMMDD_HHMMSS``, used to name datasets and runs."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def latest_run(runs_dir: str | Path = "runs") -> Path:
    """The most recently modified run folder in ``runs_dir``."""
    runs = [p for p in Path(runs_dir).glob("*") if (p / "best_model.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No run (folder with best_model.pt) found in '{runs_dir}'.")
    return max(runs, key=lambda p: (p / "best_model.pt").stat().st_mtime)
