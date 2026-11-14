"""Run a function on many inputs, in several processes."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterator


def default_workers() -> int:
    """Number of worker processes used when the user does not choose: the number of cores, at most 8."""
    return max(1, min(os.cpu_count() or 1, 8))


def parallel_map(func: Callable, *iterables, workers: int = 1) -> Iterator:
    """Like ``map(func, *iterables)``, with ``workers`` processes.

    The results come back **in the input order**, one by one as they become
    available, so the output does not depend on ``workers``. With
    ``workers <= 1`` everything runs in the current process (no pool, easier
    to debug).

    ``func`` and the inputs must be picklable: ``func`` has to be defined at the
    top level of a module (or be a method of a class defined there).
    """
    if workers <= 1:
        yield from map(func, *iterables)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(func, *iterables)
