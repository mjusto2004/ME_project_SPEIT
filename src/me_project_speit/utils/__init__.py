"""Generic helpers shared across the package.

Leaf package: it must not import from other ``me_project_speit`` modules.
"""

from .device import select_device
from .parallel import default_workers, parallel_map
from .runs import latest_run, timestamp
from .serialization import to_plain

__all__ = [
    "default_workers",
    "latest_run",
    "parallel_map",
    "select_device",
    "timestamp",
    "to_plain",
]
