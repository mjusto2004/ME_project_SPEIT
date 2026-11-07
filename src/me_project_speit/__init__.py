"""me_project_speit: deep-learning surrogate for transient temperature fields."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("me-project-speit")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
