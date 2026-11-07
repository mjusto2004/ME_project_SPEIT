"""Neural network architectures.

To add a model, create a module in this package defining a
:class:`SurrogateModel` subclass decorated with ``@register_model("name")``,
then import it below so that it registers itself.
"""

from .base import IncompatibleError, SurrogateModel
from .registry import (
    available_models,
    build_model,
    check_compatible,
    compatible_models,
    get_model_class,
    register_model,
)

# Importing the modules registers the models.
from .mlp import MLP  # noqa: E402

__all__ = [
    "MLP",
    "IncompatibleError",
    "SurrogateModel",
    "available_models",
    "build_model",
    "check_compatible",
    "compatible_models",
    "get_model_class",
    "register_model",
]
