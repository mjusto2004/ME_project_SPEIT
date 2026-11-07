"""Name -> model class registry, with compatibility checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import IncompatibleError, SurrogateModel

if TYPE_CHECKING:
    from ..dataset import DatasetSpec

_REGISTRY: dict[str, type[SurrogateModel]] = {}


def register_model(name: str):
    """Class decorator registering a model under ``name``."""

    def decorator(cls: type[SurrogateModel]) -> type[SurrogateModel]:
        if name in _REGISTRY:
            raise ValueError(f"A model named '{name}' is already registered.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model_class(name: str) -> type[SurrogateModel]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}") from None


def _is_compatible(name: str, spec: "DatasetSpec") -> bool:
    try:
        get_model_class(name).check_compatible(spec)
    except IncompatibleError:
        return False
    return True


def compatible_models(spec: "DatasetSpec") -> list[str]:
    """Names of the registered models that accept ``spec``."""
    return [n for n in available_models() if _is_compatible(n, spec)]


def check_compatible(name: str, spec: "DatasetSpec") -> None:
    """Raise :class:`IncompatibleError` (naming the alternatives) on a mismatch."""
    try:
        get_model_class(name).check_compatible(spec)
    except IncompatibleError as exc:
        raise IncompatibleError(
            f"Model '{name}' is incompatible with dataset '{spec.name}' "
            f"({spec.describe()}): {exc}. "
            f"Compatible models: {compatible_models(spec) or 'none'}"
        ) from None


def build_model(name: str, spec: "DatasetSpec", **kwargs) -> SurrogateModel:
    """Instantiate model ``name`` for a dataset, after checking compatibility."""
    check_compatible(name, spec)
    model = get_model_class(name)(spec.n_inputs, spec.n_outputs, **kwargs)
    model.init_kwargs = dict(kwargs)  # stored in checkpoints to rebuild the model
    return model
