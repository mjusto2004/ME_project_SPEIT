"""Base class shared by all models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ..datasets import Dataset, DatasetSpec


class IncompatibleError(ValueError):
    """Raised when a model cannot be used with a given dataset."""


class SurrogateModel(nn.Module):
    """Maps ``n_inputs`` parameters to ``n_outputs`` values.

    Subclasses are registered with :func:`~me_project_speit.models.register_model`
    and must accept ``(n_inputs, n_outputs, **kwargs)``.

    Compatibility with a dataset is checked by :meth:`check_compatible`. Its
    default implementation verifies that the dataset class is one of
    :attr:`supported_datasets` (or a subclass of one); models with extra
    requirements (fixed mesh, coordinate inputs, ...) extend it and call
    ``super().check_compatible``.
    """

    name: str = ""  # set by @register_model

    #: Dataset classes the model can learn from (subclasses are accepted too);
    #: ``None`` accepts any dataset.
    supported_datasets: "tuple[type[Dataset], ...] | None" = None

    @classmethod
    def check_compatible(cls, spec: "DatasetSpec") -> None:
        """Raise :class:`IncompatibleError` if ``spec`` cannot be handled."""
        if cls.supported_datasets is None:
            return
        from ..datasets import get_dataset_class

        dataset_cls = get_dataset_class(spec.kind)
        if not issubclass(dataset_cls, cls.supported_datasets):
            wanted = [c.__name__ for c in cls.supported_datasets]
            raise IncompatibleError(
                f"needs a dataset of type {wanted}, got {dataset_cls.__name__} (kind '{spec.kind}')"
            )
