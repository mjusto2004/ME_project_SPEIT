"""Base class shared by all models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ..dataset import DatasetSpec


class IncompatibleError(ValueError):
    """Raised when a model cannot be used with a given dataset."""


class SurrogateModel(nn.Module):
    """Maps ``n_inputs`` parameters to ``n_outputs`` values.

    Subclasses are registered with :func:`~me_project_speit.models.register_model`
    and must accept ``(n_inputs, n_outputs, **kwargs)``. A model that only works
    for some datasets (fixed mesh, fixed size...) overrides
    :meth:`check_compatible`.
    """

    name: str = ""  # set by @register_model

    @classmethod
    def check_compatible(cls, spec: "DatasetSpec") -> None:
        """Raise :class:`IncompatibleError` if ``spec`` cannot be handled.

        The default accepts any dataset.
        """
