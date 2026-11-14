"""Multi-layer perceptron mapping the input parameters to the output field."""

from collections.abc import Sequence

import torch
from torch import nn

from ..datasets import FieldDataset, PointwiseDataset
from .base import IncompatibleError, SurrogateModel
from .registry import register_model


@register_model("mlp")
class MLP(SurrogateModel):
    """Fully connected ReLU network; adapts to any dataset shape.

    ``n_inputs -> hidden[0] -> ... -> hidden[-1] -> n_outputs``

    Predicts a whole field from the parameters, so it needs a
    :class:`FieldDataset` (one row per simulation).
    """

    supported_datasets = (FieldDataset,)

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden: Sequence[int] = (128, 256, 256),
    ):
        super().__init__()
        sizes = [n_inputs, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@register_model("coord_mlp")
class CoordMLP(MLP):
    """MLP predicting a value at a point: ``(params..., x, y, z) -> value``.

    Works on :class:`PointwiseDataset`, whose last three inputs are the node
    coordinates. Unlike ``mlp`` it is not tied to a mesh size, since it is
    evaluated one point at a time.
    """

    supported_datasets = (PointwiseDataset,)

    @classmethod
    def check_compatible(cls, spec):
        super().check_compatible(spec)
        if list(spec.input_names[-3:]) != ["x", "y", "z"]:
            raise IncompatibleError("the last three inputs must be the coordinates x, y, z")
        if spec.n_outputs != 1:
            raise IncompatibleError(f"predicts a single value, dataset has {spec.n_outputs} outputs")
