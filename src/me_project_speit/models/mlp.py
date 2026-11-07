"""Multi-layer perceptron mapping the input parameters to the output field."""

from collections.abc import Sequence

import torch
from torch import nn

from .base import SurrogateModel
from .registry import register_model


@register_model("mlp")
class MLP(SurrogateModel):
    """Fully connected ReLU network; adapts to any dataset shape.

    ``n_inputs -> hidden[0] -> ... -> hidden[-1] -> n_outputs``
    """

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
