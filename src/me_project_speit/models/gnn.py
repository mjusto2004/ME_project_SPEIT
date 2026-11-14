"""A simple message-passing graph neural network, in pure PyTorch."""

from collections.abc import Sequence

import torch
from torch import nn

from ..datasets.graph import GraphDataset
from .base import SurrogateModel
from .registry import register_model


def _mlp(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Linear(n_out, n_out))


@register_model("gnn")
class GNN(SurrogateModel):
    """Predicts one value per node from the node inputs and the mesh connections.

    1. an MLP encodes the inputs of each node into ``width`` numbers;
    2. ``layers`` times, every node receives the *average* of a message computed
       from each neighbour, and updates its own state with it (residual);
    3. a linear layer gives the output of each node.

    After ``layers`` rounds a node has information from nodes up to ``layers``
    links away.

    Parameters
    ----------
    hidden : int or sequence of int
        Width of the node states (a sequence uses its first value, so the
        ``--hidden`` option of the command line works).
    layers : int
        Number of message-passing rounds.
    """

    supported_datasets = (GraphDataset,)

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden: int | Sequence[int] = 64,
        layers: int = 4,
    ):
        super().__init__()
        width = hidden if isinstance(hidden, int) else hidden[0]
        self.encoder = _mlp(n_inputs, width)
        self.message = nn.ModuleList(_mlp(2 * width, width) for _ in range(layers))
        self.update = nn.ModuleList(_mlp(2 * width, width) for _ in range(layers))
        self.head = nn.Linear(width, n_outputs)

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--hidden", type=int, help="width of the node states")
        parser.add_argument("--layers", type=int, help="number of message-passing rounds")

    @classmethod
    def add_fit_arguments(cls, parser):
        parser.add_argument("--weight-decay", type=float, help="L2 penalty of the optimizer")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``x``: (n_nodes, n_inputs); ``edge_index``: (2, n_edges) with rows (src, dst)."""
        src, dst = edge_index
        # Number of messages received by each node (at least 1 to divide safely).
        degree = torch.zeros(len(x), device=x.device).index_add_(
            0, dst, torch.ones(len(dst), device=x.device)
        ).clamp(min=1)

        h = self.encoder(x)
        for message, update in zip(self.message, self.update):
            m = message(torch.cat([h[src], h[dst]], dim=1))  # one message per edge
            mean = torch.zeros_like(h).index_add_(0, dst, m) / degree[:, None]
            h = h + update(torch.cat([h, mean], dim=1))
        return self.head(h)

    def fit(self, train_loader, val_loader, **kwargs):
        return self._fit_loop(train_loader, val_loader, **kwargs)
