"""A graph neural operator: message passing over a *physical* radius, not mesh links.

Made to be trained on coarse meshes and used on finer ones. The plain ``gnn``
passes messages along the mesh edges, so everything it learns is counted in
links ("8 links away") and changes meaning when the mesh is refined. Here:

- the neighbours of a node are the nodes within ``radius`` (in metres) of it,
  whatever the mesh, so ``layers`` rounds always reach ``layers * radius``;
- a message depends on the neighbour's state and on the *relative position*
  ``(dx, dy, dz) / radius`` and distance, and the messages are averaged: the
  average estimates an integral over the ball, which does not depend on how
  finely the ball is meshed (the idea of the Graph Kernel Network, Li et al.
  2020, "Neural Operator: Graph Kernel Network for Partial Differential
  Equations");
- on a fine mesh a ball holds hundreds of nodes, so at most ``max_neighbours``
  of them are drawn at random (a new draw at every training step; a fixed draw
  in evaluation, so predictions are reproducible);
- a global state (mean of all the nodes of the simulation) is given back to
  every node at each round, so every node knows about the whole part;
- inputs describing the mesh rather than the physics (``res`` by default) are
  ignored.

The mesh edges given by the dataset are only used to tell the simulations of a
batch apart (see :func:`simulation_ids`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..datasets.graph import GraphDataset
from .base import IncompatibleError, SurrogateModel
from .registry import register_model

COORDS = ("x", "y", "z")
CHUNK = 2048  # rows of the distance matrix computed at once (bounds memory on large meshes)


def _mlp(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.SiLU(), nn.Linear(n_out, n_out), nn.LayerNorm(n_out))


def simulation_ids(n_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Simulation number (0, 1, ...) of every node of a batch.

    In a batch the nodes of each simulation are contiguous and the edges never
    join two simulations. Since each mesh is connected, a new simulation starts
    exactly at the positions that no edge spans.
    """
    device = edge_index.device
    src, dst = edge_index
    lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
    cover = torch.zeros(n_nodes + 1, dtype=torch.long, device=device)
    cover.index_add_(0, lo + 1, torch.ones_like(lo))
    cover.index_add_(0, hi + 1, -torch.ones_like(hi))
    spanned = cover.cumsum(0)[:n_nodes] > 0  # spanned[b]: some edge joins a node < b to a node >= b
    return torch.cumsum((~spanned).long(), 0) - 1


def radius_neighbours(
    pos: torch.Tensor, groups: torch.Tensor, radius: float, k: int, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """At most ``k`` neighbours drawn uniformly among the nodes within ``radius``.

    Only nodes of the same simulation are neighbours; a node is not its own
    neighbour. Returns ``(src, dst)``: ``src[e]`` is a neighbour of ``dst[e]``.
    """
    srcs, dsts = [], []
    starts = torch.nonzero(torch.diff(groups, prepend=groups[:1] - 1)).flatten().tolist() + [len(pos)]
    for a, b in zip(starts[:-1], starts[1:]):
        p = pos[a:b]
        kk = min(k, b - a)
        for c in range(0, b - a, CHUNK):
            rows = p[c : c + CHUNK]
            near = torch.cdist(rows, p) <= radius
            near[torch.arange(len(rows), device=pos.device), torch.arange(c, c + len(rows), device=pos.device)] = False
            # random score for the nodes in the ball, -1 outside: the top kk are a uniform draw
            score = torch.rand(near.shape, device=pos.device, generator=generator).masked_fill_(~near, -1.0)
            value, idx = score.topk(kk, dim=1)
            keep = value >= 0
            dst = torch.arange(c, c + len(rows), device=pos.device)[:, None].expand_as(idx)
            srcs.append(idx[keep] + a)
            dsts.append(dst[keep] + a)
    return torch.cat(srcs), torch.cat(dsts)


@register_model("gno")
class GNO(SurrogateModel):
    """Graph neural operator for ``graph`` datasets, independent of the mesh resolution.

    Parameters
    ----------
    hidden : int or sequence of int
        Width of the node states (a sequence uses its first value).
    layers : int
        Rounds of message passing. Together they reach ``layers * radius``.
    radius : float
        Neighbourhood radius, in the units of ``x, y, z`` (m; the cube is 1 m).
    max_neighbours : int
        Neighbours drawn per node and per step during training.
    eval_neighbours : int or None
        Neighbours drawn in evaluation (``None``: ``max_neighbours``). More gives a
        less noisy average at a higher cost.
    exclude_inputs : sequence of str
        Inputs the model ignores (default ``res``: the mesh resolution, which
        must not matter). Names absent from the dataset are skipped.
    """

    supported_datasets = (GraphDataset,)

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        input_names: Sequence[str] | None = None,
        hidden: int | Sequence[int] = 128,
        layers: int = 6,
        radius: float = 0.15,
        max_neighbours: int = 24,
        eval_neighbours: int | None = None,
        exclude_inputs: Sequence[str] = ("res",),
    ):
        super().__init__()
        names = list(input_names) if input_names is not None else [f"in{i}" for i in range(n_inputs)]
        if len(names) != n_inputs:
            raise ValueError(f"{len(names)} input names for {n_inputs} inputs")
        if any(c not in names for c in COORDS):
            raise ValueError("The gno model needs the inputs x, y, z.")
        if any(c in exclude_inputs for c in COORDS):
            raise ValueError("x, y, z cannot be excluded: the neighbourhoods are built from them.")
        if radius <= 0 or max_neighbours < 1:
            raise ValueError("radius and max_neighbours must be positive.")
        width = hidden if isinstance(hidden, int) else hidden[0]
        self.radius, self.k = float(radius), int(max_neighbours)
        self.k_eval = int(eval_neighbours or max_neighbours)
        self.ignored_inputs = [n for n in names if n in exclude_inputs]
        self.register_buffer("keep", torch.tensor([i for i, n in enumerate(names) if n not in exclude_inputs]))
        self.register_buffer("coord_idx", torch.tensor([names.index(c) for c in COORDS]))
        # Normalisation statistics (set_scaling), to get the coordinates back in metres.
        self.register_buffer("x_mean", torch.zeros(n_inputs))
        self.register_buffer("x_std", torch.ones(n_inputs))

        self.encoder = _mlp(len(self.keep), width)
        self.message = nn.ModuleList(_mlp(2 * width + 4, width) for _ in range(layers))
        self.glob = nn.ModuleList(_mlp(width, width) for _ in range(layers))
        self.update = nn.ModuleList(_mlp(3 * width, width) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n_outputs))

    @classmethod
    def from_spec(cls, spec, **kwargs):
        return cls(spec.n_inputs, spec.n_outputs, input_names=spec.input_names, **kwargs)

    @classmethod
    def check_compatible(cls, spec) -> None:
        super().check_compatible(spec)
        missing = [c for c in COORDS if c not in spec.input_names]
        if missing:
            raise IncompatibleError(f"needs the node coordinates as inputs, missing {missing}")

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--hidden", type=int, help="width of the node states")
        parser.add_argument("--layers", type=int, help="number of message-passing rounds")
        parser.add_argument("--radius", type=float, help="neighbourhood radius, in the units of x, y, z")
        parser.add_argument("--max-neighbours", type=int, help="neighbours drawn per node in training")
        parser.add_argument("--eval-neighbours", type=int,
                            help="neighbours drawn per node in evaluation (default: --max-neighbours)")
        parser.add_argument("--exclude-inputs", nargs="*", metavar="NAME",
                            help="inputs ignored by the model ('--exclude-inputs' alone: none)")

    @classmethod
    def add_fit_arguments(cls, parser):
        parser.add_argument("--weight-decay", type=float, help="L2 penalty of the optimizer")

    def set_scaling(self, x_mean, x_std, y_mean, y_std) -> None:
        self.x_mean.copy_(torch.as_tensor(x_mean))
        self.x_std.copy_(torch.as_tensor(x_std))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``x``: (n_nodes, n_inputs), normalised; ``edge_index``: the mesh edges of the batch."""
        with torch.no_grad():
            groups = simulation_ids(len(x), edge_index)
            pos = x[:, self.coord_idx] * self.x_std[self.coord_idx] + self.x_mean[self.coord_idx]
            generator = None
            if not self.training:  # same draw every time: reproducible predictions
                generator = torch.Generator(device=x.device).manual_seed(0)
            src, dst = radius_neighbours(
                pos, groups, self.radius, self.k if self.training else self.k_eval, generator
            )
            rel = (pos[src] - pos[dst]) / self.radius
            edge_attr = torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)
            degree = torch.zeros(len(x), device=x.device).index_add_(
                0, dst, torch.ones(len(dst), device=x.device)
            ).clamp(min=1)[:, None]
            n_groups = int(groups[-1]) + 1
            group_size = torch.zeros(n_groups, device=x.device).index_add_(
                0, groups, torch.ones(len(x), device=x.device)
            )[:, None]

        h = self.encoder(x[:, self.keep])
        for message, glob, update in zip(self.message, self.glob, self.update):
            m = message(torch.cat([h[src], h[dst], edge_attr], dim=1))
            local = torch.zeros_like(h).index_add_(0, dst, m) / degree
            pooled = torch.zeros(n_groups, h.shape[1], device=x.device, dtype=h.dtype).index_add_(0, groups, h)
            g = glob(pooled / group_size)[groups]
            h = h + update(torch.cat([h, local, g], dim=1))
        return self.head(h)

    def fit(self, train_loader, val_loader, **kwargs):
        return self._fit_loop(train_loader, val_loader, **kwargs)
