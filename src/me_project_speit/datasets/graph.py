"""``graph`` datasets: random "sponge" cubes, with the node connections of each mesh.

Each simulation is a unit cube from which random spherical holes were removed,
so every simulation has its own mesh (its own number of nodes and edges). Heat
enters through one random face and convection acts on every boundary.

Three files share the same stem:

- ``name.csv``: one row per node of every simulation,
  ``sim, T0, P0, h, r, res, x, y, z, heated, T`` (``heated`` is 1 on the nodes
  of the heated face). The rows of one simulation are contiguous.
- ``name_edges.npz``: the connections, arrays ``sim``, ``src``, ``dst``. ``src``
  and ``dst`` are node indices *local to the simulation* (0 = its first row).
  Each connection appears in both directions.
- ``name.json``: the :class:`DatasetSpec` (``kind = "graph"``).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config
from ..parallel import parallel_map
from .base import Dataset, DatasetSpec, Split, register_dataset, save_spec, split_indices

MAX_TRIES = 20  # redraws allowed per simulation when a random sponge is invalid
EXTRA_INPUTS = ["x", "y", "z", "heated"]  # per-node inputs added to the parameters


def edges_from_connectivity(connect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node-to-node connections of a mesh.

    Two nodes are connected if they belong to the same element. Every
    connection is returned in both directions, without duplicates.

    Parameters
    ----------
    connect : array (n_elements, nodes_per_element)
        Node indices of each element (4 for tetrahedra).

    Returns
    -------
    src, dst : arrays (n_edges,)
        ``src[k] -> dst[k]`` is the k-th directed connection.
    """
    pairs = [connect[:, [i, j]] for i, j in combinations(range(connect.shape[1]), 2)]
    pairs = np.unique(np.sort(np.vstack(pairs), axis=1), axis=0)  # undirected, unique
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return src, dst


def edges_path(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + "_edges.npz")


@register_dataset("graph")
class GraphDataset(Dataset):
    """Variable-size graphs. ``X``/``Y`` hold the node rows of all simulations.

    Attributes
    ----------
    sim : array (rows,)
        Simulation id of each node row (sorted, 0..n_simulations-1).
    edge_sim, edge_src, edge_dst : arrays (total_edges,)
        Connections of all simulations (local node indices, see module doc).
    """

    param_ranges = config.SPONGE_PARAM_RANGES
    default_params = ("T0", "P0", "h", "r", "res")

    def __init__(self, X, Y, spec, sim, edge_sim, edge_src, edge_dst):
        super().__init__(X, Y, spec)
        self.sim = sim
        self.edge_sim, self.edge_src, self.edge_dst = edge_sim, edge_src, edge_dst
        # Where the rows / edges of each simulation start and stop.
        self.node_stop = np.cumsum(np.bincount(sim))
        self.node_start = self.node_stop - np.bincount(sim)
        self.edge_stop = np.cumsum(np.bincount(edge_sim, minlength=len(self.node_stop)))
        self.edge_start = self.edge_stop - np.bincount(edge_sim, minlength=len(self.node_stop))

    def __len__(self) -> int:
        return len(self.node_stop)

    # ---- hooks of Dataset ---------------------------------------------------
    @classmethod
    def build_spec(cls, input_names=default_params, name: str = "sponge") -> DatasetSpec:
        names = cls.check_params(input_names)
        return DatasetSpec(
            name=name,
            input_names=names + EXTRA_INPUTS,
            n_outputs=1,
            mesh_id="variable",
            input_ranges=cls.ranges(names),
            kind=cls.kind,
            rows_per_sample=None,  # varies from one simulation to the next
        )

    @classmethod
    def simulation_rows(cls, spec, params):
        raise NotImplementedError(
            "A graph simulation also returns its edges and needs random holes: "
            "use GraphDataset.generate()."
        )

    def fields(self):
        raise NotImplementedError("Graph datasets have one mesh per simulation: no common field.")

    # ---- generation ---------------------------------------------------------
    @classmethod
    def generate(
        cls, n, input_names=None, name="sponge", seed=None, verbose=True, workers=1
    ) -> "GraphDataset":
        """Run ``n`` simulations on random sponges, in ``workers`` processes.

        Each simulation has its own random generator (made from ``seed``), so
        the dataset is the same whatever the number of workers. See
        :meth:`_simulate` for what is drawn.
        """
        spec = cls.build_spec(input_names or cls.default_params, name=name)
        seeds = np.random.SeedSequence(seed).spawn(n)  # one independent seed per simulation

        node_blocks, edge_blocks = [], []
        results = parallel_map(cls._simulate, [spec] * n, seeds, workers=workers)
        for i, (nodes, edges) in enumerate(results):
            node_blocks.append(np.column_stack([np.full(len(nodes), i), nodes]))
            edge_blocks.append(np.column_stack([np.full(len(edges), i), edges]))
            if verbose:
                print(f"Simulation {i + 1}/{n} done ({len(nodes)} nodes, {len(edges)} directed edges)")

        return cls._from_blocks(spec, np.vstack(node_blocks), np.vstack(edge_blocks))

    @classmethod
    def _simulate(cls, spec: DatasetSpec, seed) -> tuple[np.ndarray, np.ndarray]:
        """One simulation on a random sponge (run in a worker process).

        Draws the parameters of ``spec`` in their ranges (the others keep their
        default), then 1..``MAX_HOLES`` hole centres and the heated face. A draw
        giving an invalid mesh (the holes cut the cube in several pieces) is
        redrawn.

        Returns the node rows ``[inputs..., output]`` and the edges ``[src, dst]``
        (without the simulation number, added by :meth:`generate`).
        """
        from ..sponge import run_sponge_simulation

        rng = np.random.default_rng(seed)
        for _ in range(MAX_TRIES):
            params = {k: rng.uniform(*spec.input_ranges[k]) for k in spec.input_ranges}
            for k in config.INTEGER_PARAMS & set(params):
                params[k] = int(round(params[k]))
            holes = rng.uniform(size=(int(rng.integers(1, config.MAX_HOLES + 1)), 3))
            face = int(rng.integers(0, 6))
            result = run_sponge_simulation(
                holes, face, **{**config.SPONGE_PARAM_DEFAULTS, **params}
            )
            if result is not None:
                break
        else:
            raise RuntimeError(f"No valid sponge after {MAX_TRIES} draws.")

        temperature, coord, connect, heated = result
        src, dst = edges_from_connectivity(connect)
        values = np.array([params[k] for k in spec.input_ranges])
        nodes = np.hstack(
            [np.tile(values, (len(coord), 1)), coord, heated[:, None], temperature[:, None]]
        )
        return nodes, np.column_stack([src, dst])

    @classmethod
    def _from_blocks(cls, spec, nodes: np.ndarray, edges: np.ndarray) -> "GraphDataset":
        """Build a dataset from the node table ``[sim, inputs..., output]`` and edges ``[sim, src, dst]``."""
        return cls(
            X=nodes[:, 1 : 1 + spec.n_inputs].astype(np.float32),
            Y=nodes[:, 1 + spec.n_inputs :].astype(np.float32),
            spec=spec,
            sim=nodes[:, 0].astype(np.int64),
            edge_sim=edges[:, 0].astype(np.int64),
            edge_src=edges[:, 1].astype(np.int64),
            edge_dst=edges[:, 2].astype(np.int64),
        )

    # ---- files --------------------------------------------------------------
    @classmethod
    def from_csv(cls, path, spec) -> "GraphDataset":
        path = Path(path)
        nodes = np.loadtxt(path, delimiter=",", ndmin=2)
        if nodes.shape[1] != 1 + spec.n_columns:
            raise ValueError(
                f"{path.name}: expected {1 + spec.n_columns} columns (sim + {spec.n_inputs} "
                f"inputs + {spec.n_outputs} outputs), got {nodes.shape[1]}"
            )
        if not np.isfinite(nodes).all():
            raise ValueError("Dataset contains NaN or infinite values.")
        sim = nodes[:, 0].astype(np.int64)
        if (np.diff(sim) < 0).any() or sim[0] != 0 or (np.diff(sim) > 1).any():
            raise ValueError(f"{path.name}: the 'sim' column must be 0, 1, 2... with contiguous rows.")
        if sim[-1] + 1 < 3:
            raise ValueError("At least 3 simulations are required.")

        edge_file = edges_path(path)
        if not edge_file.exists():
            raise ValueError(f"Missing edges file '{edge_file.name}' next to {path.name}.")
        with np.load(edge_file) as f:
            edges = np.column_stack([f["sim"], f["src"], f["dst"]])
        n_nodes = np.bincount(sim)
        if edges[:, 0].max() >= len(n_nodes) or (edges[:, 1:].max(axis=1) >= n_nodes[edges[:, 0]]).any():
            raise ValueError(f"{edge_file.name}: an edge points to a node outside its simulation.")
        return cls._from_blocks(spec, nodes, edges)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = np.column_stack([self.sim, self.X, self.Y])
        np.savetxt(path, nodes, delimiter=",", fmt=["%d"] + ["%.9g"] * (nodes.shape[1] - 1))
        np.savez_compressed(
            edges_path(path), sim=self.edge_sim, src=self.edge_src, dst=self.edge_dst
        )
        save_spec(self.spec, path)
        return path

    # ---- training support ---------------------------------------------------
    def split(self, seed: int, test_frac: float = 0.15, val_frac: float = 0.15) -> Split:
        """Split the simulations, then return the node rows of each subset."""
        by_sim = split_indices(len(self), seed, test_frac, val_frac)  # simulation ids
        rows = lambda sims: np.flatnonzero(np.isin(self.sim, sims))
        return Split(train=rows(by_sim.train), val=rows(by_sim.val), test=rows(by_sim.test))

    def count_simulations(self, row_indices) -> int:
        return len(np.unique(self.sim[row_indices]))

    def make_loader(self, X, Y, row_indices, batch_size, shuffle, pin_memory=False):
        """Batches of whole graphs ``(x, edge_index, y)``.

        ``batch_size`` is a number of *simulations*. Their nodes are
        concatenated and their edge indices shifted by the number of nodes
        before them, so that one batch is a single big graph made of
        disconnected pieces.
        """
        simulations = np.unique(self.sim[row_indices]).tolist()

        def collate(batch):
            xs, ys, srcs, dsts, offset = [], [], [], [], 0
            for s in batch:
                a, b = self.node_start[s], self.node_stop[s]
                c, d = self.edge_start[s], self.edge_stop[s]
                xs.append(X[a:b])
                ys.append(Y[a:b])
                srcs.append(self.edge_src[c:d] + offset)
                dsts.append(self.edge_dst[c:d] + offset)
                offset += b - a
            edge_index = np.stack([np.concatenate(srcs), np.concatenate(dsts)])
            return (
                torch.from_numpy(np.concatenate(xs)),
                torch.from_numpy(edge_index),
                torch.from_numpy(np.concatenate(ys)),
            )

        return DataLoader(simulations, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
