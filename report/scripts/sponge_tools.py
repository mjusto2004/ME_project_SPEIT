"""Rebuild one sponge of a graph dataset (same random draw) to get its tetrahedra for rendering."""

import numpy as np
import pyvista as pv

from me_project_speit.datasets.graph import MAX_TRIES
from me_project_speit.simulations import SpongeSimulation, config


def rebuild(seed: int, n: int, i: int, ranges: dict):
    """Simulation ``i`` of a dataset generated with ``generate -n n --seed seed`` and these ranges."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(n)[i])
    for _ in range(MAX_TRIES):
        params = {k: rng.uniform(*ranges[k]) for k in ranges}
        for k in params:
            if config.SPONGE_PARAMS[k].integer:
                params[k] = int(round(params[k]))
        holes = rng.uniform(size=(int(rng.integers(1, config.MAX_HOLES + 1)), 3))
        face = int(rng.integers(0, len(config.FACES)))
        result = SpongeSimulation.run(holes, face, **params)
        if result is not None:
            T, coord, connect, heated = result
            return dict(params=params, holes=holes, face=face, T=T, coord=coord, connect=connect, heated=heated)
    return None


def grid(coord, connect, **fields):
    cells = np.hstack([np.full((len(connect), 1), 4), connect]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(connect), pv.CellType.TETRA), np.asarray(coord, float))
    for k, v in fields.items():
        g.point_data[k] = v
    return g
