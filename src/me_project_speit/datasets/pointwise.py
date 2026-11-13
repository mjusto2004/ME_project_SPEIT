"""``pointwise`` datasets: one row per (simulation, node), coordinates as inputs."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..simulations import config
from .base import Dataset, DatasetSpec, register_dataset

COORD_NAMES = ["x", "y", "z"]


@lru_cache(maxsize=1)
def _node_coordinates() -> np.ndarray:
    """Node coordinates of the reference mesh, shape ``(n_nodes, 3)``."""
    from ..simulations import CubeSimulation

    return np.asarray(CubeSimulation.build_mesh()[0].coord)


@register_dataset("pointwise")
class PointwiseDataset(Dataset):
    """Row: ``[params..., x, y, z, T]`` (one per node of each simulation).

    The ``n_nodes`` rows of one simulation are contiguous. The model predicts
    the temperature at the given coordinates.
    """

    @classmethod
    def build_spec(cls, input_names=("T0", "Phi0", "h"), name: str = "cube") -> DatasetSpec:
        names = cls.check_params(input_names)
        return DatasetSpec(
            name=name,
            input_names=names + COORD_NAMES,
            n_outputs=1,
            mesh_id=config.MESH_ID,
            input_ranges=cls.ranges(names),
            kind=cls.kind,
            rows_per_sample=config.N_NODES,
        )

    @classmethod
    def simulation_rows(cls, spec: DatasetSpec, params: dict[str, float]) -> np.ndarray:
        from ..simulations import CubeSimulation

        field_ = CubeSimulation.run(**params)
        coords = _node_coordinates()
        values = np.array([params[k] for k in spec.input_ranges])
        return np.hstack([np.tile(values, (len(coords), 1)), coords, field_[:, None]])

    def fields(self) -> np.ndarray:
        return self.Y.reshape(len(self), self.spec.rows_per_sample)
