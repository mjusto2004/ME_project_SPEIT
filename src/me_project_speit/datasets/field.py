"""``field`` datasets: one row per simulation, parameters -> whole field."""

from __future__ import annotations

import numpy as np

from ..simulations import config
from .base import Dataset, DatasetSpec, register_dataset


@register_dataset("field")
class FieldDataset(Dataset):
    """Row: ``[params..., T_node_0, ..., T_node_{m-1}]``.

    The model predicts the whole temperature field on the mesh from the
    simulation parameters.
    """

    @classmethod
    def build_spec(cls, input_names=("T0", "Phi0", "h"), name: str = "cube") -> DatasetSpec:
        names = cls.check_params(input_names)
        return DatasetSpec(
            name=name,
            input_names=names,
            n_outputs=config.N_NODES,
            mesh_id=config.MESH_ID,
            input_ranges=cls.ranges(names),
            kind=cls.kind,
            rows_per_sample=1,
        )

    @classmethod
    def simulation_rows(cls, spec: DatasetSpec, params: dict[str, float]) -> np.ndarray:
        from ..simulations import CubeSimulation

        field_ = CubeSimulation.run(**params)
        values = [params[k] for k in spec.input_names]
        return np.concatenate((values, field_))[None, :]

    def fields(self) -> np.ndarray:
        return self.Y


def legacy_spec() -> DatasetSpec:
    """Spec of CSVs created before sidecars existed (563 columns)."""
    return FieldDataset.build_spec(("T0", "Phi0", "h"), name="legacy")
