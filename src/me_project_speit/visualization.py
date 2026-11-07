"""3D visualisation of stored temperature fields (requires ``EasyFEA``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .datasets import load_dataset
from .simulation import build_mesh

DEFAULT_COORD = Path("data") / "coord.npy"


def plot_fields(
    dataset: str | Path,
    ids: Sequence[int] = (0, 1),
    coord_file: str | Path = DEFAULT_COORD,
) -> None:
    """Plot the temperature field of the simulations ``ids`` of a dataset.

    All plots share the same colour limits so they can be compared. The mesh
    is rebuilt and checked against the node coordinates in ``coord_file``.
    """
    from EasyFEA import Models, PyVista, Simulations

    from . import config

    data = load_dataset(dataset)
    Y = data.fields()  # (n_simulations, n_nodes) whatever the kind
    mesh, _ = build_mesh()

    if data.spec.output_kind != "nodal_temperature" or Y.shape[1] != mesh.Nn:
        raise ValueError(
            f"Dataset '{data.spec.name}' ({data.spec.describe()}) cannot be plotted "
            f"on the built-in mesh ({mesh.Nn} nodes, nodal temperature)."
        )

    coord_file = Path(coord_file)
    if coord_file.exists():
        coord = np.load(coord_file)
        if not np.allclose(mesh.coord, coord):
            raise ValueError(
                f"The rebuilt mesh does not match the node coordinates in {coord_file}. "
                "Node numbering depends on the gmsh version: use the same version as "
                "the one that generated the dataset, or regenerate the dataset."
            )

    model = Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1)
    simu = Simulations.Thermal(mesh, model)

    ids = list(ids)
    clim = (Y[ids].min(), Y[ids].max())

    for i in ids:
        PyVista.Plot(
            simu,
            Y[i],
            plotMesh=True,
            nodeValues=True,
            clim=clim,
            colorbarTitle="Temperature (°C)",
        ).show()
