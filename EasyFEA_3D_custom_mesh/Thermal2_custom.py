# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Thermal2_custom
===============

``examples/Thermal/Thermal2.py`` with the mesh block replaced by a custom
3D mesh.  Only the mesh construction and the boundary-node selection are
changed; the thermal model and the transient solve are identical.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

from pathlib import Path

import custom_mesh as cm
from thermal2_custom_mesh import solve_transient
from thermal_custom_mesh import resolve_bc_nodes

if __name__ == "__main__":
    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    meshFile = Path(__file__).parent / "data" / "thermal2_TETRA4" / "mesh.json"
    Tmax = 0.5
    N = 20

    # ----------------------------------------------
    # Mesh (was: Domain + Circle + Mesh_Extrude)
    # ----------------------------------------------
    mesh = cm.load_mesh(meshFile)
    if mesh.dim != 3:
        raise ValueError(f"Expected a 3D mesh, got {mesh.dim}D.")

    nodesX0 = resolve_bc_nodes(mesh, "left")
    nodesXa = resolve_bc_nodes(mesh, "right")

    # ----------------------------------------------
    # Simulation (unchanged)
    # ----------------------------------------------
    simu, temperature = solve_transient(
        mesh,
        [(nodesX0, 0.0), (nodesXa, 40.0)],
        tmax=Tmax,
        steps=N,
        alpha=0.5,
        k=1.0,
        c=1.0,
        rho=1.0,
    )

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(f"Mesh: {mesh.elemType}, Nn={mesh.Nn}, Ne={mesh.Ne}, volume={mesh.volume:.8g}")
    print(
        f"Transient: {N} steps, dt={Tmax / N:g}, "
        f"Tmin={temperature.min():.6g}, Tmax={temperature.max():.6g}, "
        f"mean={temperature.mean():.6g}"
    )
