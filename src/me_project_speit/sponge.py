"""Thermal simulation of a random "sponge": the unit cube minus spherical holes.

Needs ``EasyFEA`` (which brings ``gmsh``). Heat enters through one face of the
cube; convection acts on every boundary (outer faces and hole walls).
The functions return plain numpy arrays so that dataset code does not have to
know about EasyFEA objects.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from . import config

# Face index -> (axis, coordinate): 0: x=0, 1: x=1, 2: y=0, 3: y=1, 4: z=0, 5: z=1
FACES = [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]


def build_sponge_mesh(res: int, centers: np.ndarray, r: float):
    """Tetrahedral mesh of the unit cube from which spheres are removed.

    Parameters
    ----------
    res : int
        The element size is ``1 / res``.
    centers : array (K, 3)
        Centres of the holes.
    r : float
        Radius of the holes.

    Returns
    -------
    mesh or None
        EasyFEA mesh, or ``None`` if the holes cut the cube in several pieces
        (or empty it): such a draw must be discarded.
    """
    import gmsh
    from EasyFEA.FEM import Mesher

    a = config.CUBE_SIZE
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sponge.msh")

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("sponge")
            occ = gmsh.model.occ
            box = occ.addBox(0, 0, 0, a, a, a)
            holes = [(3, occ.addSphere(*c, r)) for c in centers]
            volumes, _ = occ.cut([(3, box)], holes)
            occ.synchronize()
            if len(volumes) != 1:
                return None
            size = a / res
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", size / 3)
            gmsh.model.mesh.generate(3)
            gmsh.write(path)
        except Exception:
            return None
        finally:
            gmsh.finalize()

        return Mesher().Mesh_Import_mesh(path, setPhysicalGroupsFromEntities=False)


def run_sponge_simulation(
    centers: np.ndarray, face: int, T0: float, P0: float, h: float, r: float, res: int
):
    """Run one transient thermal simulation on a random sponge.

    Parameters
    ----------
    centers, r, res
        Geometry, see :func:`build_sponge_mesh`.
    face : int
        Face of the cube (0..5, see :data:`FACES`) through which heat enters.
    T0, P0, h
        Initial temperature (K), power (W) and convection coefficient
        (W.m-2.K-1), as in :func:`~me_project_speit.simulation.run_simulation`.

    Returns
    -------
    None if the geometry is invalid, otherwise a tuple
    ``(temperature, coord, connect, heated)``:

    - ``temperature``: (n_nodes,) in deg C after ``config.T_MAX``;
    - ``coord``: (n_nodes, 3) node coordinates;
    - ``connect``: (n_tetrahedra, 4) node indices of each tetrahedron;
    - ``heated``: (n_nodes,) bool, True on the nodes of the heated face.
    """
    from EasyFEA import Models, Simulations

    mesh = build_sponge_mesh(int(res), np.asarray(centers), r)
    if mesh is None:
        return None

    coord = np.asarray(mesh.coord)
    connect = np.asarray(mesh.connect)

    # Boundary = nodes of all surface triangles (outer faces and hole walls).
    boundary = np.unique(np.concatenate([g.nodes for g in mesh.Get_list_groupElem(2)]))

    axis, value = FACES[face]
    heated = np.isclose(coord[:, axis], value)
    is_heated_boundary = heated[boundary]

    # Heat flux on the heated face (W.m-2).
    flux = P0 * config.HEAT_EFFICIENCY / config.CUBE_SIZE**2

    model = Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1)
    simu = Simulations.Thermal(mesh, model)
    simu.rho = config.RHO

    dt = config.T_MAX / config.N_STEPS
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)
    simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * T0)

    t = -dt
    while t < config.T_MAX:
        t += dt
        temperature = simu.thermal
        # One combined load per boundary node: convection, plus the flux on the heated face.
        load = h * (config.T_EXTERIOR - temperature[boundary]) + flux * is_heated_boundary
        simu.Bc_Init()
        simu.add_surfLoad(boundary, [load], ["t"])
        simu.Solve()
        simu.Save_Iter()

    return simu.thermal - 273.15, coord, connect, heated
