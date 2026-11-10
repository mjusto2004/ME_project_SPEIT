"""The random "sponge": the unit cube minus spherical holes (dataset kind ``graph``).

The mesh changes with every sample (random holes), so the node count varies.
Needs ``EasyFEA`` (which brings ``gmsh``). Heat enters through one face of the
cube; convection acts on every boundary (outer faces and hole walls). The
methods return plain numpy arrays so that dataset code does not have to know
about EasyFEA objects.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from . import config
from .base import Simulation


class SpongeSimulation(Simulation):
    """Cube with random spherical holes and a mesh that changes at every run."""

    param_defaults = config.SPONGE_PARAM_DEFAULTS
    param_ranges = config.SPONGE_PARAM_RANGES

    @staticmethod
    def build_mesh(res: int, centers: np.ndarray, r: float):
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

    @classmethod
    def run(cls, centers: np.ndarray, face: int, **params: float):
        """Run one transient thermal simulation on a random sponge.

        Parameters
        ----------
        centers : array (K, 3)
            Centres of the holes, see :meth:`build_mesh`.
        face : int
            Face of the cube (0..5, see ``config.FACES``) through which heat enters.
        **params : float
            Any of the names in ``param_defaults``; missing ones take their
            default value.

            - ``T0``: initial uniform temperature (K).
            - ``P0``: power (W), spread as a flux
              ``P0 * HEAT_EFFICIENCY / CUBE_SIZE**2`` over the whole heated face
              (unlike :class:`~me_project_speit.simulations.CubeSimulation`,
              where it is injected through the source cylinder).
            - ``h``: convection coefficient (W.m-2.K-1).
            - ``r``, ``res``: hole radius and mesh resolution, see :meth:`build_mesh`.

        Returns
        -------
        None if the geometry is invalid or the simulation diverged, otherwise a
        tuple ``(temperature, coord, connect, heated)``:

        - ``temperature``: (n_nodes,) in deg C after ``config.T_MAX``;
        - ``coord``: (n_nodes, 3) node coordinates;
        - ``connect``: (n_tetrahedra, 4) node indices of each tetrahedron;
        - ``heated``: (n_nodes,) bool, True on the nodes of the heated face.
        """
        p = cls.resolve_params(params)
        t0, power, h, r, res = p["T0"], p["P0"], p["h"], p["r"], p["res"]

        mesh = cls.build_mesh(int(res), np.asarray(centers), r)
        if mesh is None:
            return None

        coord = np.asarray(mesh.coord)
        connect = np.asarray(mesh.connect)

        # Boundary = nodes of all surface triangles (outer faces and hole walls).
        boundary = np.unique(np.concatenate([g.nodes for g in mesh.Get_list_groupElem(2)]))

        axis, value = config.FACES[face]
        heated = np.isclose(coord[:, axis], value)
        is_heated_boundary = heated[boundary]

        # Heat flux on the heated face (W.m-2).
        flux = power * config.HEAT_EFFICIENCY / config.CUBE_SIZE**2

        # Convection is applied explicitly (EasyFEA has no Robin term), which is only
        # stable if h*dt < rho*c*(element size). Use more, shorter steps when the mesh
        # is fine or h is large. A low percentile of the edge lengths is used rather
        # than the minimum (gmsh makes a few micro-edges where holes meet), and the
        # number of steps is capped; a draw that still diverges is discarded below.
        edges = np.vstack([connect[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
        h_min = np.percentile(np.linalg.norm(coord[edges[:, 0]] - coord[edges[:, 1]], axis=1), 5)
        dt_stable = config.STABILITY_SAFETY * config.RHO * config.CP * h_min / h
        n_steps = min(config.MAX_STEPS, max(config.N_STEPS, int(np.ceil(config.T_MAX / dt_stable))))
        dt = config.T_MAX / n_steps
        simu = cls._init_thermal(mesh, t0, dt)

        for _ in range(n_steps):
            # One combined load per boundary node: convection, plus the flux on the heated face.
            load = cls._convection(simu, boundary, h) + flux * is_heated_boundary
            cls._step(simu, (boundary, load))

        temperature = cls._celsius(simu)
        t_min, t_max = config.T_VALID
        if not np.isfinite(temperature).all() or not (t_min < temperature.min() and temperature.max() < t_max):
            return None  # the time integration diverged: discard this draw

        return temperature, coord, connect, heated
