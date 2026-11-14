"""The fixed cube (dataset kinds ``field`` and ``pointwise``). Needs ``EasyFEA``.

The mesh is always the same (``config.N_NODES`` nodes), so only
the ranges of ``config.CUBE_PARAMS`` vary between samples. A unit cube is heated by a
cylindrical source along its z axis and cooled by convection on its six faces.
"""

from __future__ import annotations

import numpy as np

from . import config
from .base import Simulation


class CubeSimulation(Simulation):
    """Cube with a cylindrical heat source and a fixed mesh."""

    params = config.CUBE_PARAMS

    @staticmethod
    def build_mesh(
        res: int = config.MESH_RESOLUTION_DOMAIN,
        cylinder_res: int = config.MESH_RESOLUTION_SOURCE,
    ):
        """Build the 3D tetrahedral mesh of the cube and the heat-source circle.

        Returns
        -------
        mesh, circle
            The EasyFEA mesh and the circle (cross-section of the source).
        """
        from EasyFEA import ElemType
        from EasyFEA.Geoms import Circle, Domain, Point

        a = config.CUBE_SIZE
        domain = Domain(Point(0, 0), Point(a, a), a / res)
        circle = Circle(
            Point(a / 2, a / 2),
            diam=config.SOURCE_DIAMETER,
            meshSize=a / cylinder_res,
        )
        mesh = domain.Mesh_Extrude(
            [circle], [0, 0, a], [config.EXTRUSION_LAYERS], ElemType.TETRA4
        )
        return mesh, circle

    @classmethod
    def run(cls, verbose: bool = False, **params: float) -> np.ndarray:
        """Run one transient thermal simulation.

        Parameters
        ----------
        verbose : bool
            Print the progress of the time loop.
        **params : float
            Any of the names in ``params``; missing ones take their
            default value.

            - ``T0``: initial uniform temperature of the cube (K).
            - ``Phi0``: heat flux (W.m-2) injected on the surface of the
              source cylinder.
            - ``h``: convection coefficient on the cube faces (W.m-2.K-1).

        Returns
        -------
        numpy.ndarray
            Temperature (deg C) at every mesh node at the final time, shape
            ``(config.N_NODES,)``.
        """
        p = cls.resolve_params(params)
        t0, flux, h, res, cylinder_res = p["T0"], p["Phi0"], p["h"], p["res"], p["cylinder_res"]

        a = config.CUBE_SIZE
        mesh, circle = cls.build_mesh(res, cylinder_res)

        # Node sets: the six faces (convection) and the source cylinder (heating).
        faces = [
            mesh.Nodes_Conditions(lambda x, y, z: x == 0),
            mesh.Nodes_Conditions(lambda x, y, z: x == a),
            mesh.Nodes_Conditions(lambda x, y, z: y == 0),
            mesh.Nodes_Conditions(lambda x, y, z: y == a),
            mesh.Nodes_Conditions(lambda x, y, z: z == 0),
            mesh.Nodes_Conditions(lambda x, y, z: z == a),
        ]
        nodes_convection = np.unique(np.concatenate(faces))
        nodes_source = mesh.Nodes_Cylinder(circle, direction=[0, 0, 1])

        dt = config.T_MAX / config.N_STEPS
        simu = cls._init_thermal(mesh, t0, dt)

        # Note: t starts at -dt, so the loop runs N_STEPS + 1 steps (one past T_MAX).
        # Kept as is so that existing datasets stay reproducible.
        t = -dt
        while t < config.T_MAX:
            t += dt
            cls._step(
                simu,
                (nodes_convection, cls._convection(simu, nodes_convection, h)),
                (nodes_source, flux),
            )
            if verbose:
                print(f"{t:.3f} s", end="\r")

        return cls._celsius(simu)
