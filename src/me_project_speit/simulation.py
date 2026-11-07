"""Transient thermal finite-element simulation (requires ``EasyFEA``).

A unit cube is heated by a cylindrical source along its z axis and cooled by
convection on its six faces. :func:`run_simulation` returns the temperature
field at the last time step.
"""

from __future__ import annotations

import numpy as np

from . import config


def build_mesh():
    """Build the 3D tetrahedral mesh of the cube and the heat-source circle.

    Returns
    -------
    mesh, circle
        The EasyFEA mesh and the circle (cross-section of the source).
    """
    from EasyFEA import ElemType
    from EasyFEA.Geoms import Circle, Domain, Point

    a = config.CUBE_SIZE
    domain = Domain(Point(0, 0), Point(a, a), a / config.MESH_RESOLUTION_DOMAIN)
    circle = Circle(
        Point(a / 2, a / 2),
        diam=config.SOURCE_DIAMETER,
        meshSize=a / config.MESH_RESOLUTION_SOURCE,
    )
    mesh = domain.Mesh_Extrude(
        [circle], [0, 0, a], [config.EXTRUSION_LAYERS], ElemType.TETRA4
    )
    return mesh, circle


def run_simulation(verbose: bool = False, **params: float) -> np.ndarray:
    """Run one transient thermal simulation.

    Parameters
    ----------
    verbose : bool
        Print the progress of the time loop.
    **params : float
        Any of the names in ``config.PARAM_DEFAULTS``; missing ones take their
        default value.

        - ``T0``: initial uniform temperature of the cube (K).
        - ``P0``: power of the source (W); only ``HEAT_EFFICIENCY`` is injected.
        - ``h``: convection coefficient on the cube faces (W.m-2.K-1).

    Returns
    -------
    numpy.ndarray
        Temperature (deg C) at every mesh node at the final time, shape
        ``(config.N_NODES,)``.
    """
    from EasyFEA import Models, Simulations

    unknown = set(params) - set(config.PARAM_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown simulation parameter(s) {sorted(unknown)}; "
            f"available: {sorted(config.PARAM_DEFAULTS)}"
        )
    p = {**config.PARAM_DEFAULTS, **params}
    t0, power, h = p["T0"], p["P0"], p["h"]

    a = config.CUBE_SIZE
    mesh, circle = build_mesh()

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

    # Heat flux per unit of source surface.
    power_surficial = power * config.HEAT_EFFICIENCY / (np.pi * config.SOURCE_DIAMETER * 1)

    model = Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1)
    simu = Simulations.Thermal(mesh, model)
    simu.rho = config.RHO

    dt = config.T_MAX / config.N_STEPS
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)
    simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * t0)

    t = -dt
    while t < config.T_MAX:
        t += dt
        temperature = simu.thermal
        flux = h * (config.T_EXTERIOR - temperature[nodes_convection])
        simu.Bc_Init()
        simu.add_surfLoad(nodes_convection, [flux], ["t"])
        simu.add_surfLoad(nodes_source, [power_surficial], ["t"])
        simu.Solve()
        simu.Save_Iter()
        if verbose:
            print(f"{t:.3f} s", end="\r")

    return simu.thermal - 273.15
