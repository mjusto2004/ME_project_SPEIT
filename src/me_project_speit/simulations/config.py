"""Constants of the simulations: geometry, material, time, parameters, mesh.

Everything that must stay consistent between the simulators, the dataset files
and the neural network lives here.

Each simulation has one dictionary of parameters, ``<SIMULATION>_PARAMS``:
name -> :class:`Param` (default value and, if a dataset can vary it, its range).
``PARAMS`` holds the parameters common to every simulation, and the others extend
it with ``**``:

- ``CUBE_PARAMS``: :class:`~.cube.CubeSimulation` (dataset kinds ``field`` and
  ``pointwise``): the common ones plus its mesh settings;
- ``SPONGE_PARAMS``: :class:`~.sponge.SpongeSimulation` (dataset kind ``graph``):
  the common ones plus the hole radius ``r`` and the mesh resolution ``res``.

To add a parameter, add one ``Param`` line to the right dictionary.
"""

from typing import NamedTuple


class Param(NamedTuple):
    """A simulation parameter: its default and how a dataset can sample it."""

    default: float
    range: tuple[float, float] | None = None  # (min, max); None: cannot be varied
    integer: bool = False  # sampled values are rounded to an int


def defaults(params: dict[str, Param]) -> dict[str, float]:
    """name -> default value of every parameter."""
    return {name: p.default for name, p in params.items()}


def ranges(params: dict[str, Param]) -> dict[str, tuple[float, float]]:
    """name -> (min, max) of the parameters a dataset can vary."""
    return {name: p.range for name, p in params.items() if p.range is not None}


# --- Geometry (unit cube with a cylindrical heat source along z) -------------
CUBE_SIZE = 1.0  # m
SOURCE_DIAMETER = CUBE_SIZE / 3.0  # m
MESH_RESOLUTION_DOMAIN = 8
MESH_RESOLUTION_SOURCE = 10
EXTRUSION_LAYERS = 4

# --- Material (aluminium) ----------------------------------------------------
RHO = 2700.0  # kg.m-3
CP = 896.0  # J.kg-1.K-1
CONDUCTIVITY = 237.0  # W.m-1.K-1
T_EXTERIOR = 293.0  # K

# --- Time discretisation -----------------------------------------------------
T_MAX = 10_000.0  # s
N_STEPS = 100

# --- Parameters common to every simulation -----------------------------------
PARAMS = {
    "T0": Param(293.0, (273.0, 303.0)),  # initial temperature, K
    "Phi0": Param(1000.0, (15000.0, 30000.0)),  # heat flux, W.m-2
    "h": Param(300.0, (50.0, 200.0)),  # convection coefficient, W.m-2.K-1
}

# --- Cube (dataset kinds 'field' and 'pointwise') ------------------------------
# "res" / "cylinder_res": mesh divisions of the cube edge / of the source circle.
# They have no range on purpose: changing them changes the number of nodes, but
# these datasets assume N_NODES. They can be passed to CubeSimulation.run by hand.
CUBE_PARAMS = {
    **PARAMS,
    "res": Param(MESH_RESOLUTION_DOMAIN),
    "cylinder_res": Param(MESH_RESOLUTION_SOURCE),
}

# --- Reference mesh ----------------------------------------------------------
N_NODES = 560  # number of nodes of the mesh built from the settings above
MESH_ID = "cube_560"

# --- Sponge (dataset kind 'graph'): unit cube minus spherical holes ------------
SPONGE_PARAMS = {
    **PARAMS,
    "r": Param(0.15, (0.05, 0.3)),  # radius of the holes, m
    "res": Param(8, (10, 25), integer=True),  # mesh divisions of the cube edge (element size = 1/res)
}
MAX_HOLES = 20  # the number of holes is drawn in 1..MAX_HOLES

# Face index -> (axis, coordinate): 0: x=0, 1: x=1, 2: y=0, 3: y=1, 4: z=0, 5: z=1
# (the face through which a sponge is heated).
FACES = [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]

# Time stepping of the sponge (explicit convection, see SpongeSimulation.run).
STABILITY_SAFETY = 0.2  # fraction of the explicit-convection stability limit used for dt
MAX_STEPS = 400  # cap on the number of time steps
T_VALID = (-273.15, 2000.0)  # deg C; a simulation outside this range has diverged
