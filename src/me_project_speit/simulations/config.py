"""Constants of the simulations: geometry, material, time, parameters, mesh.

Everything that must stay consistent between the simulators, the dataset files
and the neural network lives here.

There are two simulations, each with its own parameters:

- ``PARAM_DEFAULTS`` / ``PARAM_RANGES``: :class:`~.cube.CubeSimulation` (kinds
  ``field`` and ``pointwise``);
- ``SPONGE_PARAM_DEFAULTS`` / ``SPONGE_PARAM_RANGES``:
  :class:`~.sponge.SpongeSimulation` (kind ``graph``), the same plus the hole
  radius ``r`` and the mesh resolution ``res``.
"""

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

# --- Simulation parameters that can be varied (name: default, sampling range) -
# These are the possible *inputs* of a dataset. Parameters that a dataset does
# not vary take their default value.
PARAM_DEFAULTS = {"T0": 293.0, "Phi0": 1000.0, "h": 300.0}
PARAM_RANGES = {
    "T0": (273.0, 303.0),  # initial temperature, K
    "Phi0": (15000.0, 30000.0),  # heat flux, W.m-2
    "h": (50.0, 200.0),  # convection coefficient, W.m-2.K-1
}

# --- Reference mesh ----------------------------------------------------------
N_NODES = 560  # number of nodes of the mesh built from the settings above
MESH_ID = "cube_560"

# --- "Sponge" geometry (dataset kind 'graph'): unit cube minus spherical holes -
# Parameters sampled for each simulation, as (min, max). 'res' is rounded to an int.
SPONGE_PARAM_RANGES = {
    "T0": PARAM_RANGES["T0"],
    "Phi0": PARAM_RANGES["Phi0"],
    "h": PARAM_RANGES["h"],
    "r": (0.05, 0.3),  # radius of the holes (m)
    "res": (10, 25),  # mesh divisions of the cube edge (element size = 1/res)
}
SPONGE_PARAM_DEFAULTS = {**PARAM_DEFAULTS, "r": 0.15, "res": 8}
INTEGER_PARAMS = {"res"}
MAX_HOLES = 20  # the number of holes is drawn in 1..MAX_HOLES

# Face index -> (axis, coordinate): 0: x=0, 1: x=1, 2: y=0, 3: y=1, 4: z=0, 5: z=1
# (the face through which a sponge is heated).
FACES = [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]

# Time stepping of the sponge (explicit convection, see SpongeSimulation.run).
STABILITY_SAFETY = 0.2  # fraction of the explicit-convection stability limit used for dt
MAX_STEPS = 400  # cap on the number of time steps
T_VALID = (-273.15, 2000.0)  # deg C; a simulation outside this range has diverged
