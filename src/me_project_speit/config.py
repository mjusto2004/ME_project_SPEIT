"""Physical constants and shared dimensions of the project.

Everything that must stay consistent between the simulator, the dataset
files and the neural network lives here.
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
HEAT_EFFICIENCY = 0.05  # fraction of the power turned into heat
T_EXTERIOR = 293.0  # K

# --- Time discretisation -----------------------------------------------------
T_MAX = 10_000.0  # s
N_STEPS = 100

# --- Simulation parameters that can be varied (name: default, sampling range) -
# These are the possible *inputs* of a dataset. Parameters that a dataset does
# not vary take their default value.
PARAM_DEFAULTS = {"T0": 293.0, "P0": 30_000.0, "h": 300.0}
PARAM_RANGES = {
    "T0": (146.5, 293.0),  # initial temperature, K
    "P0": (15_000.0, 30_000.0),  # power, W
    "h": (150.0, 300.0),  # convection coefficient, W.m-2.K-1
}

# --- Reference mesh ----------------------------------------------------------
N_NODES = 560  # number of nodes of the mesh built from the settings above
MESH_ID = "cube_560"
