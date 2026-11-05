"""Copy of the model in simulation.py that also returns the time history."""

import numpy as np
from EasyFEA import Models, ElemType, Simulations
from EasyFEA.Geoms import Circle, Domain, Point

A = 1.0
MESH_RES_R = 10
MESH_RES_DOMAIN = 8
RHO = 2700.0
CP = 896.0
K = 237.0
DIAMETER = A / 3.0
ETA = 0.05
T_EXT = 293.0
T_MAX = 10000.0
N_STEPS = 100


def run(T0, P, h):
    """Return times (s), temperature history (°C, steps x nodes) and node coordinates."""
    domain = Domain(Point(0, 0), Point(A, A), A / MESH_RES_DOMAIN)
    circle = Circle(Point(A / 2, A / 2), diam=DIAMETER, meshSize=A / MESH_RES_R)
    mesh = domain.Mesh_Extrude([circle], [0, 0, A], [4], ElemType.TETRA4)

    faces = [mesh.Nodes_Conditions(lambda x, y, z, ax=ax, v=v: (x, y, z)[ax] == v)
             for ax in range(3) for v in (0, A)]
    nodes_convection = np.unique(np.concatenate(faces))
    nodes_circle = mesh.Nodes_Cylinder(circle, direction=[0, 0, 1])

    power_surficial = P * ETA / (np.pi * DIAMETER * 1)
    dt = T_MAX / N_STEPS

    simu = Simulations.Thermal(mesh, Models.Thermal(k=K, c=CP, thickness=1))
    simu.rho = RHO
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)
    simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * T0)

    times, T_hist = [0.0], [simu.thermal.copy()]
    t = -dt  # same loop as simulation.py
    while t < T_MAX:
        t += dt
        phi = h * (T_EXT - simu.thermal[nodes_convection])
        simu.Bc_Init()
        simu.add_surfLoad(nodes_convection, [phi], ["t"])
        simu.add_surfLoad(nodes_circle, [power_surficial], ["t"])
        simu.Solve()
        simu.Save_Iter()
        times.append(t + dt)
        T_hist.append(simu.thermal.copy())

    return np.array(times), np.array(T_hist) - 273.15, mesh.coord.copy()
