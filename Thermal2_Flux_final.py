# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Thermal2_Flux
========

Transient thermal simulation: with user defined BCs neumann
"""

# sphinx_gallery_thumbnail_number = -1
import matplotlib.pyplot as plt
import numpy as np
from EasyFEA import Terminal, Folder, Models, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Circle, Domain, Point

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 3  # Set the simulation dimension (2D or 3D)

    # outputs
    folder = Folder.Results_Dir() + f"{dim}D"
    makeMovie = False
    result = "thermal"
    # Unit kg. m. s
    # geom
    a = 1
    mesh_resolution_r = 10
    mesh_resolution_Domain = 8
    # Thermal properties
    h = 300 # W.m-2.K-1
    rho = 2700 # kg.m-3
    Cp = 896 # J.kg-1.K-1
    k = 237 # W.m-1.K-1
    Diameter=a/3.
    #heat generation coefficient: Eta
    Eta = 0.05
    power = 300000*Eta # 450kW.h
    power_surficial = power/(np.pi*Diameter*1)
    print(power_surficial)
    # load
    T_ext=293 # K
    Tmax = 10000
    T0 = 293
    N = 100
    dt = Tmax / N
    #Tmax = dt
    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    domain = Domain(Point(0,0), Point(a, a), a / mesh_resolution_Domain)
    circle = Circle(Point(a / 2, a / 2), diam=Diameter, meshSize=a / mesh_resolution_r)

    if dim == 2:
        mesh = domain.Mesh_2D([circle], ElemType.TRI6)
    else:
        mesh = domain.Mesh_Extrude([circle], [0, 0, a], [4], ElemType.TETRA4) # PRISM18 TETRA4

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXa = mesh.Nodes_Conditions(lambda x, y, z: x == a)
    # YZ
    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodesYa = mesh.Nodes_Conditions(lambda x, y, z: y == a)
    nodesZ0 = mesh.Nodes_Conditions(lambda x, y, z: z == 0)
    nodesZa = mesh.Nodes_Conditions(lambda x, y, z: z == a)
    nodes_convection = nodes_convection = np.unique(np.concatenate([nodesX0, nodesXa, nodesY0, nodesYa, nodesZ0, nodesZa]))
    #nodes_convection = nodes_convection = np.unique(np.concatenate([nodesX0, nodesXa, nodesY0, nodesYa]))
    nodesCircle = mesh.Nodes_Cylinder(circle, direction=[0, 0, 1])

    if dim == 3:
        print(f"Volume: {mesh.volume:.3}")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    thermalModel = Models.Thermal(k=k, c=Cp, thickness=1)
    simu = Simulations.Thermal(mesh, thermalModel)
    simu.rho = rho

    #simu.add_dirichlet(nodesX0, [0], ["t"])
    #simu.add_dirichlet(nodesXa, [40], ["t"])
    #simu.add_dirichlet(nodes_convection, [293], ["t"])
    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)
    simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * T0)
    #simu.add_surfLoad(nodesZ0, [power_surficial], ["t"])
    t = -dt  # init time
    #simu.Get_x0(T0)
# Target point
    target = np.array([0.5, 1/3., 0.5])

    # Find the nearest mesh node
    dist = np.linalg.norm(mesh.coord - target, axis=1)
    node_id = np.argmin(dist)

    print("Selected node:", node_id)
    print("Coordinates:", mesh.coord[node_id])
    time_history = []
    temperature_history = []
    #
    while t < Tmax:
        t += dt
        T = simu.thermal
        phi = h * (T_ext - T[nodes_convection])
        simu.Bc_Init()
        simu.add_surfLoad(nodes_convection, [phi], ["t"])
        simu.add_surfLoad(nodesCircle, [power_surficial], ["t"])
        #
        simu.Solve()
        simu.Save_Iter()
        # record the temperature
        time_history.append(t)
        temperature_history.append(simu.thermal[node_id]-273)

        print(f"{t:.3f} s", end="\r")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    # Plot the temperature history
    #import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(time_history, temperature_history, "-o")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Temperature (°C)")
    # plt.title(f"Temperature at node {node_id}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    print(simu)

    #PyVista.Plot(simu, result, plotMesh=True, nodeValues=True).show()
    # Convert temperature from Kelvin to Celsius
    T_celsius = simu.thermal - 273.15

    # PyVista.Plot(
    #     simu,
    #     T_celsius,
    #     plotMesh=True,
    #     nodeValues=True,
    #     clim=(25, 40),
    #     colorbarTitle="Temperature (°C)"
    # ).show()

    if makeMovie:
       PyVista.Movie_simu(simu, result, folder, "Temperature (K).gif", plotMesh=True)