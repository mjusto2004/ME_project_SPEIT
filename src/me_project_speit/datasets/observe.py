#ce code permet d'afficher le dataset qui est sous format CSV

import numpy as np
from pathlib import Path
from EasyFEA import Models, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Circle, Domain, Point

folder = Path(__file__).resolve().parents[3] / "data" / "new_ranges"
data = np.loadtxt(folder / "dataset_20261114_180558_100.csv", delimiter=",")
X, Y = data[:, :3], data[:, 3:]

a = 1
domain = Domain(Point(0, 0), Point(a, a), a / 8)
circle = Circle(Point(a / 2, a / 2), diam=a / 3., meshSize=a / 10)
mesh = domain.Mesh_Extrude([circle], [0, 0, a], [4], ElemType.TETRA4)

simu = Simulations.Thermal(mesh, Models.Thermal(k=237, c=896, thickness=1))

ids = [4,16,25] #tu choisis les échantillons que tu veux ici
clim = (Y[ids].min(), Y[ids].max())

for i in ids:
    PyVista.Plot(
        simu,
        Y[i],
        plotMesh=True,
        nodeValues=True,
        clim=clim,
        colorbarTitle="Température (°C)",
    ).show()