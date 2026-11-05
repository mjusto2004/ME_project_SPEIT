import numpy as np
from pathlib import Path
from EasyFEA import Models, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Circle, Domain, Point

folder = Path(__file__).resolve().parent
data = np.loadtxt(folder / "data_test.csv", delimiter=",")
coord = np.load(folder / "coord.npy")
X, Y = data[:, :3], data[:, 3:]

a = 1
domain = Domain(Point(0, 0), Point(a, a), a / 8)
circle = Circle(Point(a / 2, a / 2), diam=a / 3., meshSize=a / 10)
mesh = domain.Mesh_Extrude([circle], [0, 0, a], [4], ElemType.TETRA4)

assert np.allclose(mesh.coord, coord), "le maillage ne correspond pas à coord.npy"

simu = Simulations.Thermal(mesh, Models.Thermal(k=237, c=896, thickness=1))

ids = [0,1]
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