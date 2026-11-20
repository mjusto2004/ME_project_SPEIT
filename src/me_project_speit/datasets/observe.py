import numpy as np
from EasyFEA import Models, Simulations, PyVista

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ajoute src/ au chemin
from me_project_speit.simulations import CubeSimulation, config


CSV_PATHS = [
    "/Users/tristanjin/Projects/ME_project_SPEIT/data/new_ranges/dataset_20261114_180338_360.csv",
    # "/Users/tristanjin/Projects/ME_project_SPEIT/data/new_ranges/dataset_20261114_180558_100.csv",
    # "/Users/tristanjin/Projects/ME_project_SPEIT/data/new_ranges/dataset_20261114_180637_720.csv",
]

SAMPLE_IDS = [0, 1, 2]
# -----------------------------------------------------------------------------

mesh, _ = CubeSimulation.build_mesh()
simu = Simulations.Thermal(
    mesh, Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1)
)

if not CSV_PATHS:
    raise SystemExit("Aucun dataset sélectionné : décommente au moins un chemin dans CSV_PATHS.")

for csv_path in CSV_PATHS:
    data = np.loadtxt(csv_path, delimiter=",")
    X, Y = data[:, :3], data[:, 3:]
    assert Y.shape[1] == mesh.Nn, f"{csv_path} : {Y.shape[1]} sorties, attendu {mesh.Nn}"

    print(f"\n{csv_path}")
    print(f"  {len(data)} échantillons, {Y.shape[1]} nœuds")
    for i in SAMPLE_IDS:
        print(f"  [{i}] T0={X[i, 0]:.1f} K  Phi0={X[i, 1]:.0f} W/m²  h={X[i, 2]:.1f} W/m²K")

    # Échelle de couleur commune aux échantillons d'un même dataset.
    clim = (Y[SAMPLE_IDS].min(), Y[SAMPLE_IDS].max())
    for i in SAMPLE_IDS:
        title = f"T0 = {X[i, 0]:.1f} K   P = {X[i, 1]:.0f} W/m²   h = {X[i, 2]:.1f} W/m²K"
        PyVista.Plot(
            simu,
            Y[i],
            plotMesh=True,
            nodeValues=True,
            clim=clim,
            colorbarTitle="Température (°C)",
            title=title,
        ).show()
