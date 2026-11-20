"""Dataset figures of the report: cube mesh and fields, sponge examples, data statistics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

FIG = Path("report/figures")
RES = Path("report/results")
FIG.mkdir(parents=True, exist_ok=True)
pv.OFF_SCREEN = True
CMAP = "inferno"


def cube_grid(values=None):
    coord = np.load(RES / "cube_coord.npy")
    connect = np.load(RES / "cube_connect.npy")
    cells = np.hstack([np.full((len(connect), 1), 4), connect]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(connect), pv.CellType.TETRA), coord)
    if values is not None:
        grid.point_data["T"] = values
    return grid


def shot(plotter, path, zoom=1.0):
    plotter.camera_position = "iso"
    plotter.camera.zoom(zoom)
    plotter.screenshot(str(path), transparent_background=False)
    plotter.close()


def cube_mesh_figure():
    from me_project_speit.simulations import config

    grid = cube_grid()
    coord = np.asarray(grid.points)
    r = np.hypot(coord[:, 0] - 0.5, coord[:, 1] - 0.5)
    on_source = np.isclose(r, config.SOURCE_DIAMETER / 2, atol=1e-3)
    p = pv.Plotter(window_size=(900, 800))
    p.set_background("white")
    clipped = grid.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True, crinkle=True)
    p.add_mesh(clipped, color="#d9dde3", show_edges=True, edge_color="#5a6270", line_width=0.6)
    p.add_points(coord[on_source & (coord[:, 1] <= 0.5 + 1e-9)], color="#e4572e", point_size=11,
                 render_points_as_spheres=True)
    cyl = pv.Cylinder(center=(0.5, 0.5, 0.5), direction=(0, 0, 1), radius=config.SOURCE_DIAMETER / 2,
                      height=1.0, resolution=60).clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True)
    p.add_mesh(cyl, color="#e4572e", opacity=0.35)
    shot(p, FIG / "cube_mesh.png", 1.15)


def cube_fields_figure():
    data = np.loadtxt("data/report/cube_independent_100.csv", delimiter=",")
    X, Y = data[:, :3], data[:, 3:]
    # three contrasting simulations: hot (low h, high flux), median, cold (high h, low flux)
    score = X[:, 1] / X[:, 2]
    picks = [int(np.argmax(score)), int(np.argsort(score)[len(score) // 2]), int(np.argmin(score))]
    lo, hi = Y[picks].min(), Y[picks].max()
    for k, i in enumerate(picks):
        grid = cube_grid(Y[i])
        clipped = grid.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True, crinkle=False)
        p = pv.Plotter(window_size=(700, 700))
        p.set_background("white")
        p.add_mesh(clipped, scalars="T", cmap=CMAP, clim=(lo, hi), show_scalar_bar=False)
        p.add_mesh(clipped.extract_feature_edges(feature_angle=30), color="#333333", line_width=1.5)
        shot(p, FIG / f"cube_field_{k}.png", 1.2)
    # shared colour bar
    fig, ax = plt.subplots(figsize=(6, 0.5))
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(lo, hi), CMAP), cax=ax, orientation="horizontal",
                 label="temperature after 10 000 s (°C)")
    fig.savefig(FIG / "cube_field_colorbar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(RES / "cube_field_picks.txt", "w") as f:
        for i in picks:
            f.write(f"{i} T0={X[i,0]:.1f} Phi0={X[i,1]:.0f} h={X[i,2]:.1f} Tmin={Y[i].min():.1f} Tmax={Y[i].max():.1f}\n")


def cube_stats_figure():
    data = np.loadtxt("data/report/cube_train_all720.csv", delimiter=",")
    X, Y = data[:, :3], data[:, 3:]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    sc = axs[0].scatter(X[:, 2], X[:, 1], c=Y.mean(axis=1), s=8, cmap=CMAP)
    axs[0].set_xlabel("h (W·m⁻²·K⁻¹)")
    axs[0].set_ylabel("Φ₀ (W·m⁻²)")
    fig.colorbar(sc, ax=axs[0], label="mean T (°C)")
    axs[1].hist(Y.mean(axis=1), bins=40, color="#3b6ea5")
    axs[1].set_xlabel("mean temperature of a simulation (°C)")
    axs[1].set_ylabel("simulations")
    axs[2].scatter(X[:, 1] / X[:, 2], Y.mean(axis=1), s=6, c="#3b6ea5")
    axs[2].set_xlabel("Φ₀ / h (K)")
    axs[2].set_ylabel("mean T (°C)")
    fig.tight_layout()
    fig.savefig(FIG / "cube_stats.png", dpi=200)
    plt.close(fig)


def pod_figure():
    data = np.loadtxt("data/report/cube_train_all720.csv", delimiter=",")
    Y = data[:, 3:]
    s = np.linalg.svd(Y - Y.mean(axis=0), compute_uv=False)
    energy = s**2 / (s**2).sum()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].semilogy(np.arange(1, 21), energy[:20], "o-", color="#3b6ea5")
    axs[0].set_xlabel("mode index")
    axs[0].set_ylabel("share of the variance")
    axs[0].grid(alpha=0.3)
    axs[1].semilogy(np.arange(1, 21), 1 - np.cumsum(energy)[:20] + 1e-17, "o-", color="#e4572e")
    axs[1].set_xlabel("number of modes kept")
    axs[1].set_ylabel("variance not explained")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pod_spectrum.png", dpi=200)
    plt.close(fig)
    np.savetxt(RES / "pod_energy.txt", energy[:10])

    # first three modes on the mesh
    _, _, Vt = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    for k in range(3):
        grid = cube_grid(Vt[k] * np.sign(Vt[k][np.argmax(np.abs(Vt[k]))]))
        clipped = grid.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True)
        p = pv.Plotter(window_size=(600, 600))
        p.set_background("white")
        m = np.abs(Vt[k]).max()
        p.add_mesh(clipped, scalars="T", cmap="RdBu_r", clim=(-m, m), show_scalar_bar=False)
        p.add_mesh(clipped.extract_feature_edges(feature_angle=30), color="#333333", line_width=1.5)
        shot(p, FIG / f"pod_mode_{k + 1}.png", 1.2)


if __name__ == "__main__":
    cube_mesh_figure()
    cube_fields_figure()
    cube_stats_figure()
    pod_figure()
