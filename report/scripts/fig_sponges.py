"""Sponge figures: example geometries with their temperature, at several mesh resolutions."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from sponge_tools import grid, rebuild  # noqa: E402

FIG = Path("report/figures")
RES = Path("report/results")
pv.OFF_SCREEN = True
RANGES = {"T0": (273.0, 303.0), "Phi0": (15000.0, 30000.0), "h": (50.0, 200.0), "r": (0.05, 0.3)}


def render(s, path, clim=None, scalars="T", cmap="inferno", cut=True, bar=False, title=""):
    g = grid(s["coord"], s["connect"], T=s["T"] if scalars == "T" else s[scalars])
    p = pv.Plotter(window_size=(860, 700))
    p.set_background("white")
    shown = g.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True) if cut else g
    p.add_mesh(shown, scalars="T", cmap=cmap, clim=clim, show_edges=False,
               show_scalar_bar=bar, scalar_bar_args=dict(title=title, vertical=True, position_x=0.88,
                                                          position_y=0.15, height=0.7, width=0.06,
                                                          color="black", fmt="%.3g", label_font_size=18))
    coord = s["coord"][s["heated"] & (s["coord"][:, 1] <= 0.5 + 1e-9)] if cut else s["coord"][s["heated"]]
    if len(coord):
        p.add_points(coord, color="#2a9d8f", point_size=7, render_points_as_spheres=True)
    p.camera_position = "iso"
    p.camera.zoom(1.0)
    p.screenshot(str(path))
    p.close()


def examples(seed=10, n=1000, picks=(0, 1, 2, 3, 4, 5, 6, 7)):
    """Render a few sponges of the training set (rebuilt), keep three contrasting ones."""
    sims, info = [], []
    for i in picks:
        s = rebuild(seed, n, i, {**RANGES, "res": (5, 10)})
        sims.append(s)
        info.append({"i": i, **{k: float(v) for k, v in s["params"].items()}, "holes": len(s["holes"]),
                     "face": s["face"], "nodes": len(s["T"]), "Tmin": float(s["T"].min()),
                     "Tmax": float(s["T"].max())})
    (RES / "sponge_examples.json").write_text(json.dumps(info, indent=2))
    for k, s in enumerate(sims):
        render(s, FIG / f"sponge_example_{k}.png", bar=True, title="T (°C)")


def resolutions():
    """The same sponge meshed at res 6, 12 and 25."""
    from me_project_speit.simulations import SpongeSimulation

    rng = np.random.default_rng(3)
    holes = rng.uniform(size=(6, 3))
    out = []
    for res in (6, 12, 25):
        T, coord, connect, heated = SpongeSimulation.run(holes, 4, T0=293, Phi0=25000, h=100, r=0.2, res=res)
        s = dict(T=T, coord=coord, connect=connect, heated=heated)
        g = grid(coord, connect, T=T)
        p = pv.Plotter(window_size=(700, 700))
        p.set_background("white")
        p.add_mesh(g.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True), color="#d9dde3", show_edges=True,
                   edge_color="#4a5260", line_width=0.4)
        p.camera_position = "iso"
        p.camera.zoom(1.15)
        p.screenshot(str(FIG / f"sponge_res_{res}.png"))
        p.close()
        out.append({"res": res, "nodes": len(T), "Tmin": float(T.min()), "Tmax": float(T.max()),
                    "Tmean": float(T.mean())})
    (RES / "sponge_resolutions.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    examples()
    resolutions()
