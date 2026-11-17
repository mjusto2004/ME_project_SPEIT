"""Interactive 3D view of predictions with PyVista (GPU rendering, smooth with large meshes).

One window, three linked views of the same simulation: true temperature,
prediction, absolute error. Rotating or zooming one view moves the others.

- drag the plane (left view) to cut the part and see inside; ``c`` turns the cut on/off;
- ``Right`` / ``Left``: next / previous simulation of the list given;
- ``r``: reset the camera.

Used by ``infer`` and ``plot``; :func:`available` says whether PyVista is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

HELP = "drag the plane: cut   c: cut on/off   Right/Left: next/previous simulation   r: reset view"


def available() -> bool:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return False
    return True


def _point_size(n: int) -> float:
    """Larger points on coarse meshes, smaller on fine ones (pixels)."""
    return float(np.clip(150.0 / np.cbrt(max(n, 1)), 3.0, 10.0))


class PredictionViewer:
    """Window showing ``results[index]`` (true / predicted / error), browsable with the arrow keys.

    Parameters
    ----------
    results : sequence of :class:`~.evaluation.SimulationResult` with coordinates.
    index : position in ``results`` shown first.
    label : ``label(position, result)`` -> text naming the simulation.
    half : start with the cut at y = 0.5 (shows the nodes with y <= 0.5).
    title : window title.
    off_screen : render without a window (for :meth:`screenshot`).
    """

    PANELS = (  # panel title, colour-bar title (unique per window), colour map
        ("True temperature", "true (°C)", "inferno"),
        ("Predicted temperature", "predicted (°C)", "inferno"),
        ("Absolute error", "|error| (°C)", "viridis"),
    )

    def __init__(
        self,
        results: Sequence,
        index: int = 0,
        label: Callable[[int, object], str] | None = None,
        half: bool = False,
        title: str = "Prediction",
        off_screen: bool = False,
    ):
        import pyvista as pv

        if not results or any(r.coords is None for r in results):
            raise ValueError("The 3D viewer needs the node coordinates of every simulation.")
        self.pv = pv
        self.results, self.index = list(results), index
        self.label = label or (lambda i, r: f"simulation {r.sim}")
        self.cut = half
        self.normal, self.origin = np.array([0.0, 1.0, 0.0]), np.array([0.5, 0.5, 0.5])
        self._ready = False

        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1650, 620), off_screen=off_screen, title=title)
        self.actors = []
        for col, (name, bar, cmap) in enumerate(self.PANELS):
            self.plotter.subplot(0, col)
            self.plotter.add_text(name, position="upper_edge", font_size=11)
            actor = self.plotter.add_mesh(
                pv.PolyData(np.zeros((1, 3))), scalars=np.zeros(1), cmap=cmap,
                render_points_as_spheres=True, point_size=5,
                scalar_bar_args={"title": bar, "vertical": True, "fmt": "%.3g", "title_font_size": 13,
                                 "label_font_size": 11, "position_x": 0.86, "position_y": 0.2,
                                 "width": 0.06, "height": 0.6},
            )
            self.actors.append(actor)
            self.plotter.add_axes(line_width=2, labels_off=False)
        self.plotter.link_views()

        self.plotter.subplot(0, 0)
        self.plotter.add_plane_widget(self._on_plane, normal="y", origin=self.origin,
                                      bounds=self._bounds(), normal_rotation=True)
        self.plotter.add_key_event("Right", lambda: self.show_index(self.index + 1))
        self.plotter.add_key_event("Left", lambda: self.show_index(self.index - 1))
        self.plotter.add_key_event("c", self._toggle_cut)
        self.show_index(index)
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.8)
        self._ready = True

    def _bounds(self):
        xyz = np.concatenate([r.coords for r in self.results])
        lo, hi = xyz.min(0), xyz.max(0)
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    # ---- interaction ---------------------------------------------------------
    def _on_plane(self, normal, origin) -> None:
        self.normal, self.origin = np.asarray(normal, float), np.asarray(origin, float)
        if self._ready:  # the widget calls back once at creation: keep the starting state
            self.cut = True
            self._update()

    def _toggle_cut(self) -> None:
        self.cut = not self.cut
        self._update()

    def show_index(self, index: int) -> None:
        self.index = index % len(self.results)
        self._update()

    # ---- drawing -------------------------------------------------------------
    def _update(self) -> None:
        r = self.results[self.index]
        keep = np.ones(len(r.coords), bool)
        if self.cut:
            keep = (r.coords - self.origin) @ self.normal <= 0
        error = np.abs(r.error)
        lo, hi = min(r.true.min(), r.pred.min()), max(r.true.max(), r.pred.max())
        values = ((r.true, (lo, hi)), (r.pred, (lo, hi)), (error, (0.0, max(error.max(), 1e-9))))
        size = _point_size(len(r.coords))
        for actor, (v, clim) in zip(self.actors, values):
            cloud = self.pv.PolyData(np.ascontiguousarray(r.coords[keep], dtype=float))
            cloud.point_data.set_array(np.asarray(v[keep], float), "values")
            cloud.set_active_scalars("values")
            actor.mapper.SetInputData(cloud)
            actor.mapper.scalar_range = clim
            actor.prop.point_size = size

        params = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
        info = (
            f"{self.label(self.index, r)}  [{self.index + 1}/{len(self.results)}]\n"
            f"{params}\nMAE {r.mae:.4f}   max {r.max_abs:.4f}\n{len(r.true)} nodes"
            + (f" ({int(keep.sum())} shown, cut)" if self.cut else "")
        )
        self.plotter.subplot(0, 1)
        self.plotter.add_text(info, position="lower_edge", font_size=6, name="info")
        self.plotter.subplot(0, 2)
        self.plotter.add_text(HELP.replace("   ", "\n"), position="upper_right", font_size=7, name="help")
        self.plotter.render()

    # ---- output --------------------------------------------------------------
    def show(self) -> None:
        self.plotter.show()

    def screenshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.clear_plane_widgets()  # not part of the picture
        self.plotter.screenshot(str(path))
        self.plotter.close()
        return path
