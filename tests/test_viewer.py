"""The PyVista viewer of `infer` and `plot` (rendered off-screen)."""

import numpy as np
import pytest

from me_project_speit.evaluation import Evaluation, SimulationResult, plot_evaluation, use_pyvista

pytest.importorskip("pyvista")
pytest.importorskip("matplotlib")

from me_project_speit.viewer import PredictionViewer  # noqa: E402


def result(sim, n, err, coords=True):
    rng = np.random.default_rng(sim)
    xyz = rng.uniform(size=(n, 3))
    true = xyz.sum(1)
    return SimulationResult(sim, {"p": float(sim)}, true, true + err, xyz if coords else None)


def evaluation(tmp_path, results):
    return Evaluation(tmp_path / "run" / "best_model.pt", "toy", tmp_path / "d.csv", "all", results)


def test_browse_and_cut():
    results = [result(0, 200, 0.1), result(1, 300, 0.5), result(2, 150, 0.2)]
    w = PredictionViewer(results, index=1, off_screen=True)
    assert w.actors[0].mapper.GetInput().n_points == 300
    w.show_index(w.index + 1)                      # Right
    assert w.index == 2 and w.actors[1].mapper.GetInput().n_points == 150
    w.show_index(w.index + 1)                      # wraps around
    assert w.index == 0

    w._toggle_cut()                                # c: cut at y = 0.5
    shown = w.actors[2].mapper.GetInput().n_points
    assert shown == int((results[0].coords[:, 1] <= 0.5).sum())
    assert w.actors[2].mapper.scalar_range == pytest.approx((0, 0.1))  # error scale of this simulation
    w.plotter.close()


def test_half_starts_cut():
    r = result(0, 400, 0.1)
    w = PredictionViewer([r], half=True, off_screen=True)
    assert w.actors[0].mapper.GetInput().n_points == int((r.coords[:, 1] <= 0.5).sum())
    w.plotter.close()


def test_saved_files(tmp_path):
    ev = evaluation(tmp_path, [result(0, 200, 0.1), result(1, 300, 0.5)])
    saved = plot_evaluation(ev, save=tmp_path / "a", viewer="pyvista")
    assert [p.name for p in saved] == ["a_inference.png", "a_inference_3d.png"]
    assert all(p.stat().st_size > 5000 for p in saved)
    assert [p.name for p in plot_evaluation(ev, save=tmp_path / "b", viewer="matplotlib")] == ["b_inference.png"]


def test_viewer_choice(tmp_path):
    with_coords = [result(0, 50, 0.1)]
    without = [result(0, 50, 0.1, coords=False)]
    assert use_pyvista("auto", with_coords) and not use_pyvista("auto", without)
    assert not use_pyvista("matplotlib", with_coords)
    with pytest.raises(ValueError, match="coordinates"):
        use_pyvista("pyvista", without)
    # without coordinates, auto falls back to matplotlib
    ev = evaluation(tmp_path, without)
    assert [p.name for p in plot_evaluation(ev, save=tmp_path / "c")] == ["c_inference.png"]
