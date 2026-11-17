"""Inference on a dataset: `evaluate`, its figure and the `infer` command. No EasyFEA needed."""

import numpy as np
import pytest

from me_project_speit.cli import main
from me_project_speit.datasets import DatasetSpec, load_dataset, save_spec
from me_project_speit.evaluation import evaluate, plot_evaluation
from me_project_speit.inference import Predictor
from me_project_speit.training import TrainConfig, train

from test_graph import make_graph_dataset
from test_pipeline import make_csv


@pytest.fixture
def field_run(tmp_path):
    csv = make_csv(tmp_path / "toy.csv", n=40)
    train(TrainConfig(dataset=csv, epochs=3, device="cpu", output_dir=tmp_path / "run"))
    return tmp_path / "run", csv


def test_test_subset_of_training_data_and_all_of_another(field_run, tmp_path):
    run, csv = field_run
    ev = evaluate(run)  # training dataset -> its test split
    assert ev.subset == "test" and len(ev.results) == 6  # 15 % of 40

    other = make_csv(tmp_path / "other.csv", n=10)
    ev = evaluate(run / "best_model.pt", other)
    assert ev.subset == "all" and [r.sim for r in ev.results] == list(range(10))

    # same numbers as the Predictor on the whole file
    data = load_dataset(other)
    expected = np.abs(Predictor(run / "best_model.pt").predict(data.X) - data.Y).mean()
    assert ev.metrics()["MAE"] == pytest.approx(expected, rel=1e-5)
    assert ev.worst().mae == max(r.mae for r in ev.results)

    with pytest.raises(ValueError, match="'test' subset"):
        evaluate(run, other, subset="test")


def test_per_simulation_metrics(tmp_path):
    from me_project_speit.evaluation import Evaluation, SimulationResult

    def sim(i, n, err):  # true field ranging over 0..2, constant error
        true = np.linspace(0.0, 2.0, n)
        return SimulationResult(i, {"p": 0.0}, true, true + err)

    # a big mesh with a small error and a small mesh with a large one
    ev = Evaluation(tmp_path / "m.pt", "toy", tmp_path / "d.csv", "all",
                    [sim(0, 900, 0.1), sim(1, 100, 1.0)])
    m = ev.metrics()
    assert m["MAE"] == pytest.approx((900 * 0.1 + 100 * 1.0) / 1000)  # pooled: nodes weigh
    assert m["sim_MAE"] == pytest.approx(0.55)                          # simulations weigh
    assert m["NMAE"] == pytest.approx(0.55 / 2)
    assert 0.1 <= m["sim_MAE_lo"] <= m["sim_MAE"] <= m["sim_MAE_hi"] <= 1.0
    assert m == ev.metrics()  # bootstrap is seeded


def test_mismatched_dataset_is_rejected(field_run, tmp_path):
    run, _ = field_run
    wrong = make_csv(tmp_path / "wrong.csv", n_inputs=2, n=10)
    with pytest.raises(ValueError, match="does not match"):
        evaluate(run, wrong)


def test_out_of_range_inputs_are_reported(field_run, tmp_path):
    run, _ = field_run
    ev = evaluate(run)
    ev.train_ranges = {"p0": (0.0, 0.1)}  # the toy data is uniform in [0, 1]
    assert ev.outside_training_ranges()["p0"] > 0
    assert "outside the training range" in ev.summary()


def test_field_without_coordinates_still_plots(field_run, tmp_path):
    pytest.importorskip("matplotlib")
    run, _ = field_run
    ev = evaluate(run)
    assert ev.results[0].coords is None  # mesh "toy": no known coordinates
    saved = plot_evaluation(ev, save=tmp_path / "fig")
    assert saved[0].name == "fig_inference.png" and saved[0].stat().st_size > 1000


def test_graph_inference_and_3d_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = make_graph_dataset().save(tmp_path / "g.csv")
    train(TrainConfig(dataset=path, model="gnn", model_kwargs={"hidden": [8], "layers": 1},
                      epochs=2, batch_size=4, device="cpu", output_dir=tmp_path / "run"))
    ev = evaluate(tmp_path / "run", path, subset="all")
    assert len(ev.results) == 8
    r = ev.get(3)
    assert r.coords.shape == (len(r.true), 3) and list(r.params) == ["P0"]
    assert plot_evaluation(ev, sim=3, half=True, save=tmp_path / "g")[0].exists()
    with pytest.raises(ValueError, match="not in the evaluated set"):
        ev.get(99)


def test_pointwise_inference(tmp_path):
    n_sims, n_nodes = 10, 12
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(n_nodes, 3))
    rows = []
    for _ in range(n_sims):
        p = rng.uniform()
        rows.append(np.column_stack([np.full(n_nodes, p), coords, p * coords.sum(1)]))
    path = tmp_path / "pw.csv"
    np.savetxt(path, np.vstack(rows), delimiter=",")
    save_spec(DatasetSpec("pw", ["P0", "x", "y", "z"], 1, mesh_id="toy", kind="pointwise",
                          rows_per_sample=n_nodes), path)
    train(TrainConfig(dataset=path, model="coord_mlp", epochs=2, device="cpu", output_dir=tmp_path / "run"))
    ev = evaluate(tmp_path / "run", subset="all")
    assert len(ev.results) == n_sims
    assert np.allclose(ev.results[0].coords, coords, atol=1e-6)


def test_infer_command(field_run, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    run, csv = field_run
    main(["infer", "--run", str(run), "--dataset", str(csv), "--subset", "all",
          "--csv", str(tmp_path / "errors.csv"), "--save", str(tmp_path / "fig"), "--sim", "2"])
    out = capsys.readouterr().out
    assert "95 % CI" in out and "NMAE" in out and "Worst simulations" in out
    table = np.genfromtxt(tmp_path / "errors.csv", delimiter=",", names=True)
    assert len(table) == 40 and set(table.dtype.names) >= {"sim", "p0", "MAE", "NMAE", "RMSE", "max_abs"}
    assert (tmp_path / "fig_inference.png").exists()

    with pytest.raises(SystemExit):
        main(["infer", "--run", str(run), "--sim", "999", "--no-plot"])
