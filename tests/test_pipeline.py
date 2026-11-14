"""Tests that do not need EasyFEA: model, dataset I/O, training, inference."""

from pathlib import Path

import numpy as np
import pytest
import torch

from me_project_speit.datasets import DatasetSpec, load_dataset, save_spec, split_indices
from me_project_speit.inference import Predictor
from me_project_speit.models import build_model
from me_project_speit.training import TrainConfig, train


def make_csv(path: Path, n_inputs=3, n_outputs=20, n=40, name="toy") -> Path:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(n, n_inputs))
    y = x @ rng.uniform(size=(n_inputs, n_outputs))
    np.savetxt(path, np.hstack([x, y]), delimiter=",")
    spec = DatasetSpec(
        name=name,
        input_names=[f"p{i}" for i in range(n_inputs)],
        n_outputs=n_outputs,
        mesh_id="toy",
    )
    save_spec(spec, path)
    return path


@pytest.mark.parametrize("n_in,n_out", [(3, 560), (2, 7), (5, 100)])
def test_mlp_adapts_to_any_shape(n_in, n_out):
    spec = DatasetSpec("s", [f"p{i}" for i in range(n_in)], n_out)
    out = build_model("mlp", spec)(torch.randn(10, n_in))
    assert out.shape == (10, n_out)


def test_split_is_a_partition():
    s = split_indices(100, seed=1)
    assert sorted(np.concatenate([s.train, s.val, s.test])) == list(range(100))


def test_load_dataset_rejects_wrong_columns(tmp_path):
    path = make_csv(tmp_path / "a.csv")
    other = DatasetSpec("a", ["p0"], 5)  # claims 6 columns, file has 23
    save_spec(other, path)
    with pytest.raises(ValueError, match="expected 6 columns"):
        load_dataset(path)


@pytest.mark.parametrize("n_in,n_out,hidden", [(3, 20, None), (2, 7, [16, 16])])
def test_train_and_predict_end_to_end(tmp_path, n_in, n_out, hidden):
    csv = make_csv(tmp_path / "toy.csv", n_in, n_out)
    result = train(
        TrainConfig(
            dataset=csv,
            model_kwargs={"hidden": hidden} if hidden else {},
            epochs=5,
            device="cpu",
            output_dir=tmp_path / "run",
        )
    )
    assert result["checkpoint"].exists() and result["history"].exists()
    assert np.isfinite(result["test_mae"])

    predictor = Predictor(result["checkpoint"])
    assert predictor.predict(np.zeros(n_in)).shape == (1, n_out)
    with pytest.raises(ValueError, match="expects"):
        predictor.predict(np.zeros(n_in + 1))


def test_dataset_resolved_by_name(tmp_path):
    make_csv(tmp_path / "named.csv")
    result = train(
        TrainConfig(dataset="named", data_dir=tmp_path, epochs=2, device="cpu",
                    output_dir=tmp_path / "run")
    )
    assert result["checkpoint"].exists()


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(dataset="x.csv", epochs=0)


def test_fit_kwargs_reach_fit_and_are_saved(tmp_path):
    import torch

    csv = tmp_path / "d.csv"
    rng = np.random.default_rng(0)
    np.savetxt(csv, rng.random((60, 2 + 4)), delimiter=",")
    from me_project_speit.datasets import DatasetSpec, save_spec

    save_spec(DatasetSpec("d", ["a", "b"], 4), csv)
    result = train(TrainConfig(dataset=csv, epochs=2, device="cpu", output_dir=tmp_path / "run",
                               fit_kwargs={"weight_decay": 1e-3}))
    ckpt = torch.load(result["checkpoint"], weights_only=True)
    assert ckpt["fit_kwargs"] == {"weight_decay": 1e-3}
    assert ckpt["model_kwargs"] == {}


@pytest.mark.parametrize("regressor", ["gp", "ridge"])
def test_pod_trains_saves_and_predicts(tmp_path, regressor):
    csv = make_csv(tmp_path / "toy.csv", n_inputs=2, n_outputs=30, n=50)
    result = train(TrainConfig(dataset=csv, model="pod", model_kwargs={"n_modes": 3, "regressor": regressor},
                               device="cpu", output_dir=tmp_path / "run"))
    # Linear toy data: the POD surrogate is essentially exact.
    assert result["test_mae"] < 0.05
    predictor = Predictor(result["checkpoint"])  # buffers of varying size must reload
    assert predictor.predict(np.zeros((4, 2))).shape == (4, 30)


def test_pod_rejects_unknown_regressor():
    spec = DatasetSpec("s", ["a", "b"], 10)
    with pytest.raises(ValueError, match="regressor"):
        build_model("pod", spec, regressor="nope")


def test_extra_test_dataset(tmp_path):
    csv = make_csv(tmp_path / "toy.csv", n=40)
    other = make_csv(tmp_path / "other.csv", n=10, name="other")
    result = train(TrainConfig(dataset=csv, epochs=2, device="cpu", output_dir=tmp_path / "run",
                               test_dataset=other))
    assert np.isfinite(result["independent_mae"])

    bad = make_csv(tmp_path / "bad.csv", n_inputs=2, n=10, name="bad")
    with pytest.raises(ValueError, match="does not match"):
        train(TrainConfig(dataset=csv, epochs=2, device="cpu", output_dir=tmp_path / "run2",
                          test_dataset=bad))
