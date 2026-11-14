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
