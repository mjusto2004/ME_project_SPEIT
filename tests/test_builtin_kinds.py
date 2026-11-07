"""Field vs pointwise dataset kinds and their model compatibility."""

import numpy as np
import pytest

from me_project_speit.datasets import DatasetSpec, load_dataset, save_spec, split_indices
from me_project_speit.inference import Predictor
from me_project_speit.models import IncompatibleError, build_model, compatible_models
from me_project_speit.training import TrainConfig, train

N_NODES = 12


def pointwise_spec(**kw):
    base = dict(
        name="pw", input_names=["P0", "x", "y", "z"], n_outputs=1,
        mesh_id="toy", kind="pointwise", rows_per_sample=N_NODES,
    )
    return DatasetSpec(**{**base, **kw})


def make_pointwise_csv(path, n_sims=20):
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(N_NODES, 3))
    rows = []
    for _ in range(n_sims):
        p = rng.uniform()
        rows.append(np.hstack([np.full((N_NODES, 1), p), coords,
                               (p * coords.sum(axis=1))[:, None]]))
    np.savetxt(path, np.vstack(rows), delimiter=",")
    save_spec(pointwise_spec(), path)
    return path


def test_models_are_filtered_by_dataset_kind():
    field = DatasetSpec("f", ["a", "b"], 50)
    assert compatible_models(field) == ["mlp"]
    assert compatible_models(pointwise_spec()) == ["coord_mlp"]


def test_mlp_rejects_pointwise_and_coord_mlp_rejects_field():
    with pytest.raises(IncompatibleError, match="FieldDataset"):
        build_model("mlp", pointwise_spec())
    with pytest.raises(IncompatibleError, match=r"Compatible models: \['mlp'\]"):
        build_model("coord_mlp", DatasetSpec("f", ["a"], 50))


def test_coord_mlp_requires_xyz_inputs():
    with pytest.raises(IncompatibleError, match="x, y, z"):
        build_model("coord_mlp", pointwise_spec(input_names=["P0", "a", "b", "c"]))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="Unknown dataset kind"):
        DatasetSpec("x", ["a"], 1, kind="weird")


def test_split_keeps_simulations_together():
    s = split_indices(20, seed=0, rows_per_sample=N_NODES)
    for idx in (s.train, s.val, s.test):
        assert len(set(idx // N_NODES)) * N_NODES == len(idx)  # whole simulations only
    sims = [set(i // N_NODES) for i in (s.train, s.val, s.test)]
    assert not (sims[0] & sims[1] or sims[0] & sims[2] or sims[1] & sims[2])


def test_load_pointwise_dataset(tmp_path):
    data = load_dataset(make_pointwise_csv(tmp_path / "pw.csv"))
    assert type(data).__name__ == "PointwiseDataset"
    assert len(data) == 20 and data.X.shape == (20 * N_NODES, 4)
    assert data.fields().shape == (20, N_NODES)


def test_row_count_must_match_rows_per_sample(tmp_path):
    csv = make_pointwise_csv(tmp_path / "pw.csv")
    save_spec(pointwise_spec(rows_per_sample=7), csv)
    with pytest.raises(ValueError, match="multiple"):
        load_dataset(csv)


def test_train_and_predict_pointwise(tmp_path):
    csv = make_pointwise_csv(tmp_path / "pw.csv")
    result = train(TrainConfig(dataset=csv, model="coord_mlp", epochs=3, device="cpu",
                               output_dir=tmp_path / "run"))
    assert np.isfinite(result["test_mae"])
    out = Predictor(result["checkpoint"]).predict([[0.5, 0.1, 0.2, 0.3]])
    assert out.shape == (1, 1)
