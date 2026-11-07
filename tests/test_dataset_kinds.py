"""A new dataset kind only needs the three hooks of the abstract Dataset."""

import numpy as np
import pytest

from me_project_speit import config
from me_project_speit.datasets import (
    Dataset,
    DatasetSpec,
    available_kinds,
    get_dataset_class,
    load_dataset,
    register_dataset,
)
from me_project_speit.models import IncompatibleError, SurrogateModel, build_model, register_model

N = 6  # rows per simulation of the toy kind


@register_dataset("toy_series")
class ToySeriesDataset(Dataset):
    """Row per (simulation, step): [p, t, value]; the 'simulator' is a formula."""

    @classmethod
    def build_spec(cls, input_names=("P0",), name="toy"):
        names = cls.check_params(input_names)
        return DatasetSpec(name, names + ["t"], 1, mesh_id="toy", kind=cls.kind,
                           input_ranges=cls.ranges(names), rows_per_sample=N)

    @classmethod
    def simulation_rows(cls, spec, params):
        p = params["P0"]
        t = np.arange(N, dtype=float)
        return np.column_stack([np.full(N, p), t, p * t])

    def fields(self):
        return self.Y.reshape(len(self), N)


@register_model("toy_series_model")
class ToySeriesModel(SurrogateModel):
    supported_datasets = (ToySeriesDataset,)

    def __init__(self, n_inputs, n_outputs):
        super().__init__()


def test_new_kind_is_registered_and_listed():
    assert "toy_series" in available_kinds()
    assert get_dataset_class("toy_series") is ToySeriesDataset


def test_generate_save_load_roundtrip(tmp_path):
    ds = ToySeriesDataset.generate(8, ("P0",), name="toy", seed=0, verbose=False)
    assert len(ds) == 8 and ds.X.shape == (8 * N, 2)
    path = ds.save(tmp_path / "toy.csv")
    loaded = load_dataset(path)  # dispatched on spec.kind, no extra code
    assert isinstance(loaded, ToySeriesDataset)
    np.testing.assert_allclose(loaded.Y, ds.Y, rtol=1e-5)
    assert loaded.fields().shape == (8, N)


def test_split_is_per_simulation():
    ds = ToySeriesDataset.generate(10, ("P0",), seed=1, verbose=False)
    s = ds.split(seed=0)
    for idx in (s.train, s.val, s.test):
        assert len(set(idx // N)) * N == len(idx)


def test_wrong_row_shape_is_reported():
    class Bad(ToySeriesDataset):
        @classmethod
        def simulation_rows(cls, spec, params):
            return np.zeros((2, 2))

    with pytest.raises(ValueError, match="simulation_rows returned shape"):
        Bad.generate(1, ("P0",), verbose=False)


def test_model_compatibility_follows_dataset_classes():
    toy = ToySeriesDataset.build_spec(("P0",))
    build_model("toy_series_model", toy)
    with pytest.raises(IncompatibleError, match="ToySeriesDataset"):
        build_model("toy_series_model", FieldDatasetSpec())
    with pytest.raises(IncompatibleError):
        build_model("mlp", toy)


def FieldDatasetSpec():
    return DatasetSpec("f", ["a"], config.N_NODES, mesh_id=config.MESH_ID)


def test_old_layout_key_in_json_is_still_read():
    spec = DatasetSpec.from_dict({"name": "o", "input_names": ["a"], "n_outputs": 1,
                                  "layout": "toy_series", "rows_per_sample": N})
    assert spec.kind == "toy_series"
