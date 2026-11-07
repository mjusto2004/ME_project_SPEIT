"""Dataset specs and model/dataset compatibility checks."""

import json

import numpy as np
import pytest

from me_project_speit import config
from me_project_speit.datasets import (
    DatasetSpec,
    FieldDataset,
    list_datasets,
    load_spec,
    resolve_dataset,
    save_spec,
)
from me_project_speit.datasets.field import legacy_spec
from me_project_speit.models import (
    IncompatibleError,
    SurrogateModel,
    available_models,
    build_model,
    compatible_models,
    register_model,
)

from me_project_speit.training import TrainConfig, train

LEGACY_SPEC = legacy_spec()


def make_spec(names, name="d"):
    return FieldDataset.build_spec(names, name=name)


@register_model("fixed_mesh_test")
class FixedMeshModel(SurrogateModel):
    """Only valid for the reference mesh."""

    def __init__(self, n_inputs, n_outputs):
        super().__init__()

    @classmethod
    def check_compatible(cls, spec):
        if spec.mesh_id != config.MESH_ID:
            raise IncompatibleError(f"needs mesh '{config.MESH_ID}', dataset has '{spec.mesh_id}'")


def test_spec_json_roundtrip(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("0,0\n")
    spec = make_spec(("T0", "h"), name="d")
    save_spec(spec, csv)
    assert load_spec(csv) == spec


def test_legacy_csv_without_sidecar(tmp_path):
    csv = tmp_path / "old.csv"
    np.savetxt(csv, np.zeros((3, LEGACY_SPEC.n_columns)), delimiter=",")
    assert load_spec(csv) == LEGACY_SPEC


def test_unknown_shape_without_sidecar_is_rejected(tmp_path):
    csv = tmp_path / "mystery.csv"
    np.savetxt(csv, np.zeros((3, 10)), delimiter=",")
    with pytest.raises(ValueError, match="no description file"):
        load_spec(csv)


def test_make_spec_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="Unknown input"):
        make_spec(("T0", "bogus"))


def test_incompatible_pair_fails_fast_and_names_alternatives():
    other_mesh = DatasetSpec("fine", ["T0"], 5000, mesh_id="cube_5000")
    with pytest.raises(IncompatibleError) as err:
        build_model("fixed_mesh_test", other_mesh)
    assert "cube_5000" in str(err.value)
    assert "mlp" in str(err.value)  # suggested alternative


def test_compatible_models_lists_only_valid_ones():
    other_mesh = DatasetSpec("fine", ["T0"], 5000, mesh_id="cube_5000")
    assert "mlp" in compatible_models(other_mesh)
    assert "fixed_mesh_test" not in compatible_models(other_mesh)
    assert "fixed_mesh_test" in compatible_models(LEGACY_SPEC)
    assert "fixed_mesh_test" in available_models()


def test_train_rejects_incompatible_before_loading_data(tmp_path):
    csv = tmp_path / "fine.csv"
    csv.write_text("1,2,3,4\n" * 5)
    save_spec(DatasetSpec("fine", ["a"], 3, mesh_id="other"), csv)
    with pytest.raises(IncompatibleError):
        train(TrainConfig(dataset=csv, model="fixed_mesh_test", device="cpu",
                          output_dir=tmp_path / "run"))
    assert not (tmp_path / "run").exists()  # nothing was started


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nope", LEGACY_SPEC)


def test_list_and_resolve_datasets(tmp_path):
    (tmp_path / "ok.csv").write_text("0\n")
    save_spec(DatasetSpec("ok", ["a"], 1), tmp_path / "ok.csv")
    (tmp_path / "bad.csv").write_text("1,2,3\n")  # no sidecar, unknown width
    entries = {p.stem: (s, e) for p, s, e in list_datasets(tmp_path)}
    assert entries["ok"][0].n_columns == 2
    assert entries["bad"][0] is None and entries["bad"][1]
    assert resolve_dataset("ok", tmp_path) == tmp_path / "ok.csv"
    with pytest.raises(FileNotFoundError, match="Available"):
        resolve_dataset("missing", tmp_path)
    assert json.loads((tmp_path / "ok.json").read_text())["n_outputs"] == 1
