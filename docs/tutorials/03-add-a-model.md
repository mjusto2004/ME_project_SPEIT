# 3. Add a model

Goal: add a new architecture that can be selected with `--model` and is automatically checked against datasets.

Example used below: a residual MLP, **`res_mlp`**, for `field` datasets.

## Step 1 — create `models/res_mlp.py`

```python
"""Residual MLP for field datasets."""

import torch
from torch import nn

from ..datasets import FieldDataset
from .base import IncompatibleError, SurrogateModel
from .registry import register_model


@register_model("res_mlp")                  # the name used by --model
class ResMLP(SurrogateModel):
    supported_datasets = (FieldDataset,)    # dataset types (classes) it can learn from

    def __init__(self, n_inputs: int, n_outputs: int, width: int = 256, depth: int = 3):
        super().__init__()
        self.inp = nn.Linear(n_inputs, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.ReLU(), nn.Linear(width, width)) for _ in range(depth)
        )
        self.out = nn.Linear(width, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(torch.relu(h))
```

Rules every model must follow:

- Subclass `SurrogateModel` and decorate with `@register_model("unique_name")` (a duplicate name raises an error).
- The constructor starts with `(n_inputs, n_outputs, ...)`. Both are filled from the dataset spec, so the model adapts to any shape. Extra arguments need **defaults** and become `model_kwargs`.
- `forward` maps `(batch, n_inputs)` to `(batch, n_outputs)`.
- Arguments must be plain values (numbers, strings, lists) so they can be saved in the checkpoint. Use lists, not tuples, if you pass them from the CLI.

## Step 2 — register it

Add the import in `models/__init__.py` so the decorator runs:

```python
from .mlp import MLP, CoordMLP  # noqa: E402
from .res_mlp import ResMLP     # noqa: E402
```

(and add `"ResMLP"` to `__all__` if you want it exported).

## Step 3 — say what it accepts

`supported_datasets` is enough when only the dataset type matters (subclasses of the listed classes are accepted too). When the model has extra requirements, override `check_compatible` and **call `super()` first**:

```python
@classmethod
def check_compatible(cls, spec):
    super().check_compatible(spec)               # dataset-type check
    if spec.mesh_id != "cube_560":
        raise IncompatibleError("needs mesh 'cube_560'")
    if spec.n_inputs < 2:
        raise IncompatibleError("needs at least 2 inputs")
```

The message you give is shown to the user as the reason, followed automatically by the list of compatible models.

Typical reasons to restrict a model:

| Model property | Check |
|----------------|-------|
| tied to one output size or mesh | `spec.mesh_id`, `spec.n_outputs` |
| needs coordinates as inputs | `spec.input_names[-3:] == ["x","y","z"]` |
| predicts one value per row | `spec.n_outputs == 1` |
| works on one dataset type | `supported_datasets` |

Leave `supported_datasets = None` only for a model that is truly agnostic to the dataset type.

## Step 4 — try it

```bash
uv run me-project-speit list models                      # res_mlp is listed
uv run me-project-speit list datasets                    # res_mlp appears under compatible field datasets
uv run me-project-speit train --dataset data_test --model res_mlp --epochs 20
```

`--hidden` only applies to `mlp`/`coord_mlp` (it is passed as `hidden=`). Other keyword arguments can be set from Python:

```python
from me_project_speit.training import TrainConfig, train
train(TrainConfig(dataset="data_test", model="res_mlp",
                  model_kwargs={"width": 128, "depth": 4}))
```

> To expose a model-specific option on the command line, add an argument to the `train` sub-parser in `cli.py` and put it into `model_kwargs` where `TrainConfig` is built.

The checkpoint stores `model_name` and `model_kwargs`, so `Predictor("runs/.../best_model.pt")` rebuilds a `ResMLP` with the right sizes by itself.

## Step 5 — add tests

```python
import pytest
from me_project_speit.datasets import DatasetSpec
from me_project_speit.models import IncompatibleError, build_model

def test_res_mlp_shapes():
    import torch
    spec = DatasetSpec("s", ["a", "b"], 30)
    assert build_model("res_mlp", spec)(torch.randn(4, 2)).shape == (4, 30)

def test_res_mlp_rejects_pointwise():
    spec = DatasetSpec("p", ["a", "x", "y", "z"], 1, kind="pointwise", rows_per_sample=10)
    with pytest.raises(IncompatibleError):
        build_model("res_mlp", spec)
```

Look at `tests/test_builtin_kinds.py` and `tests/test_pipeline.py` for the training and prediction round trips (`train(...)` then `Predictor(...)`), which need no EasyFEA.
