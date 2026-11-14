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

    def fit(self, train_loader, val_loader, **kwargs):
        return self._fit_loop(train_loader, val_loader, **kwargs)  # standard loop
```

Rules every model must follow:

- Subclass `SurrogateModel` and decorate with `@register_model("unique_name")` (a duplicate name raises an error).
- The constructor starts with `(n_inputs, n_outputs, ...)`. Both are filled from the dataset spec, so the model adapts to any shape. Extra arguments need **defaults** and become `model_kwargs`.
- `SurrogateModel` is abstract: you **must** implement `forward` and `fit`, or the model cannot be instantiated.
- `forward` maps `(batch, n_inputs)` to `(batch, n_outputs)`.
- `fit(train_loader, val_loader, *, epochs, lr, patience, device, y_std, on_epoch, on_best)` trains the model and returns the best epoch. It calls `on_epoch(...)` after each epoch (history log) and `on_best(epoch)` whenever validation improves (checkpoint). For batches shaped `(*inputs, target)` call `self._fit_loop(...)` (Adam + MSE + early stopping); write your own loop for anything special (several optimizers, custom losses...). Data loading, normalisation, checkpoint content and the final test evaluation stay in `training.train`; `model.evaluate(loader, device, y_std)` is available to every model.
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

Options are not on the command line until the model declares them. Override `add_cli_arguments` with plain argparse (it receives the argument group "res_mlp options" of `train`):

```python
@classmethod
def add_cli_arguments(cls, parser):
    parser.add_argument("--width", type=int, help="width of the hidden layers")
    parser.add_argument("--depth", type=int, help="number of residual blocks")
```

```bash
uv run me-project-speit train --dataset data_test --model res_mlp --width 128 --depth 4
```

Rules: the option name must match the constructor argument (`--width` -> `width`; `--n-heads` -> `n_heads`), and its argparse default must stay `None` ("not given": the constructor default applies). Options you do not declare can still be set from Python (`TrainConfig(..., model_kwargs={"width": 128})`). For a conversion the default cannot do, override `kwargs_from_cli(args)`.

**Training options** of the model (not constructor arguments) work the same way with `add_fit_arguments(parser)` and `fit_kwargs_from_cli(args)`. They reach `fit(...)` as extra keyword arguments (`TrainConfig.fit_kwargs`) and are saved in the checkpoint but never passed to the constructor, so `Predictor` is unaffected. The built-in loop `_fit_loop` already accepts `weight_decay`; `mlp` and `gnn` expose it as `--weight-decay`. A custom `fit` just declares the keyword arguments it wants. `--help` shows the options of the model chosen with `--model` (`train --model res_mlp --help`).

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
