# 4. Add a dataset type

A **dataset type** (also called *kind*) defines what one block of CSV rows means. Each type is a small class that inherits from the abstract `Dataset`:

```
Dataset (abstract, in datasets/base.py)          ← reading, writing, generating, splitting: already done
 ├── FieldDataset      "field"      one row per simulation: params → whole field
 ├── PointwiseDataset  "pointwise"  one row per (simulation, node): params, x, y, z → value
 └── YourDataset       "your_kind"  ← you write three methods
```

Create a new type when the **format** of the data changes, not just its size. Shapes can lie: the example below has 3 inputs and 560 outputs, exactly like a `field` dataset, but a different meaning. Models declare which dataset classes they accept, so `mlp` is refused on it automatically.

## What a dataset type must provide

Implement three methods; everything else is inherited.

| Method | Purpose |
|--------|---------|
| `build_spec(cls, input_names, name)` | describes the data: inputs, `n_outputs`, `rows_per_sample`, `mesh_id` |
| `simulation_rows(cls, spec, params)` | the CSV rows produced by **one** simulation (shape `(rows_per_sample, n_columns)`; inputs first, then outputs) |
| `fields(self)` | the outputs as `(n_simulations, n_nodes)`, used by `show` |

> Samples that are not independent rows (like the graphs of `datasets/graph.py`, which have a different size each) need more: override `generate`, `from_csv`, `save`, `split` and `make_loader` (how a training batch is built). `graph.py` is a complete, commented example.

Inherited for free: `generate(n, ...)`, `save(path)`, `load` (dispatched on the `kind` in the JSON), `split(seed)` per simulation, `len()` (number of simulations), validation of column and row counts, and the compatibility checks with models.

## Worked example: `snapshots`

We want the temperature field at several times: one row per (simulation, time step), `params, t` → field of 560 values.

> This example was applied to a scratch copy of the project and run end to end (generate → list → train on GPU, plus the rejection of `mlp`). Use it as a template.

### Step 1 — only if the simulator must return something new

The simulator only returned the final state. Add an opt-in flag so existing callers are unaffected.

`simulations/config.py`:

```python
N_SNAPSHOTS = 10   # time steps saved per simulation
```

`simulations/cube.py`, in `CubeSimulation`:

```python
@classmethod
def run(cls, verbose: bool = False, snapshots: bool = False, **params: float):
    ...
    every = config.N_STEPS // config.N_SNAPSHOTS
    history, times, step = [], [], 0
    t = -dt
    while t < config.T_MAX:
        t += dt
        step += 1
        ...
        simu.Save_Iter()
        if snapshots and step % every == 0:
            times.append(step * dt)
            history.append(simu.thermal - 273.15)
    if snapshots:
        return np.array(times), np.array(history)   # (S,), (S, n_nodes)
    return simu.thermal - 273.15
```

(If your type only reshapes data that `CubeSimulation.run` already returns, skip this step, as `PointwiseDataset` does.)

### Step 2 — write the class (`datasets/snapshots.py`)

```python
"""``snapshots`` datasets: one row per (simulation, saved time step)."""

import numpy as np

from ..simulations import config
from .base import Dataset, DatasetSpec, register_dataset


@register_dataset("snapshots")                       # the name stored in the JSON, used by --kind
class SnapshotsDataset(Dataset):
    """Row: ``[params..., t, T_node_0, ..., T_node_{m-1}]``."""

    @classmethod
    def build_spec(cls, input_names=("T0", "P0", "h"), name="cube"):
        names = cls.check_params(input_names)        # validates against cls.param_ranges (config.PARAM_RANGES by default)
        return DatasetSpec(
            name=name,
            input_names=names + ["t"],               # the time is an input
            n_outputs=config.N_NODES,
            mesh_id=config.MESH_ID,
            input_ranges=cls.ranges(names),          # only the simulation parameters, not "t"
            kind=cls.kind,
            rows_per_sample=config.N_SNAPSHOTS,      # rows per simulation → per-simulation splits
        )

    @classmethod
    def simulation_rows(cls, spec, params):
        from ..simulations import CubeSimulation

        times, history = CubeSimulation.run(snapshots=True, **params)   # (S,), (S, n_nodes)
        values = np.array([params[k] for k in spec.input_ranges])
        return np.hstack([np.tile(values, (len(times), 1)), times[:, None], history])

    def fields(self):
        return self.Y                                # already (rows, n_nodes)
```

`rows_per_sample` is what keeps all rows of one simulation in the same train/validation/test subset, and what `load` uses to check the row count. The rows of one simulation must be **contiguous**; `generate` checks that `simulation_rows` returns exactly `(rows_per_sample, n_columns)` and raises a clear error otherwise.

### Step 3 — register it by importing it (`datasets/__init__.py`)

```python
from .snapshots import SnapshotsDataset  # noqa: E402
```

That is all for the dataset: `generate --kind snapshots` now exists (the choices come from the registry), `list datasets` shows the type, and the JSON written next to the CSV says `"kind": "snapshots"`, so loading picks your class by itself.

### Step 4 — a model that can learn from it (`models/time_mlp.py`)

Models state which dataset **classes** they accept (subclasses are accepted too):

```python
from ..datasets.snapshots import SnapshotsDataset
from .base import IncompatibleError
from .mlp import MLP
from .registry import register_model


@register_model("time_mlp")
class TimeMLP(MLP):                                  # MLP already adapts to (n_inputs, n_outputs)
    supported_datasets = (SnapshotsDataset,)

    @classmethod
    def check_compatible(cls, spec):                 # optional extra requirements
        super().check_compatible(spec)               # dataset-type check
        if spec.input_names[-1] != "t":
            raise IncompatibleError("the last input must be the time 't'")
```

and import it in `models/__init__.py`. More about models in [tutorial 3](03-add-a-model.md). If an existing model already works for your type, just add your class to its `supported_datasets` instead.

### Step 5 — check everything

```bash
uv run --extra sim me-project-speit generate -n 12 --kind snapshots --params P0 h
uv run me-project-speit list datasets
# dataset_...  [snapshots] 3 inputs (P0, h, t) -> 560 outputs ...  compatible models: ['time_mlp']
uv run me-project-speit train --dataset dataset_... --model time_mlp --epochs 20
uv run me-project-speit train --dataset dataset_... --model mlp
# error: Model 'mlp' is incompatible ... needs a dataset of type ['FieldDataset'], got SnapshotsDataset (kind 'snapshots').
#        Compatible models: ['time_mlp']
```

### Step 6 — test it

`tests/test_dataset_kinds.py` shows the pattern with a toy type whose "simulator" is a formula, so it needs no EasyFEA. Copy it and check the round trip (`generate` → `save` → `load_dataset`), the per-simulation split, and the model compatibility.

## Checklist

1. Subclass `Dataset`, decorate with `@register_dataset("name")`, implement `build_spec`, `simulation_rows`, `fields`.
2. Import the module in `datasets/__init__.py`.
3. Set `supported_datasets` on every model that can learn from the type (and write one if none can).
4. Test, and add a row to the *Dataset types* table of the README.

## Pitfalls

- **`simulation_rows` runs in worker processes** (`generate --workers N`). It must only depend on its arguments (`spec`, `params`): no global variable filled in the main process, and the class must be defined at the top level of an importable module of the package (a class defined inside a test file or a notebook cannot be sent to the workers: use `workers=1` there). Draw random numbers in `generate`, never inside `simulation_rows`, so the dataset does not depend on the number of workers (the `graph` type shows how: one seed per simulation).

- **Same format, different data** (e.g. a second `field` dataset on a finer mesh): do not add a type, give the data a different `mesh_id` and make mesh-specific models check it.
- **Non-contiguous rows, or a varying number of rows per simulation**, break the per-simulation split. Keep `rows_per_sample` constant.
- **Splitting by row leaks information** between train and test. Always use `dataset.split(seed)`.
- **Old files**: JSONs written with the former key `"layout"` are still read as `kind`.
