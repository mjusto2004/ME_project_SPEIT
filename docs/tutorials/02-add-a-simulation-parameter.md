# 2. Add a simulation parameter

Goal: be able to vary a new physical quantity in the simulator, e.g. the diameter of the heat source, and sample it with `generate --params`.

Example used below: **`D`**, the source diameter (m).

## Step 1 — declare it in `config.py`

`generate` only accepts names found in these two dictionaries:

```python
PARAM_DEFAULTS = {"T0": 293.0, "P0": 30_000.0, "h": 300.0, "D": 1 / 3}
PARAM_RANGES = {
    "T0": (146.5, 293.0),
    "P0": (15_000.0, 30_000.0),
    "h": (150.0, 300.0),
    "D": (0.2, 0.5),          # sampling range for generate
}
```

The default is used whenever a dataset does not vary the parameter, so existing datasets stay reproducible.

## Step 2 — use it in `simulation.py`

`run_simulation(**params)` merges `PARAM_DEFAULTS` with what it receives, then reads each value. Read the new one and use it where the constant was used:

```python
p = {**config.PARAM_DEFAULTS, **params}
t0, power, h, diameter = p["T0"], p["P0"], p["h"], p["D"]
```

Then replace `config.SOURCE_DIAMETER` by `diameter` in the heat-flux formula. If the quantity changes the **geometry or mesh** (as a diameter does), also pass it to `build_mesh(...)`.

> ⚠ **Mesh changes are the dangerous case.** A different diameter generally produces a different number of nodes, which breaks the fixed-size `field` dataset type (`n_outputs = N_NODES`). Prefer parameters that do not touch the mesh (material properties, loads, boundary conditions, initial temperature). If you must change the mesh, give those datasets a new `mesh_id`, or use `pointwise`, and see [tutorial 4](04-add-a-dataset-type.md).
>
> The `graph` type is the one made for meshes that change from one simulation to the next: its parameters live in `SPONGE_PARAM_RANGES` (`config.py`) and its simulation in `sponge.py`. `field` and `pointwise` only accept the parameters of `PARAM_RANGES` and refuse e.g. `r` or `res`.

## Step 3 — generate and check

```bash
uv run --extra sim me-project-speit generate -n 5 --params P0 D
uv run me-project-speit list datasets
```

The new dataset's JSON lists `input_names: ["P0", "D"]` and `input_ranges` for both. Parameters you did not list keep their defaults.

## Step 4 — add a test

`run_simulation` needs EasyFEA, so keep tests small, e.g. that an unknown name is rejected:

```python
import pytest
from me_project_speit.datasets import FieldDataset

def test_new_parameter_is_known():
    assert FieldDataset.build_spec(("P0", "D")).input_names == ["P0", "D"]

def test_unknown_parameter_rejected():
    with pytest.raises(ValueError, match="Unknown input"):
        FieldDataset.build_spec(("P0", "bogus"))
```

## Things to remember

- Models need no change: they are built from the dataset's number of inputs.
- `mlp` trained on `[P0, h]` cannot be fed `[T0, P0, h]` later; `Predictor.predict` checks the input count and tells you which inputs the model expects.
