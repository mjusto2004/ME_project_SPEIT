# 2. Add a simulation parameter

Goal: be able to vary a new physical quantity in the simulator, e.g. the diameter of the heat source, and sample it with `generate --params`.

Example used below: **`D`**, the source diameter (m).

## Step 1 — declare it in `simulations/config.py`

Each simulation has one dictionary of parameters, `<SIMULATION>_PARAMS`, mapping a name to a `Param(default, range)`. `PARAMS` holds the ones common to all simulations; `CUBE_PARAMS` (kinds `field` and `pointwise`) and `SPONGE_PARAMS` (kind `graph`) extend it with `**`. `generate` only accepts the names that have a range. For a parameter of the cube only:

```python
CUBE_PARAMS = {
    **PARAMS,
    "res": Param(MESH_RESOLUTION_DOMAIN),
    "cylinder_res": Param(MESH_RESOLUTION_SOURCE),
    "D": Param(1 / 3, (0.2, 0.5)),   # default, then the sampling range used by generate
}
```

A parameter without a range (like `res`) can still be passed by hand to `run`, but cannot be varied by `generate`. A parameter useful to every simulation goes in `PARAMS` instead.

The default is used whenever a dataset does not vary the parameter, so existing datasets stay reproducible.

## Step 2 — use it in `simulations/cube.py`

`CubeSimulation.run(**params)` merges the defaults of `CUBE_PARAMS` with what it receives (`resolve_params`), then reads each value. Read the new one and use it where the constant was used:

```python
p = cls.resolve_params(params)
t0, flux, h, diameter = p["T0"], p["Phi0"], p["h"], p["D"]
```

Then replace `config.SOURCE_DIAMETER` by `diameter` in the heat-flux formula. If the quantity changes the **geometry or mesh** (as a diameter does), also pass it to `cls.build_mesh(...)`.

> ⚠ **Mesh changes are the dangerous case.** A different diameter generally produces a different number of nodes, which breaks the fixed-size `field` dataset type (`n_outputs = N_NODES`). Prefer parameters that do not touch the mesh (material properties, loads, boundary conditions, initial temperature). If you must change the mesh, give those datasets a new `mesh_id`, or use `pointwise`, and see [tutorial 4](04-add-a-dataset-type.md).
>
> The `graph` type is the one made for meshes that change from one simulation to the next: its parameters live in `SPONGE_PARAMS` (`simulations/config.py`) and its simulation in `simulations/sponge.py` (`SpongeSimulation`). `field` and `pointwise` only accept the parameters of `CUBE_PARAMS` and refuse e.g. `r` or `res`.

## Step 3 — generate and check

```bash
uv run --extra sim me-project-speit generate -n 5 --params Phi0 D
uv run me-project-speit list datasets
```

The new dataset's JSON lists `input_names: ["Phi0", "D"]` and `input_ranges` for both. Parameters you did not list keep their defaults.

## Step 4 — add a test

`CubeSimulation.run` needs EasyFEA, so keep tests small, e.g. that an unknown name is rejected:

```python
import pytest
from me_project_speit.datasets import FieldDataset

def test_new_parameter_is_known():
    assert FieldDataset.build_spec(("Phi0", "D")).input_names == ["Phi0", "D"]

def test_unknown_parameter_rejected():
    with pytest.raises(ValueError, match="Unknown input"):
        FieldDataset.build_spec(("Phi0", "bogus"))
```

## Things to remember

- Models need no change: they are built from the dataset's number of inputs.
- `mlp` trained on `[Phi0, h]` cannot be fed `[T0, Phi0, h]` later; `Predictor.predict` checks the input count and tells you which inputs the model expects.
