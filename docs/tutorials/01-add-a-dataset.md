# 1. Add a dataset

Goal: make a new CSV usable with `train`, `list datasets` and `show`, **without writing project code**. The dataset must be of an existing type (`field` or `pointwise`, see the README). For a new row format, see [tutorial 4](04-add-a-dataset-type.md).

## A. Generate it with the project

```bash
uv run --extra sim me-project-speit generate -n 200 --params T0 P0 h --seed 1
uv run --extra sim me-project-speit generate -n 200 --params P0 h --kind pointwise
```

This writes `data/dataset_<timestamp>_<n>.csv` and the matching `.json`. Rename **both** files together (and update `"name"` in the JSON) to keep a nice name. Generated files are git-ignored (`data/dataset_*`); a renamed file is not, so it will be committed.

## B. Bring data from somewhere else

You need two files with the same stem in `data/`:

### The CSV

No header, comma-separated, one row per sample, all numbers. Columns are the inputs first, then the outputs:

| Type (`kind`) | One row is... | Columns |
|--------|---------------|---------|
| `field` | one simulation | `p1 … pk, T_node_0 … T_node_{m-1}` |
| `pointwise` | one node of one simulation | `p1 … pk, x, y, z, T` |

For `pointwise`, the rows of one simulation must be **contiguous** and every simulation must have the same number of rows (`rows_per_sample`).

> `graph` datasets are different: they have an extra leading `sim` column, a variable number of rows per simulation, and a third file `name_edges.npz` with the connections. They are made by `generate --kind graph`; see the README (section *The `graph` type*) before building one by hand.

### The JSON (`DatasetSpec`)

```json
{
  "name": "my_data",
  "input_names": ["P0", "h"],
  "n_outputs": 560,
  "output_kind": "nodal_temperature",
  "mesh_id": "cube_560",
  "input_ranges": {"P0": [15000, 30000], "h": [150, 300]},
  "kind": "field",
  "rows_per_sample": 1
}
```

| Key | Meaning |
|-----|---------|
| `input_names` | one name per input column, in column order. For `pointwise`, end with `"x", "y", "z"`. |
| `n_outputs` | output columns per row (`field`: number of nodes; `pointwise`: `1`). |
| `mesh_id` | identifies the mesh the outputs live on. **Datasets with different node sets must have different ids.** |
| `kind` | dataset type: `field` or `pointwise` (see `me-project-speit generate --help`). |
| `rows_per_sample` | `1` for `field`; number of nodes for `pointwise`. |
| `input_ranges` | optional; only used when `generate` samples inputs. |

`n_columns` must equal `len(input_names) + n_outputs`, otherwise loading fails with a message showing both numbers.

### Writing both files from Python

```python
import numpy as np
from me_project_speit.datasets import DatasetSpec, save_spec

X = ...  # (n_sim, k)  inputs
Y = ...  # (n_sim, m)  fields

np.savetxt("data/my_data.csv", np.hstack([X, Y]), delimiter=",")
save_spec(
    DatasetSpec(name="my_data", input_names=["P0", "h"], n_outputs=Y.shape[1],
                mesh_id="my_mesh"),
    "data/my_data.csv",            # writes data/my_data.json
)
```

## Check it

```bash
uv run me-project-speit list datasets
```

Your dataset must appear with its type and a non-empty list of compatible models. Then:

```bash
uv run me-project-speit train --dataset my_data --epochs 50
```

## Common mistakes

- **No JSON** → `has N columns and no description file`. Only the legacy 563-column format is accepted without one.
- **Wrong `rows_per_sample`** → `rows is not a multiple of rows_per_sample`.
- **Same `mesh_id` for different meshes** → no error, but a mesh-specific model will silently be considered compatible. Use distinct ids.
- **Fewer than 3 simulations** is rejected; below 50 you get a warning that evaluation is unreliable.
