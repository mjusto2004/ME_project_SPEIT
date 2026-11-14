# Developer documentation

Step-by-step guides for extending the project. Start with the mental model, then pick the tutorial that matches what you want to add.

## Mental model (2 minutes)

```
 simulations/ ──► datasets/ ──► training.py ──► checkpoint (best_model.pt)
 (EasyFEA)         DatasetSpec     build_model()        │
                   CSV + JSON      check_compatible()   ▼
                                                    inference.py
```

Three ideas keep datasets and models interchangeable:

1. **A dataset describes itself.** `foo.csv` always comes with `foo.json` (a `DatasetSpec`): input names, number of outputs, `mesh_id`, and a **kind** (dataset type) that says what one row means. Each kind is a subclass of the abstract `Dataset`.
2. **A model declares what it accepts.** Each model class has `supported_datasets` (dataset classes) and an optional `check_compatible(spec)` that raises `IncompatibleError`.
3. **Compatibility is checked before any work starts.** `train` calls `check_compatible(model, spec)` first; an invalid pair stops immediately with a message listing the models that do fit.

## Which tutorial do I need?

| I want to... | Tutorial |
|--------------|----------|
| Use data I already have (or generated elsewhere) | [1. Add a dataset](tutorials/01-add-a-dataset.md) |
| Vary a new physical quantity in `generate` (e.g. source diameter) | [2. Add a simulation parameter](tutorials/02-add-a-simulation-parameter.md) |
| Try a new network architecture | [3. Add a model](tutorials/03-add-a-model.md) |
| Create a new *type* of dataset (new row format, e.g. time series): a `Dataset` subclass | [4. Add a dataset type](tutorials/04-add-a-dataset-type.md) |
| Fix an error message or check my change is complete | [5. Checklist and troubleshooting](tutorials/05-checklist-and-troubleshooting.md) |

## Where things live

| File (`src/me_project_speit/`) | Role |
|------|------|
| `simulations/config.py` | physical constants, `PARAMS` (common), `CUBE_PARAMS`, `SPONGE_PARAMS` (default and range of each parameter), `N_NODES`, `MESH_ID`, sponge parameters |
| `simulations/base.py` | abstract `Simulation`: parameter check and the EasyFEA steps shared by the two simulations |
| `simulations/cube.py` | `CubeSimulation` (kinds `field`, `pointwise`): `build_mesh()`, `run(**params)` (needs EasyFEA) |
| `datasets/base.py` | abstract `Dataset`, `DatasetSpec`, `@register_dataset`, `load_dataset`, `save_dataset`, `split_indices` |
| `datasets/field.py`, `pointwise.py`, `graph.py` | the dataset types: `FieldDataset`, `PointwiseDataset`, `GraphDataset` (variable meshes + connections; the advanced example) |
| `simulations/sponge.py` | `SpongeSimulation` (kind `graph`): cube with random spherical holes, `run(centers, face, **params)` |
| `models/base.py` | abstract `SurrogateModel` (children implement `forward` and `fit`; optional `add_cli_arguments` / `add_fit_arguments` expose model and training options to the CLI), `IncompatibleError` |
| `models/registry.py` | `@register_model`, `build_model`, `check_compatible`, `compatible_models` |
| `models/mlp.py` | `MLP` (field) and `CoordMLP` (pointwise) |
| `models/gnn.py` | `GNN` (graph) |
| `utils/` | generic helpers (leaf package, imports nothing from the project): `parallel_map`, `select_device`, `timestamp`, `latest_run`, `to_plain` |
| `training.py` | `TrainConfig`, `train()` |
| `inference.py` | `Predictor` |
| `cli.py` | the `me-project-speit` command |

Run the tests after every change: `uv run --extra dev pytest`.
