# ME_project_SPEIT

Deep-learning surrogate model that predicts **transient 3D temperature fields**.

A unit aluminium cube is heated by a cylindrical source along its z axis and cooled by convection on its six faces. A finite-element simulation ([EasyFEA](https://github.com/matnoel/EasyFEA)) gives the temperature at the 560 mesh nodes after 10 000 s. A multi-layer perceptron is trained to reproduce this field directly from three scalar inputs:

| Input | Meaning | Sampled range |
|-------|---------|---------------|
| `T0`  | initial temperature (K) | 293 (fixed) |
| `Phi0` | heat flux (W.m-2) | 500 – 1500 |
| `h`   | convection coefficient (W·m⁻²·K⁻¹) | 150 (fixed) |

Once trained, the network replaces a ~0.5 s simulation with a single forward pass.

## Project layout

```
.
├── pyproject.toml            package metadata, dependencies, CLI entry point
├── data/                     datasets (CSV) and mesh coordinates
│   ├── data_test.csv         90 sample simulations
│   ├── data_test.json        description of the dataset (inputs / outputs)
│   └── coord.npy             node coordinates of the reference mesh
├── resources/                exploratory notebooks, original scripts, Slurm job
├── src/me_project_speit/
│   ├── simulations/          the finite-element simulations (need EasyFEA)
│   │   ├── config.py         physical constants, parameters and shared dimensions
│   │   ├── cube.py           CubeSimulation: cube with a cylindrical source (kinds field, pointwise)
│   │   └── sponge.py         SpongeSimulation: cube with random spherical holes (kind graph)
│   ├── datasets/             abstract Dataset + one subclass per dataset type (field.py, pointwise.py, graph.py)
│   ├── models/               model registry + compatibility (base.py, registry.py, mlp.py, gnn.py)
│   ├── training.py           training loop, early stopping, evaluation
│   ├── inference.py          Predictor: load a checkpoint and predict
│   ├── visualization.py      3D plots of stored fields (needs EasyFEA)
│   └── cli.py                `me-project-speit` command-line interface
├── docs/                     developer tutorials (add a model, dataset, dataset type...)
└── tests/                    pytest suite
```

## Installation

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra sim --extra dev
```

or with pip:

```bash
pip install -e ".[sim,dev]"
```

**GPU:** `pyproject.toml` installs the CUDA 13.2 build of PyTorch (from `download.pytorch.org/whl/cu132`) on Windows, needed for the RTX 5070 development PC; PyPI only ships a CPU build for Windows. Linux machines (e.g. the Slurm cluster) are unaffected and get PyPI's default build. If you are on another Windows machine with a different GPU or an older driver, edit the `[tool.uv.sources]` / `[[tool.uv.index]]` section (e.g. `cu128`). Check with `uv run python -c "import torch; print(torch.cuda.is_available())"`.

Extras:

- *(none)* – `numpy` and `torch`: enough to **train** on an existing dataset.
- `sim` – adds `EasyFEA` (and gmsh): needed to **generate** data and **plot** fields.
- `plot` – adds `matplotlib`: needed by `me-project-speit plot`.
- `dev` – adds `pytest`.

## Usage

The package installs a `me-project-speit` command (also available as `python -m me_project_speit`).

### 1. Generate a dataset

```bash
me-project-speit generate -n 90 --output-dir data --seed 0
me-project-speit generate -n 90 --params T0 h      # vary only T0 and h (Phi0 keeps its default)
me-project-speit generate -n 90 --kind pointwise   # one row per node, coordinates as inputs
```

Runs `n` simulations, **in parallel** (`--workers N`, default: the number of cores up to 8; `--workers 1` runs in the current process). The dataset is the same whatever the number of workers, so a given `--seed` always gives the same file. Typical gain for 40 simulations on 8 workers: 3.8x for `graph`, 2.8x for `pointwise` (start-up of the worker processes costs a few seconds, so the gain grows with `n`). Each simulation uses random inputs (`--params`, default `T0 Phi0 h`; ranges and defaults in `simulations/config.py`) and writes `data/dataset_<timestamp>_<n>.csv` **and** `data/dataset_<timestamp>_<n>.json`. Each CSV row holds the inputs followed by the 560 nodal temperatures in °C.

### 2. Train

```bash
me-project-speit train --dataset data_test --model mlp --epochs 300 --device auto
```

`--dataset` is a CSV path or the bare name of a CSV in `--data-dir` (default `data/`).

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | *required* | CSV path, or name of a CSV in `--data-dir` |
| `--model` | `mlp` | registered model (see `list models`) |
| `--hidden` | `128 256 256` | hidden layer sizes (`mlp`) |
| `--epochs` | 300 | maximum number of epochs |
| `--batch-size` | 64 | mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 30 | early-stopping patience (epochs) |
| `--seed` | 42 | seed for the split and initialisation |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--output-dir` | `runs/<dataset>_<timestamp>` | where to write results |

The data is split 70/15/15 (train/validation/test). Inputs and outputs are standardised using training-set statistics only. The best model on validation MSE is kept and finally scored on the test set (MAE and RMSE in °C). The run directory contains:

- `best_model.pt` – weights, model name and arguments, dataset spec, normalisation statistics, split indices, dataset path;
- `history.csv` – per-epoch train/validation metrics.

### 3. Visualise simulations

```bash
me-project-speit show --dataset data/data_test.csv --ids 0 1
```

Opens an interactive 3D plot per simulation, with a shared colour scale.

### 4. Plot a training run

```bash
uv sync --extra plot                              # installs matplotlib (once)
uv run me-project-speit plot                      # the latest run in runs/
uv run me-project-speit plot --run runs/<folder> --sample 3 --half
uv run me-project-speit plot --save figures/run1  # write PNG files instead of opening windows
```

Two figures: the **training curves** (loss and validation MAE per epoch, with the best epoch marked) for any model, and, for `graph` runs, the **true temperature, the prediction and the absolute error** on the nodes of one test mesh (`--sample` picks which; `--half` keeps only y <= 0.5 to see inside the cube), with a predicted-vs-true plot and the error histogram over all test simulations. The 3D views open in a window you can rotate.

### 5. List models and datasets

```bash
me-project-speit list models
me-project-speit list datasets      # shape of each dataset + which models can use it
```

### Python API

```python
from me_project_speit.training import TrainConfig, train
from me_project_speit.inference import Predictor

result = train(TrainConfig(dataset="data_test", epochs=100, device="cpu"))

predictor = Predictor(result["checkpoint"])
print(predictor.spec.input_names)                 # ['T0', 'Phi0', 'h']
field = predictor.predict([250.0, 25_000.0, 200.0])  # (1, 560) in °C
```

## Datasets, models and compatibility

Different datasets can have different inputs and outputs, and a model only makes sense for some of them. To catch mismatches early, **every dataset describes itself** and **every model declares what it accepts**:

- Each `foo.csv` has a `foo.json` next to it (a `DatasetSpec`): `input_names`, `n_outputs`, `output_kind`, `mesh_id`, `input_ranges`. `generate` writes it automatically. A CSV without a JSON is only accepted if it has the legacy 563-column format (`T0, Phi0, h` → 560 nodes); otherwise loading fails with a message asking for the JSON.
- Models are built *from* the dataset spec (`n_inputs`, `n_outputs`), so the `mlp` adapts to any shape.
- Before training starts, `train` checks the pair. An incompatible pair fails immediately and suggests alternatives, e.g.:

  ```
  error: Model 'fixed_mesh_test' is incompatible with dataset 'fine' (1 inputs (T0) -> 5000 outputs ...):
  needs mesh 'cube_560', dataset has 'cube_5000'. Compatible models: ['mlp']
  ```
- Checkpoints store the model name, its arguments and the dataset spec, so `Predictor` rebuilds the right model and rejects inputs of the wrong size.
- Outputs defined on different meshes must use different `mesh_id`s; two datasets with the same column count but a different `mesh_id` are therefore not interchangeable for mesh-specific models.

### Dataset types

Each dataset type is a subclass of the abstract `Dataset` (`src/me_project_speit/datasets/`). The type is stored as `"kind"` in the JSON, says what a row of the CSV means, and decides which models can learn from it (`list datasets` shows it in brackets):

| Type (`kind`) | Class | Row | Inputs → output | Models |
|---------------|-------|-----|-----------------|--------|
| `field` (default) | `FieldDataset` | one per simulation | `T0, Phi0, h` → all 560 node temperatures | `mlp` |
| `pointwise` | `PointwiseDataset` | one per (simulation, node) | `T0, Phi0, h, x, y, z` → temperature at that node | `coord_mlp` |
| `graph` | `GraphDataset` | one per node; every simulation has its own mesh | `T0, Phi0, h, r, res, x, y, z, heated` + the node connections → temperature at that node | `gnn` |

A `pointwise` dataset has `n_simulations × 560` rows, with the rows of one simulation contiguous (`rows_per_sample` in the JSON). Train/validation/test are split **per simulation**, never per row, so the nodes of a test simulation never leak into training. Each model lists the dataset classes it accepts (`supported_datasets`); training `mlp` on a pointwise dataset (or `coord_mlp` on a field one) stops immediately with an error listing the models that do fit.

A new type is a subclass implementing three methods (`build_spec`, `simulation_rows`, `fields`); loading, saving, generation and splitting are inherited. See [tutorial 4](docs/tutorials/04-add-a-dataset-type.md).

#### The `graph` type: random sponges with node connections

Each simulation is a unit cube from which 1 to 6 random spherical holes (radius `r`, same for all holes of a simulation) are removed, so **every simulation has its own mesh** (about 300 to 1600 nodes with `res` between 5 and 10). Heat enters through one random face among the six; convection acts on every boundary, hole walls included. The sample also records which nodes are connected: two nodes are linked when they belong to the same tetrahedron (both directions).

```bash
uv run --extra sim me-project-speit generate -n 200 --kind graph --seed 1   # about 2 minutes, 18 MB
uv run me-project-speit train --dataset dataset_<...>_200 --model gnn --hidden 64 --batch-size 8 --device cuda
```

A graph dataset is made of three files with the same stem:

| File | Content |
|------|---------|
| `name.csv` | one row per node of every simulation: `sim, T0, Phi0, h, r, res, x, y, z, heated, T` (`heated` = 1 on the heated face; rows of a simulation are contiguous) |
| `name_edges.npz` | the connections: arrays `sim`, `src`, `dst` (node indices *local to the simulation*, each connection in both directions) |
| `name.json` | the description (`kind: "graph"`, `rows_per_sample: null` because the size varies) |

The `gnn` model (`models/gnn.py`, pure PyTorch) encodes the inputs of each node, then repeats *message passing*: each node averages a message computed from each neighbour and updates itself. On 200 simulations it reaches a test MAE of about 0.13 °C, against 0.5 °C for predicting the mean temperature. `batch-size` counts simulations, not nodes. `show` cannot plot graph datasets yet, and `Predictor.predict(x, edge_index)` needs the connections.

> **Extending the project?** Step-by-step guides are in [`docs/`](docs/README.md): [add a dataset](docs/tutorials/01-add-a-dataset.md), [add a simulation parameter](docs/tutorials/02-add-a-simulation-parameter.md), [add a model](docs/tutorials/03-add-a-model.md), [add a dataset type](docs/tutorials/04-add-a-dataset-type.md), [checklist and troubleshooting](docs/tutorials/05-checklist-and-troubleshooting.md).

### Adding a dataset

Drop `my_data.csv` (rows: inputs then outputs) and `my_data.json` in `data/`. Minimal JSON (`kind` defaults to `field`):

```json
{"name": "my_data", "input_names": ["a", "b"], "n_outputs": 1200,
 "output_kind": "nodal_temperature", "mesh_id": "my_mesh", "kind": "field"}
```

Then `me-project-speit list datasets` shows it with its compatible models, and `train --dataset my_data` uses it. To make `generate` sample a new simulation parameter, add one `Param(default, (min, max))` line to `CUBE_PARAMS` (or to the common `PARAMS`) in `simulations/config.py` and use it in `simulations/cube.py`.

### Adding a model

Create `src/me_project_speit/models/my_model.py`:

```python
from ..datasets import FieldDataset
from .base import IncompatibleError, SurrogateModel
from .registry import register_model

@register_model("my_model")
class MyModel(SurrogateModel):
    supported_datasets = (FieldDataset,)       # dataset types it can learn from

    def __init__(self, n_inputs, n_outputs, width=64):
        super().__init__()
        ...

    @classmethod
    def check_compatible(cls, spec):          # optional: only if shape-specific
        super().check_compatible(spec)         # dataset-type check
        if spec.mesh_id != "cube_560":
            raise IncompatibleError("needs mesh 'cube_560'")
```

and import it in `models/__init__.py`. It then appears in `list models` and works with `--model my_model`.

### Running on a cluster

`resources/jupyter_gpu.sbatch` is a Slurm script starting JupyterLab on a GPU node. For batch training, run `me-project-speit train --device cuda ...` in a similar job.

## Tests

```bash
uv run pytest
```

The tests cover dataset specs, the model registry and compatibility checks, and end-to-end training/prediction on synthetic data; they do not require EasyFEA.

## Notes

- **Mesh node ordering depends on the gmsh version.** `data/coord.npy` stores the node order of the mesh used to create `data/data_test.csv`. With a different gmsh version the mesh has the same nodes in another order, so `show` will refuse to plot that file (it raises an error rather than showing a scrambled field). Datasets you generate yourself are always consistent with your own installation.
- All geometry, material and sampling parameters are in `src/me_project_speit/simulations/config.py`. If you change the mesh resolution, update `N_NODES` and `MESH_ID` and regenerate the datasets.
