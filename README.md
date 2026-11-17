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
│   ├── models/               model registry + compatibility (base.py, registry.py, mlp.py, gnn.py, pod.py)
│   ├── training.py           training loop, early stopping, evaluation
│   ├── inference.py          Predictor: load a checkpoint and predict
│   ├── evaluation.py         infer a model on a dataset: errors per simulation, 3D figures
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
me-project-speit generate -n 500 --kind graph --res 8 12   # graph meshes with res drawn in 8..12 (--res 25: fixed)
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
| `--hidden` | `128 256 256` (`mlp`), `64` (`gnn`) | hidden layer sizes (`mlp`) / node state width (`gnn`) |
| `--layers` | `4` | message-passing rounds (`gnn` only) |
| `--n-modes`, `--regressor` | `5`, `gp` | `pod` only: POD modes kept, regression of their coefficients (`gp` or `ridge`) |
| `--radius`, `--max-neighbours`, `--eval-neighbours`, `--exclude-inputs` | `0.15`, `24`, = max, `res` | `gno` only, see [below](#the-gno-model-train-on-coarse-meshes-predict-on-fine-ones) |
| `--test-dataset` | | extra independent test set (same inputs and mesh); reports `Independent test MAE` |
| `--weight-decay` | 0 | L2 penalty of the optimizer (`mlp`, `coord_mlp`, `gnn`) |
| *model options* | | each model declares its own (`train --model <name> --help` lists them) |
| `--epochs` | 300 | maximum number of epochs |
| `--batch-size` | 64 | mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 30 | early-stopping patience (epochs) |
| `--scheduler` | `none` | learning-rate schedule: `none` (constant `--lr`), `cosine`, `plateau` (see below) |
| `--warmup` | 0 | `cosine`: epochs of linear warmup |
| `--min-lr` | 0 | `cosine`: final rate; `plateau`: lowest rate |
| `--plateau-patience`, `--plateau-factor` | 10, 0.5 | `plateau`: epochs without progress before the rate is multiplied by the factor |
| `--seed` | 42 | seed for the split and initialisation |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--output-dir` | `runs/<dataset>_<timestamp>` | where to write results |

The data is split 70/15/15 (train/validation/test). Inputs and outputs are standardised using training-set statistics only. The best model on validation MSE is kept and finally scored on the test set (MAE and RMSE in °C). The run directory contains:

- `best_model.pt` – weights, model name and arguments, dataset spec, normalisation statistics, split indices, dataset path;
- `history.csv` – per-epoch train/validation metrics and learning rate (`lr`, the rate of the first batch of the epoch).

**Learning-rate schedules** (`mlp`, `coord_mlp`, `gnn`, `gno`; `pod` has no learning rate):

- `--scheduler cosine`: after `--warmup` epochs rising linearly to `--lr`, the rate follows a cosine down to `--min-lr` at the last epoch, updated at every batch. The decay is spread over `--epochs`, so set a realistic `--epochs` (e.g. 300–500) and a `--patience` large enough not to stop the run in the middle of it (`train` prints a note otherwise).
- `--scheduler plateau`: the rate is multiplied by `--plateau-factor` after `--plateau-patience` epochs without a better validation MSE, never below `--min-lr`; training stops when the rate is at `--min-lr` and still not improving. Keep `--patience` larger than `--plateau-patience`, e.g. `--scheduler plateau --plateau-patience 10 --patience 40 --min-lr 1e-6`.

`plot` draws the rate next to the training curves.

### 3. Infer a trained model on a dataset

```bash
uv run me-project-speit infer                                        # latest run, test split of its training data
uv run me-project-speit infer --run runs/<folder> --dataset independent_100
uv run me-project-speit infer --run runs/<folder> --dataset independent_100 --sim 12 --half --mesh
uv run me-project-speit infer --dataset independent_100 --csv errors.csv --save figures/indep --no-plot
```

Loads `best_model.pt` (model in `eval` mode), predicts every simulation of the dataset and prints the errors in °C, plus the worst and best simulations with their parameters:

- **per simulation** (recommended to compare models): the MAE of each simulation, averaged so that every simulation weighs the same, with a **95 % confidence interval** (bootstrap over the simulations), the median and the 95th percentile, and the **NMAE**: MAE divided by the range (max − min) of the simulation's true field, in %. The simulation is the independent unit: the nodes of one simulation are strongly correlated, so if the intervals of two models overlap, this test set cannot separate them;
- **over all node values**: MAE, RMSE, max |error| and bias (mean of predicted − true). On `graph` datasets, large meshes weigh more here. It warns when simulations have parameters **outside the training ranges** (the model is extrapolating there). The dataset must have the same kind, inputs and mesh as the training data.

| Option | Default | Description |
|--------|---------|-------------|
| `--run` | latest in `runs/` | run folder or `best_model.pt` |
| `--dataset` | training dataset | CSV path or name in `--data-dir` |
| `--subset` | `auto` | `test` (test split of the run, training dataset only), `all`; `auto` = `test` on the training dataset, `all` otherwise |
| `--sim` | worst simulation | simulation (number in the dataset) drawn in 3D |
| `--half` | | draw only the nodes with y <= 0.5 |
| `--mesh` | | also open interactive PyVista views (true, predicted, error) on the cube mesh (`field`, `pointwise`) |
| `--csv` | | write MAE / NMAE / RMSE / max error of every simulation to a CSV |
| `--save` | | write `<SAVE>_inference.png` instead of opening a window |
| `--no-plot`, `--top` | | only print the errors; number of worst/best simulations listed (5) |

The figure shows, for the chosen simulation, the true temperature, the prediction and the absolute error at every node in 3D, then, over all evaluated simulations, predicted vs true, the error histogram and the MAE of each simulation. Every dataset kind has 3D coordinates: `pointwise` and `graph` store them as inputs, `field` datasets on the cube mesh rebuild it with EasyFEA.

Python API:

```python
from me_project_speit.evaluation import evaluate, plot_evaluation

ev = evaluate("runs/<folder>", "data/independent_100.csv")
print(ev.summary())
ev.metrics()                  # {'sim_MAE', 'sim_MAE_lo', 'sim_MAE_hi', 'NMAE', 'MAE', 'RMSE', ...}
ev.worst().params, ev.get(12).pred
plot_evaluation(ev, sim=12)
```

### 4. Visualise simulations

```bash
me-project-speit show --dataset data/data_test.csv --ids 0 1
```

Opens an interactive 3D plot per simulation, with a shared colour scale.

### 5. Plot a training run

```bash
uv sync --extra plot                              # installs matplotlib (once)
uv run me-project-speit plot                      # the latest run in runs/
uv run me-project-speit plot --run runs/<folder> --sample 3 --half
uv run me-project-speit plot --save figures/run1  # write PNG files instead of opening windows
```

Two figures: the **training curves** (loss and validation MAE per epoch, with the best epoch marked), and the **true temperature, the prediction and the absolute error** on the nodes of one test simulation (`--sample` picks which; `--half` keeps only y <= 0.5 to see inside the cube), with a predicted-vs-true plot, the error histogram and the error of each test simulation (same figure as `infer` on the test split). The 3D views open in a window you can rotate.

### 6. List models and datasets

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
| `field` (default) | `FieldDataset` | one per simulation | `T0, Phi0, h` → all 560 node temperatures | `mlp`, `pod` |
| `pointwise` | `PointwiseDataset` | one per (simulation, node) | `T0, Phi0, h, x, y, z` → temperature at that node | `coord_mlp` |
| `graph` | `GraphDataset` | one per node; every simulation has its own mesh | `T0, Phi0, h, r, res, x, y, z, heated` + the node connections → temperature at that node | `gnn`, `gno` |

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

#### The `gno` model: train on coarse meshes, predict on fine ones

`gnn` passes messages along mesh links, so what it learns is counted in links: 8 rounds reach 8/`res` of the cube, and a model trained at `res` 5–10 fails at `res` 25. `gno` (`models/gno.py`, a *graph neural operator*, after Li et al. 2020, "Neural Operator: Graph Kernel Network") works in physical distances instead:

- the neighbours of a node are the nodes within `--radius` (m), whatever the mesh, so `--layers` rounds reach `layers × radius` at any resolution;
- a message depends on the neighbour's state and on its relative position and distance; averaging the messages estimates an integral over the ball, which does not depend on how finely it is meshed;
- at most `--max-neighbours` nodes of the ball are drawn at random (a new draw each training step, a fixed one in evaluation so predictions are reproducible; `--eval-neighbours` draws more at inference);
- a global state (mean over the simulation) goes back to every node at each round;
- `--exclude-inputs` (default `res`) lists inputs the model ignores: the resolution must not matter. `infer` does not warn about inputs the model ignores.

The mesh edges are only used to tell the simulations of a batch apart. Defaults: `--hidden 128 --layers 6 --radius 0.15 --max-neighbours 24` (reach 0.9 m, about 0.9 M parameters). A training step on 2 coarse meshes takes ~20 ms and 0.6 GB on the RTX 5070; a res-25 mesh (16 000 nodes) is predicted in ~1 s.

Resolution experiment:

```bash
uv run me-project-speit generate -n 1000 --kind graph --res 8 12 --seed 1            # train: coarse
uv run me-project-speit generate -n 50 --kind graph --res 10 --seed 2                # test, inside the training range
uv run me-project-speit generate -n 50 --kind graph --res 18 --seed 3                # test, finer
uv run me-project-speit generate -n 50 --kind graph --res 25 --seed 4                # test, much finer
uv run me-project-speit train --dataset <train> --model gno --batch-size 8 --device cuda
uv run me-project-speit infer --dataset <test_res10> --no-plot    # then res 18 and 25: MAE vs res
```

Compare with `--model gnn` on the same files. Part of the error at a new resolution is not the model's: the reference temperatures of a coarse finite-element mesh are themselves less accurate than those of a fine one.

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
