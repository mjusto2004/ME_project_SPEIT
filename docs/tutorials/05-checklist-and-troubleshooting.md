# 5. Checklist and troubleshooting

## Did I finish? — checklist per kind of change

**New dataset (existing type)**
- [ ] `name.csv` and `name.json` side by side in `data/`
- [ ] `uv run me-project-speit list datasets` shows it, with at least one compatible model
- [ ] a short `train --epochs 3` runs

**New simulation parameter**
- [ ] added to `PARAM_DEFAULTS` **and** `PARAM_RANGES` (`config.py`)
- [ ] used in `run_simulation`
- [ ] does not change the mesh (or gets a new `mesh_id`)

**New model**
- [ ] `@register_model("name")`, constructor `(n_inputs, n_outputs, …defaults)`
- [ ] imported in `models/__init__.py`
- [ ] `supported_datasets` set; `check_compatible` calls `super()` first
- [ ] appears in `list models` and under the right datasets in `list datasets`
- [ ] test for shapes and for rejection of a wrong dataset

**New dataset type**
- [ ] see the checklist at the end of [tutorial 4](04-add-a-dataset-type.md)

**Always**
- [ ] `uv run --extra dev pytest` is green
- [ ] README / docs updated

## Error messages and what they mean

| Message | Cause | Fix |
|---------|-------|-----|
| `Model 'x' is incompatible with dataset 'y' (...): needs a dataset of type [...]` | the model does not accept this dataset type | pick a model from the `Compatible models` list at the end of the message, or another dataset |
| `... incompatible ...: needs mesh 'cube_560', dataset has '...'` | a mesh-specific model on another mesh | use another model, or regenerate the data on the right mesh |
| `Unknown model 'x'. Available: [...]` | typo, or the module is not imported in `models/__init__.py` | fix the name / add the import |
| `A model named 'x' is already registered.` | two models share a name | rename one |
| `has N columns and no description file 'x.json'` | CSV without sidecar | create the JSON ([tutorial 1](01-add-a-dataset.md)) |
| `expected N columns (k inputs + m outputs), got M` | JSON and CSV disagree | fix `input_names` / `n_outputs` |
| `rows is not a multiple of rows_per_sample` | wrong `rows_per_sample`, or truncated CSV | fix the JSON or regenerate |
| `Unknown dataset kind 'x'. Available: [...]` | `kind` in the JSON is not registered (typo, or the module is not imported in `datasets/__init__.py`) | fix the JSON or register it ([tutorial 4](04-add-a-dataset-type.md)) |
| `Unknown input parameter(s) [...]` / `Unknown simulation parameter(s)` | name not in `config.PARAM_*` | add it ([tutorial 2](02-add-a-simulation-parameter.md)) |
| `This model expects k inputs (...), got n` | wrong number of inputs for `Predictor.predict` | pass inputs in the order of `predictor.spec.input_names` |
| `The rebuilt mesh does not match the node coordinates in data/coord.npy` | different gmsh version → different node order | use the same gmsh version, or regenerate the dataset |
| `CUDA is unavailable` | `--device cuda` without a CUDA build of torch / no GPU | see *GPU* in the README, or use `--device cpu` |

## Debugging tips

- **Inspect what a dataset says about itself:**
  ```python
  from me_project_speit.datasets import load_spec
  print(load_spec("data/my_data.csv").describe())
  ```
- **Ask the registry directly:**
  ```python
  from me_project_speit.datasets import load_spec
  from me_project_speit.models import compatible_models, check_compatible
  spec = load_spec("data/my_data.csv")
  print(compatible_models(spec))
  check_compatible("mlp", spec)      # raises IncompatibleError with the reason
  ```
- **Inspect a checkpoint:** `torch.load(path, weights_only=True)` contains `model_name`, `model_kwargs`, `spec`, `x_mean`, `x_std`, `y_mean`, `y_std`, `train_indices`, ...
- **EasyFEA is optional.** Tests, `train`, `list` and `Predictor` need only `numpy` and `torch`; install `--extra sim` for `generate` and `show`.
- `uv run` re-syncs the environment to the given extras. Use `uv run --extra dev --extra sim ...` if tools such as `pytest` or `EasyFEA` seem to disappear.
