# New-range runs

Datasets are not in git. They are deterministic: same seed + same code + same `uv.lock` = identical files.

Regenerate from the repository root:

```bash
uv sync --extra sim --extra plot
uv run me-project-speit generate -n 360 --seed 1 --output-dir data/new_ranges   # train
uv run me-project-speit generate -n 100 --seed 2 --output-dir data/new_ranges   # test
uv run me-project-speit generate -n 720 --seed 3 --output-dir data/new_ranges   # train
```

Ranges: `PARAM_RANGES` in `simulations/config.py` (T0 273–303 K, Phi0 15 000–30 000, h 50–200).

Results (`mlp_20261114_182014/`, MLP trained with `--epochs 2000 --patience 30 --seed 42`), errors in °C on the 100-run test set:

| Model | MAE | RMSE |
|---|---|---|
| MLP, 360 runs | 0.179 | 0.248 |
| MLP, 720 runs | 0.124 | 0.161 |
