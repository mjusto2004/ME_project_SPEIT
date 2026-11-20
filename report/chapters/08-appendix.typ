#import "../common.typ": *

#heading(numbering: none)[Appendix: reproducing the results]

Datasets (CPU only, about 12 minutes in total on 6 cores):

```bash
uv sync --extra sim --extra plot --extra dev
uv run me-project-speit generate -n 720 --seed 1 --output-dir data/report                  # cube, training
uv run me-project-speit generate -n 100 --seed 2 --output-dir data/report                  # cube, independent test
uv run me-project-speit generate -n 360 --seed 1 --kind pointwise --output-dir data/report
uv run me-project-speit generate -n 100 --seed 2 --kind pointwise --output-dir data/report
uv run me-project-speit generate -n 1000 --kind graph --res 5 10 --seed 10 --output-dir data/report
uv run me-project-speit generate -n 50 --kind graph --res 7  --seed 11 --output-dir data/report
uv run me-project-speit generate -n 20 --kind graph --res 12 --seed 12 --output-dir data/report
uv run me-project-speit generate -n 20 --kind graph --res 18 --seed 13 --output-dir data/report
uv run me-project-speit generate -n 12 --kind graph --res 25 --seed 14 --output-dir data/report
```

The files are then renamed as in the scripts (`cube_train_all720`, `cube_independent_100`,
`cube_pointwise_360`, `cube_pointwise_independent_100`, `sponge_train_1000`, `sponge_test_res<r>`).
Experiments and figures:

```bash
python report/scripts/exp_cube.py data/report/cube_train_all720.csv data/report/cube_independent_100.csv
python report/scripts/exp_cube2.py            # extrapolation, small data, timing
uv run me-project-speit train --dataset data/report/cube_pointwise_360.csv --model coord_mlp     --hidden 256 256 256 256 --epochs 300 --patience 300 --scheduler cosine --warmup 5     --batch-size 4096 --output-dir runs/report/cube_coord_mlp_360
python report/scripts/exp_sponge.py gnn gnn8 gno
python report/scripts/fig_datasets.py; python report/scripts/fig_sponges.py
python report/scripts/fig_cube_results.py; python report/scripts/fig_sponge_results.py
typst compile report/report.typ
```

Hardware: laptop with an RTX 5070 Laptop GPU (8 GB). The heaviest run (GNO, 100 epochs, 20 min) uses
6.8 GB of GPU memory at about 55 °C and 50 W.
