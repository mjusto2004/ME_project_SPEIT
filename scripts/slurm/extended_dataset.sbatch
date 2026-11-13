#!/bin/bash
# Extend data_test.csv with random runs, retrain the baseline MLP, compare on shared test sets.
# See docs/experiments.md. Existing runs in datasets/raw/extended_runs/ are skipped.
#SBATCH --job-name=extended_dataset
#SBATCH --output=extended_dataset-%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -e
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "/home/asoare/ME_project"
source "/home/asoare/ME_project/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N_EXTRA=270      # new training runs added to the 90 in data_test.csv
N_TEST=100       # independent test runs
EPOCHS=2000
STAMP=$(date +%Y%m%d_%H%M%S)
PKG=src.me_project_speit
RUNS=src/me_project_speit/runs
DATA=src/me_project_speit/datasets
N_TOTAL=$((90 + N_EXTRA))
BASE_DIR=$RUNS/mlp_trained_on_data_test_$STAMP
EXT_DIR=$RUNS/mlp_trained_on_data_test_extended_${N_TOTAL}_$STAMP

# 1) Simulations
python -m $PKG.extended_dataset.generate --name train --n $N_EXTRA --seed 1 --workers "$SLURM_CPUS_PER_TASK"
python -m $PKG.extended_dataset.generate --name test --n $N_TEST --seed 2 --workers "$SLURM_CPUS_PER_TASK"

# 2) Baseline MLP on data_test.csv (also fixes the original test split)
python -m $PKG.train --dataset $DATA/data_test.csv --epochs $EPOCHS --device cpu --output-dir "$BASE_DIR"

# 3) Extended dataset and MLP trained on it
python -m $PKG.extended_dataset.build --baseline-ckpt "$BASE_DIR/best_model.pt" --n-extra $N_EXTRA
python -m $PKG.extended_dataset.train --data $DATA/data_test_extended_$N_TOTAL.npz \
    --epochs $EPOCHS --output-dir "$EXT_DIR"

# 4) Comparison
python -m $PKG.extended_dataset.compare \
    --model "MLP trained on data_test.csv (90)=$BASE_DIR/best_model.pt" \
    --model "MLP trained on data_test_extended ($N_TOTAL)=$EXT_DIR/best_model.pt" \
    --baseline-ckpt "$BASE_DIR/best_model.pt" \
    --independent $DATA/independent_test_$N_TEST.npz --out $RUNS/comparison_$STAMP.csv
