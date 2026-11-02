#!/bin/bash
#SBATCH --job-name=MLP_training
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Load environment
module purge
module load python/3.11 cuda/12.4

source /home/infres/justo-24/ME\ project/.venv/bin/activate

# Move to project directory
cd "/home/infres/justo-24/ME project/pipeline"

# Dataset
DATASET="/home/infres/justo-24/ME project/pipeline/datasets/data_test.csv"

# Check dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found: $DATASET"
    exit 1
fi

# CPU configuration
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Job information
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Dataset: $DATASET"

# Check GPU
nvidia-smi

# Train MLP
python -u train.py \
    --dataset "$DATASET" \
    --device cuda \
    --epochs 1500 \
    --batch-size 64 \
    --lr 0.001

echo "Job finished at $(date)"
