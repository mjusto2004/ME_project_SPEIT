#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=jupyter-%j.out
#SBATCH --error=jupyter-%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module purge
module load python/3.12.3 cuda/12.4

cd /home/infres/justo-24/ME\ project
source /home/infres/justo-24/ME\ project/.venv/bin/activate

echo "GPU node: $(hostname)"
nvidia-smi

python -m jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888