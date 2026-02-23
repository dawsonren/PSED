#!/bin/bash
#SBATCH --account=p33174
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4
#SBATCH --job-name=gpumd

#------------------------------------------------------------------------------
module purge all
module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.6.2-gcc-12.4.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

# Config file controls the entire pipeline (GB generation + RNEMD).
# Change this to run a different experiment:
CONFIG=${1:-configs/small_box.yaml}

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Running config: $CONFIG"
$PYTHON gb_generation/generate_gbs.py --config "$CONFIG"
$PYTHON rnemd/run_rnemd.py --config "$CONFIG"
