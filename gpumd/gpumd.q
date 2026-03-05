#!/bin/bash
#SBATCH --account=p33174
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64
#SBATCH --job-name=gpumd

#------------------------------------------------------------------------------
module purge all
module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.6.2-gcc-12.4.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Running config: small_box"
$PYTHON gb_generation/generate_gbs.py --config configs/small_box.yaml
$PYTHON thermo/run_rnemd.py --config configs/small_box.yaml

echo "Running config: medium_box"
$PYTHON gb_generation/generate_gbs.py --config configs/medium_box.yaml
$PYTHON thermo/run_rnemd.py --config configs/medium_box.yaml

echo "Running config: large_box"
$PYTHON gb_generation/generate_gbs.py --config configs/large_box.yaml
$PYTHON thermo/run_rnemd.py --config configs/large_box.yaml