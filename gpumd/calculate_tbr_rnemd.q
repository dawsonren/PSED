#!/bin/bash
#SBATCH --account=p33174
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=gpumd-calculate-tbr-rnemd-nve-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dawsonren@u.northwestern.edu

#------------------------------------------------------------------------------
module purge all
module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.6.2-gcc-12.4.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Running config: nve_test.yaml"
$PYTHON thermo/run_rnemd.py --config configs/nve_test.yaml