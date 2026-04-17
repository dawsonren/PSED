#!/bin/bash
#SBATCH --account=p33174
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=gpumd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dawsonren@u.northwestern.edu

#------------------------------------------------------------------------------

# to change --time and config programmatically, run 
# sbatch --time=5:00:00 --job-name gpumd-nve-test gpumd.q nve_test

module purge all
module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.6.2-gcc-12.4.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

CONFIG=${1:?Usage: sbatch gpumd.q <config_name> (e.g. nve_test)}

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Running config: ${CONFIG}.yaml"
$PYTHON gb_generation/generate_gbs.py --config configs/${CONFIG}.yaml
$PYTHON thermo/run_rnemd.py --config configs/${CONFIG}.yaml
