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
cd gpumd
/home/djr2473/.conda/envs/chem/bin/python gpumd.py