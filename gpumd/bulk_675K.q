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
# sbatch --time=5:00:00 --job-name bulk bulk_675K.q

module purge all
module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.6.2-gcc-12.4.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Running bulk estimation..."
$PYTHON gb_generation/generate_gbs.py --config configs/bulk/long_675K.yaml
$PYTHON gb_generation/generate_gbs.py --config configs/bulk/longish_675K.yaml
$PYTHON gb_generation/generate_gbs.py --config configs/bulk/xlong_675K.yaml
$PYTHON gb_generation/generate_gbs.py --config configs/bulk/xxlong_675K.yaml
$PYTHON thermo/run_rnemd.py --config configs/bulk/long_675K.yaml
$PYTHON thermo/run_rnemd.py --config configs/bulk/longish_675K.yaml
$PYTHON thermo/run_rnemd.py --config configs/bulk/xlong_675K.yaml
$PYTHON thermo/run_rnemd.py --config configs/bulk/xxlong_675K.yaml
