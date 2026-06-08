#!/bin/bash
#SBATCH --account=p33174
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=72G
#SBATCH --job-name=descriptors
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dawsonren@u.northwestern.edu

#------------------------------------------------------------------------------

# to change --time and config programmatically, run 
# sbatch --time=4:00:00 --job-name descriptors descriptors.q full

module purge all
module load gcc/12.4.0-gcc-8.5.0
module load openblas/0.3.28-gcc-12.4.0

source /home/${USER}/.bashrc
source activate chem

CONFIG=${1:?Usage: sbatch descriptors.q <config_name> (e.g. full)}

cd gpumd
PYTHON=/home/djr2473/.conda/envs/chem/bin/python

echo "Creating ML Descriptors for: ${CONFIG}.yaml"
$PYTHON ml/format_cleaned_data.py --config configs/${CONFIG}.yaml
