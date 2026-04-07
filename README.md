# PSED Project

Studying Si Grain Boundaries. Structure-property relationship for thermal boundary resistance.

## Quest Commands
To request a GPU for some amount of time in interactive mode:
`srun --partition=gengpu --gres=gpu:1 --mem <mem>G --time=<hh>:<mm>:<ss> --pty --account <account> bash -i`
    
To submit a batch job to run the full GB generation/RNEMD campaign:
`sbatch --time=8:00:00 --job-name gpumd-nve-test-xlarge gpumd/gpumd.q nve_test_xlarge`
