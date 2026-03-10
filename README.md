# PSED Project

Studying Si Grain Boundaries. Structure-property relationship for thermal boundary resistance.

## Quest Commands
To request a GPU for some amount of time in interactive mode:
`srun --partition=gengpu --gres=gpu:1 --mem <mem>G --time=<hh>:<mm>:<ss> --pty --account <account> bash -i`
    
To submit a batch job:
`sbatch <script>`

I have my prinmary script in `gpumd/gpumd.q`.