# Couldn't find exact documentation for old GPUMD v3.3.1
# so this my best guess for order of parameters.
# You can check out old GPUMD SW files here:
# https://github.com/brucefan1983/GPUMD/tree/f76237c692ce286d27afb37dafcbb2874551a6a7/potentials/sw

# Format guess:
# sw_1985 1
# ε   λ   A   B   a   γ   σ   cos(θ₀)

# Constants provided by (Lee and Hwang 2012) specifically for thermal conduction in Si
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.85.125204

sw_1985 1
1.41992 29.5304 7.049556277 0.6022245584 1.80 1.20 2.1051937 -3.333333333333333e-1