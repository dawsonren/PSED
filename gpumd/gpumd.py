import os
import time
import warnings
warnings.filterwarnings("ignore") # critical

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
from matplotlib.colors import to_rgba, Normalize

from ase import units
from ase.io import read, write
from ase.build import bulk
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.filters import ExpCellFilter
from ase.md.bussi import Bussi
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.geometry.rdf import get_rdf

from calorine.calculators import GPUNEP, CPUNEP
from calorine.tools import relax_structure

from utils import detect_a100_h100
from nemd import swap_velocities, bin_atoms

t_start = time.perf_counter()

###
### Constants
###
# NOTE: please change to yours
NET_ID = "djr2473" 
DEBUG = True
# set path to current file, NEP_FILE should be in same directory as this script
os.chdir(os.path.dirname(__file__))
NEP_FILE = "Si_GAP_nep.txt"
# NOTE: if you followed my setup instructions, this is fine, otherwise change
GPUMD_EXEC_LOCATION = f"/home/{NET_ID}/exec/gpumd"
# location of GPUMD outputs
RUN_DIR = "rNEMD_test"
# name of file for relaxed unit cell atom locations
RELAXED_FILE = "relax.traj"
# name of file for equilibrated full system
EQUILIBRATED_FILE = "equilibrated.traj"
# [x, y, z] repetitions of the unit cells
BOX_SIZE = [20, 5, 5]
info = detect_a100_h100()

ON_GPU_NODE = info["has_gpu_node"]
ON_A100     = info["has_a100"]
ON_H100     = info["has_h100"]

# parameters for relaxation/annealing
TEMP_START = 10
TEMP_END = 300

RELAX_MD_PARAMS = [
    ('dump_exyz', [1000, 1]), # [write out positions every 100 steps, write out velocities too]
    ('dump_position', 1000),
    ('time_step', 1), # 1fs
    # ('ensemble', ['nvt_nhc', temp_1, temp_2, 100]),
    ('ensemble', ['npt_scr', TEMP_START, TEMP_END, 200, 0, 100, 1000]), # [start_temp, end_temp, tau_t (coupling constant), pressure, bulk modulus, tau_p (coupling constant)]
    ('run', 200000) # run for n steps, 4000 atoms for 200,000 steps ~ 5 mins on home GPU
]

# parameters for reverse non-equilibrium MD (to measure kappa)
TEMPERATURE_K = 300
STEPS_PER_CYCLE = 400 # controls energy flux
TIMESTEP = .4
N_CYCLES = 50 # n_cycles * steps_per_cycle = total steps
NBINS = 15 # needs to be an odd number for now
# Define which one is the cold bin and which one is the hot bin, assuming periodic boundaries. 
# NOTE: NEED AN EVEN NUMBER OF BINS FOR NOW.
COLD_BIN = NBINS // 4 + 1
HOT_BIN = 3 * NBINS // 4 + 1

RNEMD_PARAMS = [
    ('dump_position', STEPS_PER_CYCLE),
    ('dump_velocity', STEPS_PER_CYCLE),
    ('dump_thermo', STEPS_PER_CYCLE),
    ('dump_exyz', [STEPS_PER_CYCLE, 1]), # [write out positions every 100 steps, write out velocities too
    ('time_step', TIMESTEP),
    # ('ensemble', ['npt_scr', TEMPERATURE_K, TEMPERATURE_K, 10, 0, 100, 1000]), # [start_temp, end_temp, tau_t (coupling constant), pressure, bulk modulus, tau_p (coupling constant)]
    ('ensemble', ['npt_scr', TEMPERATURE_K, TEMPERATURE_K, 20, 0, 100, 200]), # [start_temp, end_temp, tau_t (coupling constant), pressure, bulk modulus, tau_p (coupling constant)]
    # ('ensemble', ['nvt_nhc', TEMPERATURE_K, TEMPERATURE_K, 100]),
    # ('ensemble', 'nve'),
    ('run', STEPS_PER_CYCLE) # run for n steps, 4000 atoms for 200,000 steps ~ 5 mins on home GPU
]

###
### Setup and Logging
###

if not ON_GPU_NODE:
    print("No GPU detected, probably on a login node.")
elif ON_H100:
    print("Running on an H100 node (compute capacity 9)")
elif ON_A100:
    print("Running on an A100 node (compute capacity 8)")
else:
    print("Quest has either installed new GPUs or something has gone horribly wrong. Exiting.")
    quit()

# set up calculators
# the CPU calculator is to relax the structure
# the GPU calculator is for the NEMD simulation

cpu_calc = CPUNEP(NEP_FILE)
gpu_calc = GPUNEP(
    NEP_FILE,
    command=GPUMD_EXEC_LOCATION,
    gpu_identifier_index=0,
    directory=RUN_DIR
)

###
### ACE Sanity Check for Si Crystal Structure
###

if DEBUG or not ON_GPU_NODE:
    print("Conducting ACE Sanity Check for Si Crystal Structure")
    struct = bulk('Si', a=5.431)
    struct.calc = cpu_calc

    cell = struct.copy().cell[:]

    # scale lattice parameter, optimal is ~1 since
    # known a = 5.431
    scales = np.linspace(0.75, 1.5, 20)
    energies = np.zeros(scales.shape)
    volumes = np.zeros(scales.shape)

    for ind, scale in enumerate(scales):
        struct.set_cell(cell * scale)
        energies[ind] = struct.get_potential_energy()
        volumes[ind] = struct.get_volume()

    fig, ax = plt.subplots()
    ax.plot(scales, energies)
    plt.savefig("ACE_Sanity_Check.png")
    plt.close()

###
### Relax Si unit cell
###

if not os.path.exists(RELAXED_FILE):
    print("Relaxing Si unit cell with CPU calculator...")
    # BUG: code doesn't have enough atoms when I set cubic=False
    # and get error "Cannot use triclinic box with only 1 target pressure component."
    # if I set constant_cell=False.
    # However, in the original code, these issues were not present.
    struct = bulk('Si', a=5.431, cubic=True)
    struct.calc = cpu_calc

    relax_structure(struct, constant_cell=True)

    write(RELAXED_FILE, struct)

atoms = read(RELAXED_FILE)
# sanity check: should still be grid aligned
assert np.allclose(atoms.cell, np.diag(np.diag(atoms.cell)), atol=1e-6)

###
### Equilibrate Structure
###

if not ON_GPU_NODE:
    print("No GPU detected, can't run MD simulation. Terminating.")
    quit()

if not os.path.exists(EQUILIBRATED_FILE):
    # Read relaxed atomic positions, attach calculator
    atoms = read(RELAXED_FILE).repeat(BOX_SIZE)
    gpu_calc.set_atoms(atoms)
    atoms.calc = gpu_calc

    # NOTE: should the number of atoms should be # of unit cells * # of atoms / unit cell?
    assert len(atoms) == np.prod(BOX_SIZE) * 8

    # make sure movie.xyz is removed before running relaxation MD
    if os.path.exists(os.path.join(RUN_DIR, 'movie.xyz')):
        os.remove(os.path.join(RUN_DIR, 'movie.xyz'))

    print("Relaxing structure with GPU calculator...")
    atoms = gpu_calc.run_custom_md(RELAX_MD_PARAMS, return_last_atoms=True)
    print("Finished relaxing structure.")

    if DEBUG:
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax)
        plt.savefig("Initial_Atom_Positions.png")

        rdf = get_rdf(atoms=atoms, rmax=5, nbins=100)

        fig, ax = plt.subplots(figsize = (3,3))
        ax.plot(rdf[1], rdf[0])
        ax.set_xlabel("Radial Distance [$\AA$]")
        ax.set_ylabel("RDF")
        plt.savefig("Initial_RDF.png")

    write(EQUILIBRATED_FILE, atoms)

###
### Setup for RNEMD
###

atoms = read(EQUILIBRATED_FILE)
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
print(atoms.get_temperature())

# Set atoms in bins
bins = np.linspace(0, 1, NBINS + 1) # nbins + 1 = n_divisions
scaled_x_positions = [atom.scaled_position[0] for atom in atoms] # compiles x coordinates of all atoms

binned_atom_indices = bin_atoms(bins, scaled_x_positions)

# visualize bins
fig, ax = plt.subplots()
colorlist = np.empty(len(atoms), dtype='object')
for bin_ind, atom_indices in enumerate(binned_atom_indices):
    if bin_ind == HOT_BIN:
        colorlist[atom_indices] = 'red'
    elif bin_ind == COLD_BIN:
        colorlist[atom_indices] = 'blue'
    else:
        colorlist[atom_indices] = 'grey'

plot_atoms(atoms, ax, colors=colorlist)
plt.savefig('RNEMD_Setup.png')

###
### RNEMD (Florian Müller-Plathe)
###

# set outputs to average
temps_times = np.zeros((N_CYCLES, NBINS)) # output temps at each time step to gauge temp convergence
velocities_hot_cold = np.zeros((N_CYCLES, 2)) # output velocities for calculating TC later

# Run simulation in cycles
for ind, cycle in enumerate(range(N_CYCLES)):
    print(f"Starting cycle {ind}...")

    # Set calculator - have to redefine it everytime because calorine is weird
    gpu_calc = GPUNEP(
        NEP_FILE, 
        command=GPUMD_EXEC_LOCATION,
        gpu_identifier_index=0,
        directory=RUN_DIR,
        atoms=atoms
    )

    # run simulation for steps_per_cycle
    atoms = gpu_calc.run_custom_md(RNEMD_PARAMS, return_last_atoms=True)

    # weird quirk of calorine, does not return atomic velocities, so have to assign them from an output file
    vels = pd.read_csv(RUN_DIR + '/velocity.out', sep = ' ', header = None).iloc[-len(atoms):, :]
    atoms.set_velocities(vels / 0.098) # another quirk, velocity units are mismatched between the two packages (see ase.units module)
    print(atoms.get_temperature()) # sanity check
    
    # Swap velocities every cycle (every n steps_per_cycle)
    hot_atom_vel, cold_atom_vel = swap_velocities(atoms, binned_atom_indices[COLD_BIN], binned_atom_indices[HOT_BIN]) # swaps velocities of a hot and cold atom in specified slabs; also returns those velocities for later processing.
    # Record swapped velocities for energy flux calculation later:
    velocities_hot_cold[ind,:] = [hot_atom_vel, cold_atom_vel]

    # record temps for averaging later
    temps = np.zeros(NBINS)
    for bin_ind, atom_indices in enumerate(binned_atom_indices):
        temps[bin_ind] = atoms[atom_indices].get_temperature()
    temps_times[ind,:] = temps
    
    # save outputs each step so you can look at them
    np.save(f'{RUN_DIR}/temps_times_cycle{ind}.npy', temps_times)
    np.save(f'{RUN_DIR}/velocities_hot_cold_cycle{ind}.npy', velocities_hot_cold)

if DEBUG:
    rdf = get_rdf(atoms=atoms, rmax=5, nbins=100)
    fig, ax = plt.subplots(figsize = (3,3))
    ax.plot(rdf[1], rdf[0])
    ax.set_xlabel('Radial Distance [$\AA$]')
    ax.set_ylabel('RDF')
    plt.savefig("Final_RDF.png")    

###
### Data Visualization from RNEMD run
###

# BUG: generated by ChatGPT, I have no idea if this is right
# but I'm not sure how rNEMD.log is generated, since my outputs didn't include it

# >>>> BEGIN CHATGPT CODE >>>>
# After RNEMD is done and `atoms` still refers to the final structure:
N_atoms = len(atoms)

thermo_path = os.path.join(RUN_DIR, 'thermo.out')
thermo = pd.read_csv(
    thermo_path,
    sep=r'\s+',
    header=None,
    names=['T', 'K', 'U', 'Pxx', 'Pyy', 'Pzz', 'Pyz', 'Pxz', 'Pxy',
           'ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']
)

# Reconstruct time in ps (GPUMD time_step is in fs)
# For the RNEMD part we're using TIMESTEP (in fs)
n_steps = len(thermo)
time_ps = np.arange(n_steps) * TIMESTEP * 1e-3

Etot_per = (thermo['K'] + thermo['U']) / N_atoms
Epot_per = thermo['U'] / N_atoms
Ekin_per = thermo['K'] / N_atoms

rnemd_df = pd.DataFrame({
    'Time[ps]': time_ps,
    'Etot/N[eV]': Etot_per,
    'Epot/N[eV]': Epot_per,
    'Ekin/N[eV]': Ekin_per,
    'T[K]': thermo['T'],
    'StressXX[GPa]': thermo['Pxx'],
    'StressYY[GPa]': thermo['Pyy'],
    'StressZZ[GPa]': thermo['Pzz'],
    'StressXY[GPa]': thermo['Pxy'],
    'StressXZ[GPa]': thermo['Pxz'],
    'StressYZ[GPa]': thermo['Pyz'],
})

rnemd_df.to_csv('rNEMD.log', sep=' ', index=False)
# <<<< END CHATGPT CODE <<<<

# Load in data from quest run:
# traj = Trajectory(filename='rNEMD.traj', mode='r')

data = pd.read_csv('rNEMD.log', sep='\s+')
data.columns = ['Time[ps]', 'Etot/N[eV]', 'Epot/N[eV]', 'Ekin/N[eV]', 'T[K]', 'StressXX[GPa]','StressYY[GPa]','StressZZ[GPa]','StressXY[GPa]','StressXZ[GPa]','StressYZ[GPa]']

temps_times = np.load(f'{RUN_DIR}/temps_times_cycle{N_CYCLES - 1}.npy', allow_pickle=True)
temps_times = np.array(list(temps_times[:]), dtype=float)

temps_times = temps_times[~np.all(temps_times == 0, axis=1)]

velocities_hot_cold = np.load(f'{RUN_DIR}/velocities_hot_cold_cycle{N_CYCLES - 1}.npy')
velocities_hot_cold = velocities_hot_cold[~np.all(velocities_hot_cold == 0, axis=1)]

# BUG: where does this come from?
bins_containing_atom_indices = np.load('bin_atom_indices.npy', allow_pickle = True)

nbins = bins_containing_atom_indices.shape[0]
bin_dims = bins * atoms.cell[0, 0]

### Fitting Fourier's Law

# calculates running average for each time step
cumulative_averages = np.cumsum(temps_times, axis=0) / (np.tile(np.atleast_2d(np.arange(temps_times.shape[0])).T, temps_times.shape[1]) + 1)

cutoff = 3
fit_params = np.polyfit(bin_dims[cutoff:-cutoff], cumulative_averages[-1,cutoff:-cutoff], 1)
x = np.linspace(bin_dims[cutoff], bin_dims[-cutoff], 10)

### Final Plot
cmap = cm.Oranges
norm = Normalize(vmin=0, vmax = len(temps_times))

# Plot temps_times
fig, axes = plt.subplots(3,1)
plt.subplots_adjust(hspace=0)
for ind, cycle in enumerate(temps_times):
    axes[0].plot(bin_dims, cycle, marker = 'o', color = cmap(norm(ind)), label = 'cycle' + str(ind))

axes[0].set_xlabel('Bin')
axes[0].set_ylabel('Temp. [K]')

# Plot cumulative averages
for ind, cycle in enumerate(cumulative_averages):
    axes[1].plot(bin_dims, cycle, marker = 'o', color = cmap(norm(ind)), label = ['Cumulative Ave. Temp' if ind == len(cumulative_averages)-1 else None])

axes[1].plot(x, fit_params[0] * x + fit_params[1], color = 'blue', linestyle = 'dashed', linewidth = 2)
axes[1].plot(x[0], fit_params[0] * x[0] + fit_params[1], marker = 'o', color = 'blue', markerfacecolor = 'white', markersize = '8', label = 'fitting range')
axes[1].plot(x[-1], fit_params[0] * x[-1] + fit_params[1], marker = 'o', color = 'blue', markerfacecolor = 'white', markersize = '8')

axes[1].set_xlabel('Bin')
axes[1].set_ylabel('Avg Temp./cycle [K]')
axes[1].legend()

# Plot Atoms
colorlist
colorlist = np.empty(len(atoms), dtype='object')
for bin_ind, atom_indices in enumerate(bins_containing_atom_indices):
    if bin_ind == HOT_BIN:
        colorlist[atom_indices] = 'red'
    elif bin_ind == COLD_BIN:
        colorlist[atom_indices] = 'blue'
    else:
        colorlist[atom_indices] = 'grey'
axes[2].set_aspect(axes[1].get_aspect())
plot_atoms(atoms, axes[2], colors = colorlist)
plt.savefig('TempTimes_CumulativAverages_Atoms.png')

t_end = time.perf_counter()
print(f"Time taken: {t_end - t_start} seconds")