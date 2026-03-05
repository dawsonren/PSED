import time

#Standard Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd

#ASE
from ase.io import read, write
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.filters import ExpCellFilter

# MD
from ase import units

from ase.md.bussi import Bussi
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from calorine.calculators import GPUNEP

from utils.muller_plathe import swap_velocities, bin_atoms

#
import glob
import os
import sys
import json
import shutil as sh

import warnings
warnings.filterwarnings("ignore") # critical

# ----------------------------------------------------------------------------- #
# Set MD Parameters
run_dir = 'rNEMD'
temperature_K = 300
steps_per_cycle = 400 # controls energy flux
timestep = .4
n_cycles = 50 # n_cycles * steps_per_cycle = total steps
nbins = 15 # needs to be an odd number for now

md_params = [
('dump_position', steps_per_cycle),
('dump_velocity', steps_per_cycle),
('dump_exyz', [steps_per_cycle, 1]), # [write out positions every 100 steps, write out velocities too]
 ('time_step', timestep),
# ('ensemble', ['npt_scr', temperature_K, temperature_K, 10, 0, 100, 1000]), # [start_temp, end_temp, tau_t (coupling constant), pressure, bulk modulus, tau_p (coupling constant)]
('ensemble', ['npt_scr', temperature_K, temperature_K, 20, 0, 100, 200]), # [start_temp, end_temp, tau_t (coupling constant), pressure, bulk modulus, tau_p (coupling constant)]
# ('ensemble', ['nvt_nhc', temperature_K, temperature_K, 100]),
# ('ensemble', 'nve'),
 ('run', steps_per_cycle)] # run for n steps, 4000 atoms for 200,000 steps ~ 5 mins on home GPU

# ----------------------------------------------------------------------------- #
# Set Structure, initial velocities
atoms = read('equilibrated.traj')
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
print(atoms.get_temperature())

# -------------------------------------------------------------------------------------------- #
# Set atoms in bins
bins = np.linspace(0,1,nbins + 1) # nbins + 1 = n_divisions
scaled_x_positions = [atom.scaled_position[0] for atom in atoms] # compiles x coordinates of all atoms

binned_atom_indices = bin_atoms(bins, scaled_x_positions)

# Define which one is the cold bin and which one is the hot bin, assuming periodic boundaries. 
# NEED AN EVEN NUMBER OF BINS FOR NOW.
cold_bin = nbins // 4 + 1 
hot_bin = 3 * nbins // 4 + 1

# Visualize Bins
fig, ax = plt.subplots()
colorlist = np.empty(len(atoms), dtype='object')
for bin_ind, atom_indices in enumerate(binned_atom_indices):
    if bin_ind == hot_bin:
        colorlist[atom_indices] = 'red'
    elif bin_ind == cold_bin:
        colorlist[atom_indices] = 'blue'
    else:
        colorlist[atom_indices] = 'grey'

plot_atoms(atoms, ax, colors=colorlist)
plt.savefig('rNEMD_Setup.png', format='png')

# -------------------------------------------------------------------------------------------- #
# reverse Non-Equilibrium Molecular Dynamics a la Florian Muller-Plathe:

# set outputs to average
temps_times = np.zeros((n_cycles, nbins)) # output temps at each time step to gauge temp convergence
velocities_hot_cold = np.zeros((n_cycles,2)) # output velocities for calculating TC later

# Run simulation in cycles
for ind, cycle in enumerate(range(n_cycles)):
    # -------------------------------------- #
    print(ind)

    # Set calculator - have to redefine it everytime because calorine is weird
    calc = GPUNEP('Si_GAP_nep.txt', 
              command='/home/dawson-smith/Desktop/Research/OtherSoftware/GPUMD/src/gpumd',
              gpu_identifier_index=0,
              directory=run_dir,
              atoms = atoms) # replace path on quest

    # run simulation for steps_per_cycle
    atoms = calc.run_custom_md(md_params, return_last_atoms=True)

    # weird quirk of calorine, does not return atomic velocities, so have to assign them from an output file
    vels = pd.read_csv(run_dir + '/velocity.out', sep = ' ', header = None).iloc[-len(atoms):, :]
    atoms.set_velocities(vels / 0.098) # another quirk, velocity units are mismatched between the two packages (see ase.units module)
    print(atoms.get_temperature()) # sanity check
    
    # -------------------------------------- #
    # Swap velocities every cycle (every n steps_per_cycle)
    hot_atom_vel, cold_atom_vel = swap_velocities(atoms, binned_atom_indices[cold_bin], binned_atom_indices[hot_bin]) # swaps velocities of a hot and cold atom in specified slabs; also returns those velocities for later processing.
    # Record swapped velocities for energy flux calculation later:
    velocities_hot_cold[ind,:] = [hot_atom_vel, cold_atom_vel]

    # record temps for averaging later
    temps = np.zeros(nbins)
    for bin_ind, atom_indices in enumerate(binned_atom_indices):
        temps[bin_ind] = atoms[atom_indices].get_temperature()
    temps_times[ind,:] = temps
    
    # save outputs each step so you can look at them
    np.save('temps_times.npy', temps_times)
    np.save('velocities_hot_cold.npy', velocities_hot_cold)

plt.rcParams['figure.dpi'] = 150
from matplotlib import cm
from matplotlib.colors import Normalize

# ----------------------------------------------------------------- #
# Load in data from quest run:
# traj = Trajectory(filename='rNEMD.traj', mode='r')

data = pd.read_csv('rNEMD.log', sep='\s+')
data.columns = ['Time[ps]', 'Etot/N[eV]', 'Epot/N[eV]', 'Ekin/N[eV]', 'T[K]', 'StressXX[GPa]','StressYY[GPa]','StressZZ[GPa]','StressXY[GPa]','StressXZ[GPa]','StressYZ[GPa]']

temps_times = np.load('temps_times.npy', allow_pickle=True)
temps_times = np.array(list(temps_times[:]), dtype=float)

temps_times = temps_times[~np.all(temps_times == 0, axis=1)]

velocities_hot_cold = np.load('velocities_hot_cold.npy')
velocities_hot_cold = velocities_hot_cold[~np.all(velocities_hot_cold == 0, axis=1)]

bins_containing_atom_indices = np.load('bin_atom_indices.npy', allow_pickle = True)

nbins = bins_containing_atom_indices.shape[0]
bins = np.linspace(0,1,nbins) # nbins + 1 = n_divisions
cold_bin = nbins // 4 + 1
hot_bin = 3 * nbins // 4 + 1

bin_dims = bins * atoms.cell[0,0]

# ----------------------------------------------------------------- #
# Do some math; fit profile here
cumulative_averages = np.cumsum(temps_times, axis=0) / (np.tile(np.atleast_2d(np.arange(temps_times.shape[0])).T, temps_times.shape[1]) + 1)# calculates running average for each time step

print(cumulative_averages.shape)
print(bin_dims.shape)

cutoff = 3
fit_params = np.polyfit(bin_dims[cutoff:-cutoff], cumulative_averages[-1,cutoff:-cutoff], 1)
print(fit_params)
x = np.linspace(bin_dims[cutoff], bin_dims[-cutoff], 10)

# ------------------------------------------------------------------ #
# Setup Plots
cmap = cm.Oranges
norm = Normalize(vmin=0, vmax = len(temps_times))

    # ------------------------------------ #
    # Plot temps_times
fig, axes = plt.subplots(3,1)
plt.subplots_adjust(hspace=0)
for ind, cycle in enumerate(temps_times):
    axes[0].plot(bin_dims, cycle, marker = 'o', color = cmap(norm(ind)), label = 'cycle' + str(ind))

axes[0].set_xlabel('Bin')
axes[0].set_ylabel('Temp. [K]')

    # ------------------------------------ #
    # Plot cumulative averages
for ind, cycle in enumerate(cumulative_averages):
    axes[1].plot(bin_dims, cycle, marker = 'o', color = cmap(norm(ind)), label = ['Cumulative Ave. Temp' if ind == len(cumulative_averages)-1 else None])

axes[1].plot(x, fit_params[0] * x + fit_params[1], color = 'blue', linestyle = 'dashed', linewidth = 2)
axes[1].plot(x[0], fit_params[0] * x[0] + fit_params[1], marker = 'o', color = 'blue', markerfacecolor = 'white', markersize = '8', label = 'fitting range')
axes[1].plot(x[-1], fit_params[0] * x[-1] + fit_params[1], marker = 'o', color = 'blue', markerfacecolor = 'white', markersize = '8')

axes[1].set_xlabel('Bin')
axes[1].set_ylabel('Avg Temp./cycle [K]')
axes[1].legend()

    # ------------------------------------ #
    # Plot Atoms
colorlist
colorlist = np.empty(len(atoms), dtype='object')
for bin_ind, atom_indices in enumerate(bins_containing_atom_indices):
    if bin_ind == hot_bin:
        colorlist[atom_indices] = 'red'
    elif bin_ind == cold_bin:
        colorlist[atom_indices] = 'blue'
    else:
        colorlist[atom_indices] = 'grey'
axes[2].set_aspect(axes[1].get_aspect())
plot_atoms(atoms, axes[2], colors = colorlist)