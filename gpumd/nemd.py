import numpy as np

def swap_velocities(atoms, cold_bin_indices, hot_bin_indices):
    """
    Swaps velocities of hottest atom in cold region and coldest atom in hot region.
    """
    # -------------------------------------------------------------- #
    # get coldest and hottest indices from each bin
    hottest_ind = cold_bin_indices[np.argmax(np.linalg.norm(atoms[cold_bin_indices].get_velocities(), axis = 1))] # Hottest atom in cold bin
    coldest_ind = hot_bin_indices[np.argmin(np.linalg.norm(atoms[hot_bin_indices].get_velocities(), axis = 1))] # coldest atom in hot bin

    # swap momenta (later, velocities)
    assert atoms[coldest_ind].symbol == atoms[hottest_ind].symbol
    cold_vel = atoms[coldest_ind].momentum.copy()
    hot_vel = atoms[hottest_ind].momentum.copy()

    atoms[coldest_ind].momentum = hot_vel
    atoms[hottest_ind].momentum = cold_vel

    # return velocities (hot, cold)
    return np.linalg.norm(hot_vel / atoms[hottest_ind].mass), np.linalg.norm(cold_vel/ atoms[coldest_ind].mass)


def bin_atoms(bins, scaled_x_positions):
    """
    Returns a list of lists containing atom indices per bin.
    [bin_1[atom indices in bin], bin_2[atom indices in bin 2]...]

    Args:
    bins (np.array): scaled x positions designating each bin.
    scaled_x_positions (np.array): scaled x positions for every atom in the simulation box 
    """
    
    # -------------------------------- #
    # assign atoms to bins
    atom_bin_assignments = np.digitize(scaled_x_positions, bins) # places atoms in bins based on index - len(atoms), 1 = first bin, 2 = second...

    # Find a way to vectorize this
    bins_containing_atom_indices = np.empty(np.unique(atom_bin_assignments).size, dtype = 'object')
    for ind, search_val in enumerate(np.unique(atom_bin_assignments)):
        bins_containing_atom_indices[ind] = np.where(atom_bin_assignments == search_val)[0]

    return bins_containing_atom_indices