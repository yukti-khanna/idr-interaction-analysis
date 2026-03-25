from energy_funcs import yukawa_potential, ah_potential, calc_dmap, genParamsDH
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable




@njit(nopython=True)
def calc_frame_energy_per_pair_numba(dmap, sig, lam, eps_lj, qmap, eps_yu, k_yu, rc_lj=10.0, rc_yu=4.0):
    # Flatten all input 2D arrays to 1D
    dmap_flat = dmap.flatten()
    sig_flat = sig.flatten()
    lam_flat = lam.flatten()
    qmap_flat = qmap.flatten()
    # Initialize energy arrays
    u_ah = np.zeros(dmap_flat.shape)
    u_yu = np.zeros(dmap_flat.shape)
    
    for i in range(len(dmap_flat)):
        ri = dmap_flat[i]       # Correct indexing
        sigi = sig_flat[i]      # Correct indexing
        lami = lam_flat[i]      # Correct indexing
        qi = qmap_flat[i]       # Correct indexing
        # Ashbaugh-Hatch potential
        u_ah[i] = ah_potential(ri, sigi, eps_lj, lami, rc_lj)
        # Yukawa potential
        u_yu[i] = yukawa_potential(ri, qi, k_yu, eps_yu, rc_yu)
    return u_ah, u_yu  # Return the per pair energy


def get_atom_pairs(selection1, selection2):
    """
    Generate a list of atom/residue pairs.
    """
    res1 = selection1.resids  # Residue numbers for selection 1
    res2 = selection2.resids  # Residue numbers for selection 2

    # Generate all possible pairs of residues
    atom_pairs = [(res1[i], res2[j]) for i in range(len(res1)) for j in range(len(res2))]

    return atom_pairs


def calc_per_pair_average_energy(u, params, eps_lj, temp, ionic, rc_lj=2.0, rc_yu=4.0):
    eps_yu, k_yu = genParamsDH(temp, ionic)
    selection1 = u.select_atoms("segid A")
    selection2 = u.select_atoms("segid B")
    n_frames = len(u.trajectory)
    res1 = selection1.resnames
    res2 = selection2.resnames

    indices1 = selection1.indices
    indices2 = selection2.indices
    # Get the atom pairs with residue numbers
    atom_pairs = get_atom_pairs(selection1, selection2)
    
    pairs_array = np.array(atom_pairs)

    # Precompute sigma, lambda, and qmap
    sig = np.array([[np.array((params.loc[res1[i], 'sigmas'] + params.loc[res2[j], 'sigmas'])/2) 
                    for j in range(len(res2))] for i in range(len(res1))])

    lam = np.array([[np.array((params.loc[res1[i], 'lambdas'] + params.loc[res2[j], 'lambdas'])/2) 
                    for j in range(len(res2))] for i in range(len(res1))])

    qmap = np.array([[params.loc[res1[i], 'q'] * params.loc[res2[j], 'q'] for j in range(len(res2))] for i in range(len(res1))])

    # Initialize arrays to store total per pair energies
    total_ah_per_pair = np.zeros(len(pairs_array))
    total_yu_per_pair = np.zeros(len(pairs_array))

    # Store energies for each frame
    all_ah_per_pair = np.zeros((n_frames, len(atom_pairs)))
    all_yu_per_pair = np.zeros((n_frames, len(atom_pairs)))

    # Define the stride (skip every nth frame)
    #stride = 5  # Change as needed

    # Iterate over every nth frame in the trajectory
    for frame_idx, ts in enumerate(u.trajectory):
        
        #if frame_idx % stride != 0:
            #continue  # Skip frames that are not multiples of stride

        pos1 = u.atoms[indices1].positions  # Get current positions for atoms in selection1
        pos2 = u.atoms[indices2].positions  # Get current positions for atoms in selection2
        box = ts.dimensions  # Box dimensions for periodic boundary conditions
        
        # Use MDAnalysis distance_array to calculate distance map with PBC
        dmap = calc_dmap(pos1, pos2, box=box)
        
        # Calculate per-pair energy for current frame
        u_ah, u_yu = calc_frame_energy_per_pair_numba(dmap, sig, lam, eps_lj, qmap, eps_yu, k_yu, rc_lj, rc_yu)
        # Accumulate total energies per pair
        total_ah_per_pair += u_ah
        total_yu_per_pair += u_yu

        # Store energies for the current frame (adjusted index for storage)
        all_ah_per_pair[frame_idx] = u_ah
        all_yu_per_pair[frame_idx] = u_yu

    # Calculate average energies per pair
    avg_ah_per_pair = total_ah_per_pair / n_frames
    avg_yu_per_pair = total_yu_per_pair / n_frames

    #avg_ah = np.mean(all_ah_per_pair[1])
    #avg_yu = np.mean(all_yu_per_pair[1])
    # Return the per-pair energies and the list of atom pairs
    return avg_ah_per_pair, avg_yu_per_pair, pairs_array, res1, res2


def calc_avg_energies(u, params, eps_lj, temp, ionic, rc_lj=2.0, rc_yu=4.0):
    eps_yu, k_yu = genParamsDH(temp, ionic)
    selection1 = u.select_atoms("segid A")
    selection2 = u.select_atoms("segid B")
    n_frames = len(u.trajectory)
    res1 = selection1.resnames
    res2 = selection2.resnames

    indices1 = selection1.indices
    indices2 = selection2.indices
    atom_pairs = get_atom_pairs(selection1, selection2)
    pairs_array = np.array(atom_pairs)

    # Precompute sigma, lambda, and qmap
    sig = np.array([[np.array((params.loc[res1[i], 'sigmas'] + params.loc[res2[j], 'sigmas'])/2) 
                    for j in range(len(res2))] for i in range(len(res1))])

    lam = np.array([[np.array((params.loc[res1[i], 'lambdas'] + params.loc[res2[j], 'lambdas'])/2) 
                    for j in range(len(res2))] for i in range(len(res1))])

    qmap = np.array([[params.loc[res1[i], 'q'] * params.loc[res2[j], 'q'] for j in range(len(res2))] for i in range(len(res1))])

    # Initialize arrays to store total per pair energies
    total_ah_per_pair = np.zeros(len(pairs_array))
    total_yu_per_pair = np.zeros(len(pairs_array))

    # Store energies for each frame
    all_ah_per_pair = np.zeros((n_frames, len(atom_pairs)))
    all_yu_per_pair = np.zeros((n_frames, len(atom_pairs)))

    # Iterate over frames in the trajectory
    for frame_idx, ts in enumerate(u.trajectory):
        pos1 = u.atoms[indices1].positions  # Get positions for selection1 atoms
        pos2 = u.atoms[indices2].positions  # Get positions for selection2 atoms
        box = ts.dimensions  # Box dimensions for periodic boundary conditions

        # Use MDAnalysis distance_array to calculate distance map with PBC
        dmap = calc_dmap(pos1, pos2, box=box)
        
        # Calculate per-pair energy for the current frame
        u_ah, u_yu = calc_frame_energy_per_pair_numba(dmap, sig, lam, eps_lj, qmap, eps_yu, k_yu, rc_lj, rc_yu)
        
        # Accumulate total energies per pair
        total_ah_per_pair += u_ah
        total_yu_per_pair += u_yu

        # Store energies for the current frame
        all_ah_per_pair[frame_idx] = u_ah
        all_yu_per_pair[frame_idx] = u_yu

    # Calculate average energies per pair over all frames
    avg_ah_per_pair = total_ah_per_pair / n_frames
    avg_yu_per_pair = total_yu_per_pair / n_frames

    # Calculate the overall average energy (sum of energies for all atom pairs divided by the total number of pairs)
    avg_ah = np.mean(avg_ah_per_pair)
    avg_yu = np.mean(avg_yu_per_pair)
    err_ah = np.std(avg_ah_per_pair)
    err_yu = np.std(avg_yu_per_pair)

    # Return the overall average energies along with other data
    return avg_ah, avg_yu, err_ah, err_yu


def plot_energy_matrix(res1, res2, pairs, emap_ave, box_size, temp, key):
    """
    Plot the energy matrix for residue pairs.
    
    Parameters:
    pairs : list of tuple
        List of residue pairs (atom indices)
    emap_ave : np.array
        Average interaction energies for each pair
    """

    n_rows = len(res1)
    n_cols = len(res2)
    # Initialize the energy matrix with the correct shape
    E_matrix = np.zeros((n_rows, n_cols))

    # Check for non-finite values in emap_ave
    if not np.isfinite(emap_ave).all():
        print("emap_ave contains non-finite values:", emap_ave)
        emap_ave = np.nan_to_num(emap_ave, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs or infinities if needed
    # Fill the energy matrix
    for i, e in enumerate(emap_ave):
        row = pairs[i][0] - 1  # Adjust for zero-based indexing
        col = pairs[i][1] - 1  # Adjust for zero-based indexing
        E_matrix[row][col] = e
    
    if not np.isfinite(E_matrix).all():
        print("E_matrix contains non-finite values. Replacing with zeroes.")
        E_matrix = np.nan_to_num(E_matrix, nan=0.0, posinf=0.0, neginf=0.0)  # or another appropriate value

    # Set color limits based on the absolute maximum value in E_matrix
    max_val = np.max(np.abs(E_matrix))
    min_val = np.min(np.abs(E_matrix))
    if not np.isfinite(max_val):
        max_val = np.nanmax(np.abs(E_matrix))  # Use a fallback if max_val is NaN or infinite
        print("yes")

    # Skip plotting if max_val is zero
    if max_val == 0:
        print("Maximum value in E_matrix is zero. Skipping plot.")
        return
    
    # Create color map
    cmp = mpl.colors.ListedColormap(['#0500cf','#6d6dff','#8a8aff','w','#fea3a3','#fd6c6d','#de0102'])
    #print("min", -1*max_val, "half min", -1*max_val*0.5, "one tenth min", -1*max_val*0.1, "one hundreth min", -1*max_val*0.01, "one hundreth max", 1*max_val*0.01, "one tenth max", max_val*0.1, "half max", max_val*0.5, "max", max_val)
    norm = mpl.colors.BoundaryNorm([-max_val, -max_val*0.5, -max_val*0.1, -max_val*0.01, max_val*0.01, max_val*0.1, max_val*0.5, max_val], cmp.N)

    fig, ax1 = plt.subplots()
    im1 = ax1.imshow(E_matrix, norm=norm, cmap=cmp, origin='lower', aspect='equal')
    
    # Add colorbar
    divider = make_axes_locatable(ax1)
    ax_cb = divider.new_vertical(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im1, cax=ax_cb, label="Energy [KJ/mol]", orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    ax_cb.xaxis.set_label_position('top')
    ax_cb.xaxis.set_ticks_position('top')
    
    # Set x-axis and y-axis ticks and labels to residue names
    ax1.set_xticks(np.arange(len(res2)))
    ax1.set_xticklabels(res2, rotation=90)  # Rotate for better visibility
    ax1.set_yticks(np.arange(len(res1)))
    ax1.set_yticklabels(res1)
    ax1.tick_params(axis='x', labelsize=3)  # Change size of x-axis tick labels
    ax1.tick_params(axis='y', labelsize=3)  # Change size of y-axis tick labels

    ax1.set_xlabel("Residue number for Chain 1")
    ax1.set_ylabel("Residue number for Chain 2")
    plt.savefig(f"{key}_energy_map_{box_size}_{temp}K.png", dpi=300, bbox_inches='tight')
    plt.close()



