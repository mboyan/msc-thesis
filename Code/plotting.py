import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import plotly.graph_objects as go
import pandas as pd
import h5py

def plot_spore_positions(Ns, Hs, spore_arrangements, dx, titles=None, top_view=False):
    """
    Plot the spore positions in 3D.
    inputs:
        Ns (list of int): the sizes of the bottom of the lattice for each arrangement
        Hs (list of int): the heights of the lattice for each arrangement
        spore_arrangements (list of numpy arrays): the coordinates of the spores for each arrangement,
            each numpy array should have shape (n_spores, 3) where the columns are the x, y, and z coordinates
        dx (float): the lattice spacing in micrometers
        title (str): the title of the plot
        top_view (bool): whether to plot in 2D
    """

    assert len(Ns) == len(Hs) == len(spore_arrangements), "The number of spore arrangements must match the number of lattice sizes and heights"
    if titles is not None:
        assert len(titles) == len(spore_arrangements), "The number of titles must match the number of spore arrangements"

    if top_view:
        fig, ax = plt.subplots(1, len(spore_arrangements), figsize=(4*len(spore_arrangements), 4))
    else:
        fig, ax = plt.subplots(1, len(spore_arrangements), subplot_kw={'projection': '3d'}, figsize=(4*len(spore_arrangements), 4))
    
    for i, spore_coords in enumerate(spore_arrangements):
        
        if top_view:
            ax[i].scatter(spore_coords[:, 0] * dx, spore_coords[:, 1] * dx, marker='.', s=1)
        else:
            ax[i].scatter(spore_coords[:, 0] * dx, spore_coords[:, 1] * dx, spore_coords[:, 2] * dx, marker='.')
    
        ax[i].set_xlim(0, Ns[i] * dx)
        ax[i].set_ylim(0, Ns[i] * dx)
        ax[i].set_xlabel('$x$ [$\mu$m]')
        ax[i].set_ylabel('$y$ [$\mu$m]')

        if not top_view:
            ax[i].set_zlim(0, Hs[i] * dx)
            ax[i].set_zlabel('$z$ [$\mu$m]')

        if titles:
            ax[i].set_title(titles[i])
    
    plt.tight_layout(pad=2)
    plt.show()


def plot_experiment_results(expID, select_sims=None, logx=False, logy=False, target_thresh=None, mark_spore=True, color_pairs=False):
    """
    Plot the results of a general diffusion experiment.
    inputs:
        expID (int): the ID of the experiment
        select_sim (list of int): the IDs of the simulations to plot
        semilogy (bool): whether to plot the y-axis in log scale
        target_thresh (float): the target threshold
        mark_spore (bool): whether to mark the spore on the final concentration plot
        color_pairs (bool): whether to use a color palette with pairs of colors
    """

    exp_params = pd.read_csv(f"Data/{expID}_exp_params.csv")
    sim_results = pd.read_csv(f"Data/{expID}_sim_results.csv")

    if select_sims is not None:
        unique_simIDs = select_sims
    else:
        unique_simIDs = exp_params['simID'].unique()

    # Create figure for evolution graph
    figA, axA = plt.subplots(figsize=(10, 6))
    if logx:
        axA.set_xscale('log')
    if logy:
        axA.set_yscale('log')
    axA.set_xlabel('Time [s]')
    axA.set_ylabel('Concentration [M]')
    axA.set_title('Concentration evolution')
    axA.grid()

    # Color palette
    if color_pairs:
        palette = plt.get_cmap('tab20')
    else:
        palette = plt.get_cmap('tab10')

    # Create figure for final concentration
    nrows = np.ceil(len(unique_simIDs) / 2).astype(int)
    figB, axsB = plt.subplots(nrows, 2, figsize=(4, nrows*2))
    ax_ct = 0

    for simID in unique_simIDs:
        sim_params = exp_params[exp_params['simID'] == simID].iloc[0]
        sim_results_data = sim_results[sim_results['simID'] == simID]

        label = sim_results_data['label'].iloc[0]

        print(f"Plotting simulation {simID}: {label}")

        # Plot the concentration evolution
        axA.plot(sim_results_data['time'], sim_results_data['c_numerical'], label=label, color=palette(ax_ct))
        axA.plot(sim_results_data['time'], sim_results_data['c_analytical'], color=palette(ax_ct), linestyle='dashed')

        # Filter out thresholds that are not within the simulation time
        times_thresh = sim_results_data['times_thresh'].iloc[-1].strip('[]')
        times_thresh = [float(x) for x in times_thresh.split()]
        c_thresh = sim_results_data['c_thresh'].iloc[-1].strip('[]')
        c_thresh = [float(x) for i, x in enumerate(c_thresh.split()) if times_thresh[i] > 0]
        times_thresh = [x for x in times_thresh if x > 0]

        # Plot the concentration thresholds
        axA.vlines(times_thresh, 0, c_thresh, colors='r', color=palette(ax_ct), linestyles='dotted', linewidth=1)
        axA.hlines(c_thresh, 0, times_thresh, colors='r', color=palette(ax_ct), linestyles='dotted', linewidth=1)
        # axA.set_ylim(max(1e-12, np.min(sim_results_data['c_numerical'])), 1.2*np.max(sim_results_data['c_numerical']))
        axA.set_ylim(1e-12, 1.2*np.max(sim_results_data['c_numerical']))
        # axA.set_xlim(0, 1000)

        # Identify spore location
        spore_idx = sim_results_data['spore_idx'].iloc[-1].strip('()')
        spore_idx = [int(x) for x in spore_idx.split(',')]
        
        # Get concentration frames
        # L = sim_params['N'] * sim_params['dx']
        c_evolution = np.load(f"Data/{expID}_{simID}_frames.npy")
        # with h5py.File(f"Data/{expID}_frames.h5", 'r') as f:
        #     c_lattice = f[simID][:]
        if sim_params['dims'] == 2:
            # c_start = c_lattice[0, ...]
            c_final = c_evolution[-1, ...]
        elif sim_params['dims'] == 3:
            # c_start = c_lattice[0, :, (c_lattice.shape[2] - 1) // 2, :]
            c_final = c_evolution[-1, :, spore_idx[1], :].T

        # Plot the final concentration
        if nrows > 1:
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].imshow(c_final, cmap='viridis', origin='lower')
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].set_title(label)
            
            # Mark spore with a red circle
            if mark_spore: axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        else:
            axsB[ax_ct].imshow(c_final, cmap='viridis', origin='lower')
            axsB[ax_ct].set_title(label)
            
            # Mark spore with a red circle
            if mark_spore: axsB[ax_ct].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        
        ax_ct += 1
    
    if target_thresh is not None:
        axA.axhline(y=target_thresh, color='r', linestyle='-.', label='Target threshold')

    axA.legend(fontsize='small', ncol=np.ceil((len(unique_simIDs) + 1)/2))
    
    plt.tight_layout()
    plt.show()


def plot_periodic_experiment_results(expID, select_sims=None, logx=False, logy=False, target_thresh=None, mark_spore=True, color_pairs=False):
    """
    Aligns latices of periodically repeating experiments
    with different spore densities and plots the results.
    inputs:
        expID (int): the ID of the experiment
        select_sim (list of int): the IDs of the simulations to plot
        semilogy (bool): whether to plot the y-axis in log scale
        target_thresh (float): the target threshold
        mark_spore (bool): whether to mark the spore on the final concentration plot
        color_pairs (bool): whether to use a color palette with pairs of colors
    """

    exp_params = pd.read_csv(f"Data/{expID}_exp_params.csv")
    sim_results = pd.read_csv(f"Data/{expID}_sim_results.csv")

    if select_sims is not None:
        unique_simIDs = select_sims
    else:
        unique_simIDs = exp_params['simID'].unique()
    
    # Align lattices to largest lattice
    N_max = sim_results['N'].max()

    # Create figure for evolution graph
    figA, axA = plt.subplots(figsize=(10, 6))
    if logx:
        axA.set_xscale('log')
    if logy:
        axA.set_yscale('log')
    axA.set_xlabel('Time [s]')
    axA.set_ylabel('Concentration [M]')
    axA.set_title('Concentration evolution')
    axA.grid()

    # Color palette
    if color_pairs:
        palette = plt.get_cmap('tab20')
    else:
        palette = plt.get_cmap('tab10')

    # Create figure for final concentration
    nrows = np.ceil(len(unique_simIDs) / 2).astype(int)
    figB, axsB = plt.subplots(nrows, 2, figsize=(5, nrows*2.5))
    ax_ct = 0

    # Get concentration frames and global concentration range
    c_evolutions = {}
    c_min = np.inf
    c_max = -np.inf
    for simID in unique_simIDs:
        c_evolution = np.load(f"Data/{expID}_{simID}_frames.npy")
        c_evolutions[simID] = c_evolution
        c_min = min(c_min, np.min(c_evolution[-1]))
        c_max = max(c_max, np.max(c_evolution[-1]))

    for simID in unique_simIDs:
        # Read simulation data
        sim_params = exp_params[exp_params['simID'] == simID].iloc[0]
        sim_results_data = sim_results[sim_results['simID'] == simID]
        
        # Get label and lattice size from simulation data
        label = sim_results_data['label'].iloc[0]
        N = sim_results_data['N'].iloc[0]

        print(f"Plotting simulation {simID}: {label}")

        # Plot the concentration evolution
        axA.plot(sim_results_data['time'], sim_results_data['c_numerical'], label=label, color=palette(ax_ct))
        axA.plot(sim_results_data['time'], sim_results_data['c_analytical'], color=palette(ax_ct), linestyle='dashed')

        # Filter out thresholds that are not within the simulation time
        times_thresh = sim_results_data['times_thresh'].iloc[-1].strip('[]')
        times_thresh = [float(x) for x in times_thresh.split()]
        c_thresh = sim_results_data['c_thresh'].iloc[-1].strip('[]')
        c_thresh = [float(x) for i, x in enumerate(c_thresh.split()) if times_thresh[i] > 0]
        times_thresh = [x for x in times_thresh if x > 0]

        # Plot the concentration thresholds
        axA.vlines(times_thresh, 0, c_thresh, colors='r', color=palette(ax_ct), linestyles='dotted', linewidth=1)
        axA.hlines(c_thresh, 0, times_thresh, colors='r', color=palette(ax_ct), linestyles='dotted', linewidth=1)
        axA.set_ylim(max(1e-12, min(0.1*np.min(sim_results_data['c_numerical']), 0.1*target_thresh)), 1.2*np.max(sim_results_data['c_numerical']))
        # axA.set_xlim(0, 1000)
        
        # Get concentration frames
        # L = sim_params['N'] * sim_params['dx']
        c_evolution = c_evolutions[simID]
        
        # with h5py.File(f"Data/{expID}_frames.h5", 'r') as f:
        #     c_lattice = f[simID][:]
        if sim_params['dims'] == 2:
            # Repeat lattice to align with largest lattice
            c_lattice_repeat = np.pad(c_evolution, ((0, 0), (0, N_max - N), (0, N_max - N)), 'wrap')
            spore_idx = (N // 2, N // 2, N // 2)
            c_final = c_lattice_repeat[-1, ...]

        elif sim_params['dims'] == 3:
            # Repeat lattice to align with largest lattice
            c_lattice_repeat = np.pad(c_evolution, ((0, 0), (0, N_max - N), (0, N_max - N), (0, N_max - N)), 'wrap')
            spore_idx = (N // 2, N // 2, N // 2)
            c_final = c_lattice_repeat[-1, :, spore_idx[1], :].T
        
        # Plot the final concentration
        if nrows > 1:
            im=axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].imshow(c_final, cmap='viridis', origin='lower', vmin=c_min, vmax=c_max)
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].set_title(label)

            # Add colorbar
            # cbar = figB.colorbar(im, ax=axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2])
            
            # Mark spore with a red circle
            if mark_spore: axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        else:
            axsB[ax_ct].imshow(c_final, cmap='viridis', origin='lower', vmin=c_min, vmax=c_max)
            axsB[ax_ct].set_title(label)
            
            # Mark spore with a red circle
            if mark_spore: axsB[ax_ct].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        
        # Mark boundary of original lattice
        if N < N_max:
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].vlines(N, 0, N_max, colors='r', linestyles='dotted')
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].hlines(N, 0, N_max, colors='r', linestyles='dotted')

        ax_ct += 1
    
    if target_thresh is not None:
        axA.axhline(y=target_thresh, color='r', linestyle='-.', label='Target threshold')

    if color_pairs:
        ncols = ncol=np.ceil(len(unique_simIDs)/2)
    else:
        ncols = 1
    axA.legend(fontsize='small', ncol=ncols)
    
    plt.tight_layout()
    plt.show()