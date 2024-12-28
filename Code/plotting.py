import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import plotly.graph_objects as go
import pandas as pd
import ast

def plot_spore_positions(N, H, spores_x, spores_y, spores_z, dx, title=None, top_view=False):
    """
    Plot the spore positions in 3D.
    inputs:
        N (int): the size of the bottom of the lattice
        H (int): the height of the lattice
        spores_x (numpy array): the x coordinates of the spores
        spores_y (numpy array): the y coordinates of the spores
        spores_z (numpy array): the z coordinates of the spores
        dx (float): the lattice spacing in micrometers
        title (str): the title of the plot
        top_view (bool): whether to plot in 2D
    """
    
    if top_view:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(spores_x * dx, spores_y * dx, marker='.', s=1)
    else:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(spores_x * dx, spores_y * dx, spores_z * dx, marker='.')
    
    ax.set_xlim(0, N * dx)
    ax.set_ylim(0, N * dx)
    ax.set_xlabel('$x$ $[\mu m]$')
    ax.set_ylabel('$y$ $[\mu m]$')

    if not top_view:
        ax.set_zlim(0, H * dx)
        ax.set_zlabel('$z$ $[\mu m]$')

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_experiment_results(expID, select_sim=None, semilogy=False, target_thresh=None):
    """
    Plot the results of an experiment.
    inputs:
        expID (int): the ID of the experiment
        select_sim (int): the ID of the simulation to plot
        semilogy (bool): whether to plot the y-axis in log scale
        target_thresh (float): the target threshold
    """

    exp_params = pd.read_csv(f"Data/{expID}_exp_params.csv")
    sim_results = pd.read_csv(f"Data/{expID}_sim_results.csv")

    if select_sim is not None:
        unique_simIDs = [select_sim]
    else:
        unique_simIDs = exp_params['simID'].unique()

    # Create figure for evolution graph
    figA, axA = plt.subplots(figsize=(10, 6))
    if semilogy:
        axA.set_yscale('log')
    axA.set_xlabel('Time [s]')
    axA.set_ylabel('Concentration [M]')
    axA.set_title('Concentration evolution')
    axA.grid()

    # Color palette
    palette = plt.get_cmap('tab10')

    # Create figure for final concentration
    nrows = np.ceil(len(unique_simIDs) / 2).astype(int)
    figB, axsB = plt.subplots(nrows, 2, figsize=(5, nrows*2.5))
    ax_ct = 0

    for simID in unique_simIDs:
        sim_params = exp_params[exp_params['simID'] == simID].iloc[0]
        sim_results_data = sim_results[sim_results['simID'] == simID]

        label = sim_results_data['label'].iloc[0]

        # Plot the concentration evolution
        axA.plot(sim_results_data['time'], sim_results_data['c_numerical'], label=label, color=palette(ax_ct))
        axA.plot(sim_results_data['time'], sim_results_data['c_analytical'], color=palette(ax_ct), linestyle='dashed')

        # Plot the concentration thresholds
        times_thresh = sim_results_data['times_thresh'].iloc[-1].strip('[]')
        times_thresh = [float(x) for x in times_thresh.split()]
        c_thresh = sim_results_data['c_thresh'].iloc[-1].strip('[]')
        c_thresh = [float(x) for i, x in enumerate(c_thresh.split()) if times_thresh[i] > 0]
        times_thresh = [x for x in times_thresh if x > 0]
        axA.vlines(times_thresh, 0, c_thresh, colors='r', color=palette(ax_ct), linestyles='dotted')
        axA.hlines(c_thresh, 0, times_thresh, colors='r', color=palette(ax_ct), linestyles='dotted')
        axA.set_ylim(1e-12, 1.1*np.max(sim_results_data['c_numerical']))
        
        # Get concentration frames
        L = sim_params['N'] * sim_params['dx']
        c_lattice = np.load(f"Data/{expID}_{simID}_frames.npy")
        if sim_params['dims'] == 2:
            c_start = c_lattice[0, ...]
            c_final = c_lattice[-1, ...]
        elif sim_params['dims'] == 3:
            c_start = c_lattice[0, :, (c_lattice.shape[2] - 1) // 2, :]
            c_final = c_lattice[-1, :, (c_lattice.shape[2] - 1) // 2, :].T
        
        # Identify spore location
        c_max_init = np.max(c_start)
        spore_idx = np.where(c_start == c_max_init)

        # Plot the final concentration
        if nrows > 1:
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].imshow(c_final, cmap='viridis', origin='lower')
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].set_title(label)
        else:
            axsB[ax_ct].imshow(c_final, cmap='viridis', origin='lower')
            axsB[ax_ct].set_title(label)
        
        # Mark spore with a red circle
        axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')

        ax_ct += 1
    
    if target_thresh is not None:
        axA.axhline(y=target_thresh, color='r', linestyle='dotted', label='Target threshold')

    axA.legend(fontsize='small')
    
    plt.tight_layout()
    plt.show()