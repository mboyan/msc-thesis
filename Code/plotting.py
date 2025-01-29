import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import plotly.graph_objects as go
import pandas as pd
import conversions as conv
from scipy.optimize import curve_fit

def plot_spore_positions(Ns, Hs, spore_arrangements, dx, titles=None, top_view=False):
    """
    Plot the spore positions in 3D.
    inputs:
        Ns (list of int): the sizes of the bottom of the lattice for each arrangement
        Hs (list of int): the heights of the lattice for each arrangement
        spore_arrangements (list of numpy arrays): the coordinates of the spores for each arrangement,
            each numpy array should have shape (n_spores, 3) where the columns are the x, y, and z coordinates
        dx (float): the lattice spacing in micrometers
        titles (list of str): the titles of the plots for each arrangement
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

        if titles is not None:
            ax[i].set_title(titles[i])
    
    plt.tight_layout(pad=2)
    plt.show()


def plot_experiment_results(expID, select_sims=None, logx=False, logy=False, target_thresh=None,
                            mark_spore=True, color_pairs=False, heatmap_size=5, title=None, label_last=False, t_max=None):
    """
    Plot the results of a general diffusion experiment.
    inputs:
        expID (int): the ID of the experiment
        select_sims (list of int): the IDs of the simulations to plot
        logy (bool): whether to plot the y-axis in log scale
        target_thresh (float): the target threshold
        mark_spore (bool): whether to mark the spore on the final concentration plot
        color_pairs (bool): whether to use a color palette with pairs of colors
        title (str): the title of the plot
        label_last (bool): whether to label only the first and last concentration values
        t_max (float): the maximum time to plot
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
    if title is None:
        axA.set_title('Concentration evolution')
    else:
        axA.set_title(title)
    axA.grid()

    # Color palette
    if color_pairs:
        palette = plt.get_cmap('tab20')
        pal_discrete = True
    elif len(unique_simIDs) < 10:
        palette = plt.get_cmap('tab10')
        pal_discrete = True
    else:
        palette = plt.get_cmap('viridis')
        pal_discrete = False

    # Create figure for final concentration
    nrows = np.ceil(len(unique_simIDs) / 2).astype(int)
    figB, axsB = plt.subplots(nrows, 2, figsize=(heatmap_size, nrows*0.5*heatmap_size))
    ax_ct = 0

    # Get concentration frames and global concentration range
    c_evolutions = {}
    lattice_sizes = {}
    c_min = np.inf
    c_max = -np.inf
    N_max = -np.inf
    for simID in unique_simIDs:
        c_evolution = np.load(f"Data/{expID}_{simID}_frames.npy", mmap_mode='r')
        c_evolutions[simID] = c_evolution
        lattice_sizes[simID] = (c_evolution.shape[1], c_evolution.shape[-1])
        c_min = min(c_min, np.min(c_evolution[-1]))
        c_max = max(c_max, np.max(c_evolution[-1]))
        N_max = max(N_max, lattice_sizes[simID][0])

    # Plot results
    for simID in unique_simIDs:
        # Read simulation data
        sim_params = exp_params[exp_params['simID'] == simID].iloc[0]
        sim_results_data = sim_results[sim_results['simID'] == simID]
    	
        # Get label from simulation data
        if label_last:
            if simID == unique_simIDs[0] or simID == unique_simIDs[-1]:
                label = sim_results_data['label'].iloc[0]
            else:
                label = None
        else:
            label = sim_results_data['label'].iloc[0]

        # Lattice size
        N = lattice_sizes[simID][0]
        H = lattice_sizes[simID][1]

        print(f"Plotting simulation {simID}: {label}")

        # Palette index
        if pal_discrete:
            pal_idx = ax_ct % palette.N
        else:
            pal_idx = ax_ct / len(unique_simIDs)

        # Plot the concentration evolution
        if t_max is not None:
            time_mask = sim_results_data['time'] <= t_max
        else:
            time_mask = np.ones(len(sim_results_data['time']), dtype=bool)
        nonzero_mask = sim_results_data['c_numerical'] > 0
        total_mask = time_mask & nonzero_mask
        axA.plot(sim_results_data['time'][total_mask], sim_results_data['c_numerical'][total_mask], label=label, color=palette(pal_idx))
        axA.plot(sim_results_data['time'][total_mask], sim_results_data['c_analytical'][total_mask], color=palette(pal_idx), linestyle='dashed')

        # Filter out thresholds that are not within the simulation time
        times_thresh = sim_results_data['times_thresh'].iloc[-1].strip('[]')
        times_thresh = [float(x) for x in times_thresh.split()]
        c_thresh = sim_results_data['c_thresh'].iloc[-1].strip('[]')
        c_thresh = [float(x) for i, x in enumerate(c_thresh.split()) if times_thresh[i] > 0]
        times_thresh = [x for x in times_thresh if x > 0]

        # Plot the concentration thresholds
        axA.vlines(times_thresh, 0, c_thresh, colors='r', color=palette(pal_idx), linestyles='dotted', linewidth=1)
        axA.hlines(c_thresh, 0, times_thresh, colors='r', color=palette(pal_idx), linestyles='dotted', linewidth=1)
        # axA.set_ylim(max(1e-12, np.min(sim_results_data['c_numerical'])), 1.2*np.max(sim_results_data['c_numerical']))
        axA.set_ylim(max(1e-12, min(0.1*np.min(sim_results_data['c_numerical'][total_mask]), 0.1*target_thresh)), 1.2*np.max(sim_results_data['c_numerical'][total_mask]))
        # axA.set_ylim(1e-12, 1.2*np.max(sim_results_data['c_numerical']))
        # axA.set_xlim(0, 1000)

        # Identify spore location
        if 'spore_idx' in sim_results_data.columns:
            spore_idx = sim_results_data['spore_idx'].iloc[-1].strip('()')
            spore_idx = [int(x) for x in spore_idx.split(',')]
        else:
            if sim_params['dims'] == 2:
                spore_idx = (N // 2, N // 2)
            elif sim_params['dims'] == 3 and N == H:
                spore_idx = (N // 2, N // 2, N // 2)
            elif sim_params['dims'] == 3 and N != H:
                # Detect bottom spore arrangement
                spore_idx = (N // 2, N // 2, 1)
            else:
                print("Error: spore index not found")
        print(f"Spore index: {spore_idx}")
        
        # Get concentration frames
        c_evolution = c_evolutions[simID][total_mask, ...]

        if sim_params['dims'] == 2:
            if N < N_max:
                # Repeat lattice to align with largest lattice
                c_final = np.pad(c_evolution[-1, ...], ((0, N_max - N), (0, N_max - N)), 'wrap')
            else:
                c_final = c_evolution[-1, ...]
        elif sim_params['dims'] == 3:
            if N < N_max:
                # Repeat lattice to align with largest lattice
                c_final = np.pad(c_evolution[-1, :, spore_idx[1], :].T, ((0, N_max - N), (0, N_max - N)), 'wrap')
            else:
                c_final = c_evolution[-1, :, spore_idx[1], :].T

        # Plot the final concentration
        if nrows > 1:
            im=axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].imshow(c_final, cmap='viridis', origin='lower', vmin=c_min, vmax=c_max)
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].set_title(sim_results_data['label'].iloc[0])
            
            # Add colorbar
            # cbar = figB.colorbar(im, ax=axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2])

            # Mark spore with a red circle
            if mark_spore: axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        else:
            im=axsB[ax_ct].imshow(c_final, cmap='viridis', origin='lower')
            axsB[ax_ct].set_title(sim_results_data['label'].iloc[0])
            
            # Mark spore with a red circle
            if mark_spore: axsB[ax_ct].scatter(spore_idx[0], spore_idx[1], color='r', marker='o', facecolors='none')
        
        # Mark boundary of original lattice
        if N < N_max:
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].vlines(N, 0, N_max-1, colors='r', linestyles='dotted')
            axsB[np.floor(ax_ct / 2).astype(int), ax_ct % 2].hlines(N, 0, N_max-1, colors='r', linestyles='dotted')

        ax_ct += 1
    
    # Add a single horizontal colorbar at the bottom
    cbar_ax = figB.add_axes([0.15, -0.02, 0.7, 0.02])
    figB.colorbar(im, cax=cbar_ax, orientation='horizontal')

    if target_thresh is not None:
        axA.axhline(y=target_thresh, color='r', linestyle='-.', label='Target threshold')

    if color_pairs:
        ncols = np.ceil(len(unique_simIDs)/2)
    else:
        ncols = 1
    axA.legend(fontsize='small', ncol=ncols)
    
    plt.tight_layout()
    plt.show()


def plot_densities_vs_concentrations_at_time(expID, time, alogx=False, alogy=False, blogx=False, blogy=False, model_fit=None):
    """
    Plot the concentration at a specific time for all simulations in an experiment.
    inputs:
        expID (int): the ID of the experiment
        time (float): the time at which to plot the concentration
        alogx (bool): whether to plot the x-axis in log scale of the first plot
        alogy (bool): whether to plot the y-axis in log scale of the first plot
        blogx (bool): whether to plot the x-axis in log scale of the second plot
        blogy (bool): whether to plot the y-axis in log scale of the second plot
        model_fit (function): the function to fit the data
    """

    exp_params = pd.read_csv(f"Data/{expID}_exp_params.csv")
    sim_results = pd.read_csv(f"Data/{expID}_sim_results.csv")

    unique_simIDs = exp_params['simID'].unique()

    # Read simulation results
    results_dict = {'concentration': [], 'density': [], 'spore_dist': []}
    for simID in unique_simIDs:
        # Get concentration frames
        c_evolution = np.load(f"Data/{expID}_{simID}_frames.npy", mmap_mode='r')

        # Read simulation data
        sim_params = exp_params[exp_params['simID'] == simID].iloc[0]
        sim_results_data = sim_results[sim_results['simID'] == simID]
        
        # Reconstruct spore density
        N = c_evolution.shape[1]
        dx = sim_params['dx']
        spore_dist = N * dx
        lattice_vol = spore_dist**3
        spore_density = 1 / lattice_vol
        
        # Convert to 1/mL
        spore_density = conv.inverse_micrometers_cubed_to_mL(spore_density)
        results_dict['density'].append(spore_density)
        results_dict['spore_dist'].append(spore_dist)

        time_mask = (sim_results_data['time'] <= time) & (sim_results_data['c_numerical'] > 0)
        c_t = sim_results_data['c_numerical'][time_mask].iloc[-1]
        print(c_t)
        results_dict['concentration'].append(c_t)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_dict)

    # Sort by density
    results_df = results_df.sort_values('density')

    # Fit linear regression
    if model_fit is not None:
        # fit = np.polyfit(results_df['density'], results_df['concentration'], 1)
        coeffs, _ = curve_fit(model_fit, results_df['density'], results_df['concentration'], p0=np.full(len(model_fit.__code__.co_varnames) - 1, 0.5))
        print(f"Fitted coefficients: {coeffs}")

    # Plot figure
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle(f'Concentration at time {time} s')

    # Plot concentration vs. density
    ax[0].plot(results_df['density'], results_df['concentration'], marker='o')
    ax[0].set_xlabel('Spore density [$1/mL$]')
    ax[0].set_ylabel('Concentration at spore [M]')
    ax[0].grid()
    if alogx:
        ax[0].set_xscale('log')
    if alogy:
        ax[0].set_yscale('log')
    
    # Plot concentration vs. distance
    ax[1].plot(results_df['spore_dist'], results_df['concentration'], marker='o')
    ax[1].set_xlabel('Spore distance [$\mu m$]')
    ax[1].set_ylabel('Concentration at spore [M]')
    ax[1].grid()
    if blogx:
        ax[1].set_xscale('log')
    if blogy:
        ax[1].set_yscale('log')
    plt.tight_layout()
    plt.show()