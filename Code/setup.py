import numpy as np
import conversions as conv
import diffusion as diff
import pandas as pd
import os

def setup_lattice(N, H, spores_x, spores_y, spores_z, c_spore_init=1):
    """
    Set up the lattice with spores at the given locations.
    inputs:
        N (int): the size of the bottom of the lattice
        H (int): the height of the lattice
        spores_x (numpy array): the x coordinates of the spores
        spores_y (numpy array): the y coordinates of the spores
        spores_z (numpy array): the z coordinates of the spores
        c_spore_init (float): the initial concentration at the spores
    outputs:
        c_lattice (numpy array): the initial concentration lattice
    """

    assert spores_x.size == spores_y.size == spores_z.size, "spores_x, spores_y, and spores_z must have the same size"

    c_lattice = np.zeros((N+1, N+1, H+1), dtype=np.float64)
    c_lattice[spores_x, spores_y, spores_z] = c_spore_init

    return c_lattice


def lattice_size_from_density(spore_density, dx, H=None):
    """
    Compute the lattice size from the spore density.
    inputs:
        spore_density (float): the density of spores in micrometers^-3
        dx (float): the lattice spacing in micrometers
        H (int): the height of the lattice; if specified, only the bottom is populated
    outputs:
        N (int): the size of the lattice in each dimension
    """

    if H is None:
        L = np.cbrt(1 / spore_density)
        N = int(np.ceil(L / dx))
    else:
        L = np.sqrt(1 / (dx * spore_density))
        N = int(np.ceil(L / dx))

    return N


def populate_spore_grid_coords(N, dx, spore_density, H=None, start_height=1):
    """
    Generate spore grid coordinates for a given spore density.
    inputs:
        N (int): the size of the bottom of the lattice
        dx (float): the lattice spacing in micrometers
        spore_density (float): the density of spores in micrometers^-3
        H (int): the height of the lattice; if specified, only the bottom is populated
        start_height (int): the distance of the spores from the bottom (if only the bottom is populated)
    outputs:
        spore_coords (numpy array): the coordinates of the spores
        spore_spacing (float): the spacing between spores in micrometers
    """

    N = N + 1

    if H is None:
        bottom_only = False
    else:
        bottom_only = True
    
    if bottom_only:

        # Calculate the number of spores to place
        V_grid = N**2 * H * dx**3
        n_spores = spore_density * V_grid

        V_occupied = N**2 * dx**3
        n_spores_1D = int(np.sqrt(n_spores))
        print(f"Effective density: {n_spores / V_occupied} spores/micrometer^3")
    else:

        # Calculate the number of spores to place
        V_grid = N**3 * dx**3
        n_spores = spore_density * V_grid

        V_occupied = V_grid
        n_spores_1D = int(np.cbrt(n_spores))
        print(f"Effective density: {n_spores / V_occupied} spores/micrometer^3")

    L = N * dx
    spore_spacing = L / n_spores_1D

    print(f"Populating volume of {V_occupied} micrometers^3 with {n_spores} spores, {n_spores_1D} spores per dimension")
    print(f"Spore spacing: {spore_spacing} micrometers")

    # Generate the spore grid coordinates
    spores_x = np.arange(0, N+1, spore_spacing / dx)
    spores_y = np.arange(0, N+1, spore_spacing / dx)
    if bottom_only:
        spores_z = np.array([start_height / dx])
    else:
        spores_z = np.arange(0, N+1, spore_spacing / dx)

    # Crete a meshgrid of the spore coordinates
    spores_x, spores_y, spores_z = np.meshgrid(spores_x, spores_y, spores_z, indexing='ij')

    spore_coords = np.array([spores_x.flatten(), spores_y.flatten(), spores_z.flatten()]).T

    return spore_coords, spore_spacing


def run_diffusion_experiments_single_spore(exp_params, t_max, N, dt, dx, n_save_frames, V_spore, c_thresh_factors=None):
    """
    Run diffusion experiments with a single spore source.
    inputs:
        exp_params (dict): the parameters of the experiment
        t_max (float): the maximum time of the experiment
        N (int): the size of the lattice in each dimension
        dt (float): the time step of the simulation
        dx (float): the lattice spacing in micrometers
        n_save_frames (int): the number of frames to save
        V_spore (float): the volume of the spore in micrometers^3
        c_thresh_factors (list): concentration reduction factor thresholds to save times at
    """

    sim_results_global = pd.DataFrame()
    sim_results_list = []
    exp_params_data = pd.DataFrame(exp_params)

    # Add new parameters
    exp_params_data['N'] = N
    exp_params_data['dt'] = dt
    exp_params_data['dx'] = dx
    exp_params_data['A_spore'] = None
    exp_params_data['V_spore'] = V_spore

    # Create folder for data
    if not os.path.exists('Data'):
        os.makedirs('Data')

    for i, params in enumerate(exp_params):

        exp_id = params['expID']
        sim_id = params['simID']
        label = params['label']
        dims = params['dims']
        Db = params['D']
        Ps = params['Ps']
        c0 = params['c0']

        print(f"{sim_id}: Running simulation {label}")

        assert dims in [2, 3], "dims must be either '2D' or '3D'"

        if dims == 2:
            c_lattice = np.zeros((N+1, N+1), dtype=np.float64)
            spore_idx = (N // 2, N // 2)
            A_spore = np.cbrt(V_spore) * 4
        elif dims == 3:
            c_lattice = np.zeros((N+1, N+1, N+1), dtype=np.float64)
            spore_idx = (N // 2, N // 2, N // 2)
            A_spore = (np.cbrt(V_spore) ** 2) * 6
        
        exp_params_data.loc[i, 'A_spore'] = A_spore

        # Default D in medium
        D = 600 # micrometers^2/s

        # Initialise concentration at spore
        c_spore_init = 1
        c_lattice[spore_idx] = c_spore_init

        c_thresholds = c_thresh_factors * c_spore_init

        # Run numerical experiment
        c_evolution, times, times_thresh = diff.diffusion_time_dependent_GPU(c_lattice, t_max, D, Db, Ps, dt, dx, n_save_frames, spore_idx, None, c_thresholds)
        c_numerical = c_evolution[:, spore_idx[0], spore_idx[1]] if dims == 2 else c_evolution[:, spore_idx[0], spore_idx[1], spore_idx[2]]

        # Compute analytical solution
        if D == Db:
            # print(f"Db={Db}")
            c_analytical = diff.diffusion_time_dependent_analytical_src(c_spore_init, D, times, V_spore, dims)
        elif Ps is not None:
            # print(f"Ps={Ps}, A_spore={A_spore}, V_spore={V_spore}")
            c_analytical = diff.permeation_time_dependent_analytical(c_spore_init, 0, times, Ps, A_spore, V_spore)
        # else:
        #     print("foo")
        #     Ps = conv.convert_D_to_Ps(Db, 1, 1)
        #     c_analytical = diff.permeation_time_dependent_analytical(c_spore_init, 0, times, Ps, A_spore, V_spore)

        # Save results
        np.save(f"Data/{exp_id}_{sim_id}_frames.npy", c_evolution)
        sim_results = pd.DataFrame({'time': times, 'c_numerical': c_numerical*c0, 'c_analytical': c_analytical*c0, 'spore_idx': [spore_idx] * len(times)})

        # Add column for ID and threshold times
        sim_results['simID'] = sim_id
        sim_results['label'] = label
        sim_results['times_thresh'] = list(np.tile(times_thresh, (len(sim_results), 1)))
        sim_results['c_thresh'] = list(np.tile(c_thresh_factors*c0, (len(sim_results), 1)))

        sim_results_list.append(sim_results)
    
    sim_results_global = pd.concat(sim_results_list)
    
    # Write parameters and results to file
    exp_params_data.to_csv(f'Data/{exp_params[0]['expID']}_exp_params.csv')
    sim_results_global.to_csv(f'Data/{exp_params[0]['expID']}_sim_results.csv')


def run_diffusion_experiments_multi_spore(exp_params, t_max, dt, dx, n_save_frames, V_spore, c_thresh_factors=None, H=None, c_cutoff=None):
    """
    Run diffusion experiments with multiple uniformly spaced spore sources
    on a 3D lattice with periodic boundaries.
    inputs:
        exp_params (dict): the parameters of the experiment
        t_max (float): the maximum time of the experiment
        N (int): the size of the lattice in each dimension
        dt (float): the time step of the simulation
        dx (float): the lattice spacing in micrometers
        n_save_frames (int): the number of frames to save
        V_spore (float): the volume of the spore in micrometers^3
        c_thresh_factors (list): concentration reduction factor thresholds to save times at
        H (int): the height of the lattice, if specified, a 2D lattice array at the bottom is implied
        c_cutoff (float): the concentration threshold at which to terminate the simulation
    """

    sim_results_global = pd.DataFrame()
    sim_results_list = []
    exp_params_data = pd.DataFrame(exp_params)

    # Add new parameters
    exp_params_data['dt'] = dt
    exp_params_data['dx'] = dx
    exp_params_data['A_spore'] = None
    exp_params_data['V_spore'] = V_spore
    exp_params_data['c_cutoff'] = c_cutoff

    # Create folder for data
    if not os.path.exists('Data'):
        os.makedirs('Data')
    
    for i, params in enumerate(exp_params):

        exp_id = params['expID']
        sim_id = params['simID']
        label = params['label']
        Db = params['D']
        Ps = params['Ps']
        c0 = params['c0']
        spore_density = params['spore_density']

        # Infer lattice size from spore density
        spore_density = conv.inverse_mL_to_micrometers_cubed(spore_density)
        if H is None:
            N = lattice_size_from_density(spore_density, dx)
        else:
            N = lattice_size_from_density(spore_density, dx, H)

        print(f"{sim_id}: Running simulation {label} on lattice with size {N}")

        # Compute spore area
        A_spore = (np.cbrt(V_spore) ** 2) * 6
        exp_params_data.loc[i, 'A_spore'] = A_spore

        # Default D in medium
        D = 600 # micrometers^2/s

        # Initialise concentration at spores
        c_spore_init = 1
        c_lattice = np.zeros((N+1, N+1, N+1), dtype=np.float64)
        c_thresholds = c_thresh_factors * c_spore_init

         # Normalize cutoff concentration
        c_cutoff_norm = c_cutoff / c0

        # Reference spores for measurements
        spore_idx = (N // 2, N // 2, N // 2)
        c_lattice[spore_idx] = c_spore_init

        # Run numerical experiment
        if H is None:
            bottom_arrangement = False
        else:
            bottom_arrangement = True
        c_evolution, times, times_thresh = diff.diffusion_time_dependent_GPU(c_lattice, t_max, D, Db, Ps, dt, dx, n_save_frames, spore_idx, None,
                                                                             c_thresholds, bottom_arrangement, c_cutoff_norm)
        c_numerical = c_evolution[:, spore_idx[0], spore_idx[1], spore_idx[2]]

        # Save results
        np.save(f"Data/{exp_id}_{sim_id}_frames.npy", c_evolution)
        sim_results = pd.DataFrame({'time': times, 'c_numerical': c_numerical*c0, 'c_analytical': c_numerical*c0, 'spore_idx': [spore_idx] * len(times)}) # Correct this if analytical solution is available!~

        # Add column for ID and threshold times
        sim_results['simID'] = sim_id
        sim_results['label'] = label
        sim_results['N'] = N
        sim_results['H'] = H
        sim_results['times_thresh'] = list(np.tile(times_thresh, (len(sim_results), 1)))
        sim_results['c_thresh'] = list(np.tile(c_thresh_factors*c0, (len(sim_results), 1)))

        sim_results_list.append(sim_results)

    sim_results_global = pd.concat(sim_results_list)
    
    # Write parameters and results to file
    exp_params_data.to_csv(f'Data/{exp_params[0]['expID']}_exp_params.csv')
    sim_results_global.to_csv(f'Data/{exp_params[0]['expID']}_sim_results.csv')


# def run_diffusion_experiments_multi_spore_bottom(exp_params, t_max, dt, dx, H, n_save_frames, V_spore, c_thresh_factors=None):
#     """
#     Run diffusion experiments with multiple uniformly spaced spore sources
#     at the bottom a 3D lattice with periodic boundaries in x and y
#     and von Neumann boundaries in z.
#     inputs:
#         exp_params (dict): the parameters of the experiment
#         t_max (float): the maximum time of the experiment
#         N (int): the size of the lattice in each dimension
#         dt (float): the time step of the simulation
#         dx (float): the lattice spacing in micrometers
#         H (int): the number of lattice points in the z direction
#         n_save_frames (int): the number of frames to save
#         V_spore (float): the volume of the spore in micrometers^3
#         c_thresh_factors (list): concentration reduction factor thresholds to save times at
#     """

#     sim_results_global = pd.DataFrame()
#     sim_results_list = []
#     exp_params_data = pd.DataFrame(exp_params)

#     # Add new parameters
#     exp_params_data['dt'] = dt
#     exp_params_data['dx'] = dx
#     exp_params_data['A_spore'] = None
#     exp_params_data['V_spore'] = V_spore

#     # Create folder for data
#     if not os.path.exists('Data'):
#         os.makedirs('Data')
    
#     for i, params in enumerate(exp_params):

#         exp_id = params['expID']
#         sim_id = params['simID']
#         label = params['label']
#         Db = params['D']
#         Ps = params['Ps']
#         c0 = params['c0']
#         spore_density = params['spore_density']

#         # Infer lattice size from spore density
#         spore_density = conv.inverse_mL_to_micrometers_cubed(spore_density)
#         N = lattice_size_from_density(spore_density, dx, H)

#         # spore_idx_spacing = spore_spacing / dx

#         print(f"{sim_id}: Running simulation {label} on lattice with size {N} and height {H}")

#         # Compute spore area
#         A_spore = (np.cbrt(V_spore) ** 2) * 6
#         exp_params_data.loc[i, 'A_spore'] = A_spore

#         # Default D in medium
#         D = 600 # micrometers^2/s

#         # Initialise concentration at spores
#         c_spore_init = 1
#         c_lattice = np.zeros((N+1, N+1, H+1), dtype=np.float64)
#         c_thresholds = c_thresh_factors * c_spore_init

#         # Reference spores for measurements
#         spore_idx = (N // 2, N // 2, 1)
#         c_lattice[spore_idx] = c_spore_init

#         # Run numerical experiment
#         if H is None:
#             bottom_arrangement = False
#         else:
#             bottom_arrangement = True
#         c_evolution, times, times_thresh = diff.diffusion_time_dependent_GPU(c_lattice, t_max, D, Db, Ps, dt, dx, n_save_frames, spore_idx, None, c_thresholds, bottom_arrangement)
#         c_numerical = c_evolution[:, spore_idx[0], spore_idx[1], spore_idx[2]]

#         # Save results
#         np.save(f"Data/{exp_id}_{sim_id}_frames.npy", c_evolution)
#         sim_results = pd.DataFrame({'time': times, 'c_numerical': c_numerical*c0, 'c_analytical': c_numerical*c0, 'spore_idx': [spore_idx] * len(times)}) # Correct this if analytical solution is available!~

#         # Add column for ID and threshold times
#         sim_results['simID'] = sim_id
#         sim_results['label'] = label
#         sim_results['N'] = N
#         sim_results['times_thresh'] = list(np.tile(times_thresh, (len(sim_results), 1)))
#         sim_results['c_thresh'] = list(np.tile(c_thresh_factors*c0, (len(sim_results), 1)))

#         sim_results_list.append(sim_results)

#     sim_results_global = pd.concat(sim_results_list)
    
#     # Write parameters and results to file
#     exp_params_data.to_csv(f'Data/{exp_params[0]['expID']}_exp_params.csv')
#     sim_results_global.to_csv(f'Data/{exp_params[0]['expID']}_sim_results.csv')