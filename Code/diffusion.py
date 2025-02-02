import numpy as np
from numba import jit, cuda, float32

# ===== ANALYTICAL SOLUTIONS =====
def permeation_time_dependent_analytical(c_in, c_out, t, Ps, A, V, alpha=1.0):
    """
    Compute the concentration of a solute in a spore given the initial and external concentrations.
    inputs:
        c_in (float) - the initial concentration at the spore
        c_out (float) - the initial external concentration
        t (float) - time
        Ps (float) - the spore membrane permeation constant
        A (float) - the surface area of the spore
        V (float) - the volume of the spore
        alpha (float) - permeable fraction of the area; defaults to 1
    """
    tau = V / (alpha * A * Ps)
    print(tau)
    c = c_out - (c_out - c_in) * np.exp(-t / tau)
    return c


def diffusion_time_dependent_analytical(dist, c_init, D, time, vol, dims=3):
    """
    Compute the analytical solution of the time-dependent diffusion equation at any point x.
    inputs:
        dist (float) - the distance from the source
        c_init (float) - the initial concentration
        D (float) - the diffusion constant
        time (float) - the time at which the concentration is to be computed
        vol (float) - the volume of the initial concentration source
    """
    if dims == 2:
        result = np.power(vol * c_init, 2/3) / (4*np.pi*D*time) * np.exp(-dist**2 / (4*D*time))
    elif dims == 3:
        result = vol * c_init / np.power(4*np.pi*D*time, 1.5) * np.exp(-dist**2 / (4*D*time))
    return result


def diffusion_time_dependent_analytical_src(c_init, D, time, vol, dims=3):
    """
    Compute the analytical solution of the time-dependent diffusion equation at the source.
    inputs:
        c_init (float) - the initial concentration
        D (float) - the diffusion constant
        time (float) - the time at which the concentration is to be computed
        vol (float) - the volume of the initial concentration source
    """
    if dims == 2:
        result = np.power(vol * c_init, 2/3) / (4*np.pi*D*time)
    elif dims == 3:
        result = vol * c_init / np.power(4*np.pi*D*time, 1.5)
    result[0] = c_init
    return result


def compute_permeation_constant(c_in_target, c_out, c_0, t, A, V, alpha=1.0):
    """
    Compute the permeation constant P_s given the constant external concentration,
    the initial concentration of the solute, the concentration of the solute at time t
    and the surface area and volume of the spore.
    inputs:
        c_in (float) - the concentration at the spore at time t
        c_out (float) - the constant external concentration
        c_0 (float) - the initial concentration of the solute at the spore
        t (float) - time
        A (float) - the surface area of the spore
        V (float) - the volume of the spore
        alpha (float) - permeable fraction of the area; defaults to 1
    """
    Ps = V / (alpha * A * t) * np.log((c_out - c_0) / (c_out - c_in_target))
    return Ps


# ===== NUMERICAL SOLUTIONS =====
def diffusion_time_dependent_sequential(c_init, t_max, D=1.0, Db=1.0, Ps=1.0, dt=0.001, dx=0.005, n_save_frames=100, spore_idx=(None, None, None), c_thresholds=None):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation, using a single point source.
    Uses vectorized operations.
    inputs:
        c_init (numpy.ndarray) - the initial state of the lattice;
        t_max (int) - a maximum number of iterations;
        D (float) - the diffusion constant; defaults to 1;
        Db (float) - the diffusion constant through the spore; defaults to 1;
        Ps (float) - the permeation constant through the spore barrier; defaults to 1;
        dt (float) - timestep; defaults to 0.001;
        dx (float) - spatial increment; defaults to 0.005;
        n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100;
        c_thresholds (float) - threshold values for the concentration; defaults to None.
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time.
    """

    assert c_init.ndim == 2 or c_init.ndim == 3, 'input array must be 2- or 3-dimensional'
    # assert c_init.shape[0] == c_init.shape[1] == c_init.shape[2], 'lattice must have equal size along each dimension'

    dims = c_init.ndim

    # Determine number of lattice rows/columns
    N = c_init.shape[0]

    # Save update factor
    Ddtdx2 = D * dt / (dx ** 2)
    Dbdtdx2 = Db * dt / (dx ** 2)

    # Correction factor for permeation
    if Db is None:
        if dims == 2:
            Db = Ps
        else:
            Db = Ps * dx

    if  Ddtdx2 > 0.5:
        print("Warning: inappropriate scaling of dx and dt, may result in an unstable simulation.")

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))
    print(f"Simulation running for {n_frames} steps on a lattice of size {np.array(c_init.shape) * dx} mm.")

    # Array for storing lattice states
    if dims == 2:
        c_evolution = np.zeros((n_save_frames + 1, N, N))
    else:
        c_evolution = np.zeros((n_save_frames + 1, N, N, N))
    times = np.zeros(n_save_frames + 1)
    save_interval = max(np.floor(n_frames / n_save_frames), 1)
    save_ct = 0

    # Array for storing times at which thresholds are reached
    if type(c_thresholds) == np.ndarray:
        times_thresh = np.zeros(c_thresholds.shape)
    else:
        times_thresh = None
    thresh_ct = 0

    # Initialise current state
    c_curr = np.array(c_init)

    for t in range(n_frames):

        print(f"Frame {t} of {n_frames}", end="\r")

        # Save frame
        if t % save_interval == 0:
            c_evolution[save_ct] = np.array(c_curr)
            times[save_ct] = t * dt
            save_ct += 1

        # Compute next state
        c_curr_bottom = np.roll(c_curr, -1, axis=0)
        c_curr_top = np.roll(c_curr, 1, axis=0)
        c_curr_left = np.roll(c_curr, -1, axis=1)
        c_curr_right = np.roll(c_curr, 1, axis=1)
        c_curr_front = np.roll(c_curr, -1, axis=2)
        c_curr_back = np.roll(c_curr, 1, axis=2)
        
        # General update
        c_next = Ddtdx2 * (c_curr_bottom + c_curr_top + c_curr_left + c_curr_right + c_curr_front + c_curr_back - 6 * c_curr) + c_curr

        # Update around spore
        if spore_idx[0] is not None:
            c_next[spore_idx] = Dbdtdx2 * (c_curr_bottom[spore_idx] + c_curr_top[spore_idx] + c_curr_left[spore_idx] + c_curr_right[spore_idx] + c_curr_front[spore_idx] + c_curr_back[spore_idx] - 6 * c_curr[spore_idx]) + c_curr[spore_idx]
            
            idx_bottom_spore = (spore_idx[0] + 1, spore_idx[1], spore_idx[2])
            idx_top_spore = (spore_idx[0] - 1, spore_idx[1], spore_idx[2])
            idx_left_spore = (spore_idx[0], spore_idx[1] + 1, spore_idx[2])
            idx_right_spore = (spore_idx[0], spore_idx[1] - 1, spore_idx[2])
            idx_front_spore = (spore_idx[0], spore_idx[1], spore_idx[2] + 1)
            idx_back_spore = (spore_idx[0], spore_idx[1], spore_idx[2] - 1)
            
            diff1 = Dbdtdx2 * (c_curr_top[idx_bottom_spore] - c_curr[idx_bottom_spore])
            diff2 = Ddtdx2 * (c_curr_bottom[idx_bottom_spore] + c_curr_left[idx_bottom_spore] + c_curr_right[idx_bottom_spore] + c_curr_front[idx_bottom_spore] + c_curr_bottom[idx_bottom_spore] - 5*c_curr[idx_bottom_spore])
            c_next[idx_bottom_spore] = diff1 + diff2 + c_curr[idx_bottom_spore]

            diff1 = Dbdtdx2 * (c_curr_bottom[idx_top_spore] - c_curr[idx_top_spore])
            diff2 = Ddtdx2 * (c_curr_top[idx_top_spore] + c_curr_left[idx_top_spore] + c_curr_right[idx_top_spore] + c_curr_front[idx_top_spore] + c_curr_back[idx_top_spore] - 5*c_curr[idx_top_spore])
            c_next[idx_top_spore] = diff1 + diff2 + c_curr[idx_top_spore]

            diff1 = Dbdtdx2 * (c_curr_right[idx_left_spore] - c_curr[idx_left_spore])
            diff2 = Ddtdx2 * (c_curr_bottom[idx_left_spore] + c_curr_top[idx_left_spore] + c_curr_left[idx_left_spore] + c_curr_front[idx_left_spore] + c_curr_back[idx_left_spore] - 5*c_curr[idx_left_spore])
            c_next[idx_left_spore] = diff1 + diff2 + c_curr[idx_left_spore]

            diff1 = Dbdtdx2 * (c_curr_left[idx_right_spore] - c_curr[idx_right_spore])
            diff2 = Ddtdx2 * (c_curr_bottom[idx_right_spore] + c_curr_top[idx_right_spore] + c_curr_right[idx_right_spore] + c_curr_front[idx_right_spore] + c_curr_back[idx_right_spore] - 5*c_curr[idx_right_spore])
            c_next[idx_right_spore] = diff1 + diff2 + c_curr[idx_right_spore]

            diff1 = Dbdtdx2 * (c_curr_back[idx_front_spore] - c_curr[idx_front_spore])
            diff2 = Ddtdx2 * (c_curr_bottom[idx_front_spore] + c_curr_top[idx_front_spore] + c_curr_left[idx_front_spore] + c_curr_right[idx_front_spore] + c_curr_front[idx_front_spore] - 5*c_curr[idx_front_spore])
            c_next[idx_front_spore] = diff1 + diff2 + c_curr[idx_front_spore]

            diff1 = Dbdtdx2 * (c_curr_front[idx_back_spore] - c_curr[idx_back_spore])
            diff2 = Ddtdx2 * (c_curr_bottom[idx_back_spore] + c_curr_top[idx_back_spore] + c_curr_left[idx_back_spore] + c_curr_right[idx_back_spore] + c_curr_back[idx_back_spore] - 5*c_curr[idx_back_spore])
            c_next[idx_back_spore] = diff1 + diff2 + c_curr[idx_back_spore]

        
        # Update current array
        c_curr = np.array(c_next)

        # Save time if threshold is reached
        if c_thresholds is not None:
            if thresh_ct < times_thresh.shape[0] and c_thresholds[thresh_ct] == 0 and np.max(c_curr) < c_thresholds[thresh_ct]:
                times_thresh[thresh_ct] = t * dt
                thresh_ct += 1

    # Save final frame
    c_evolution[-1] = np.array(c_curr)
    times[-1] = t_max

    return c_evolution, times, times_thresh


def invoke_smart_kernel_2D(size, threads_per_block=(16, 16)):
    """
    Invoke a kernel with the appropriate number of blocks and threads per block.
    """
    blocks_per_grid = [(size + (tpb - 1)) // tpb for tpb in threads_per_block]
    return tuple(blocks_per_grid), tuple(threads_per_block)


def invoke_smart_kernel_3D(size, threads_per_block=(8, 8, 8)):
    """
    Invoke a kernel with the appropriate number of blocks and threads per block.
    """
    blocks_per_grid = [(size + (tpb - 1)) // tpb for tpb in threads_per_block]
    return tuple(blocks_per_grid), tuple(threads_per_block)


@cuda.jit()
def update_GPU_2D(c_old, c_new, N, dtdx2, D, Db, spore_idx):
    """
    Update the concentration of a 2D lattice point based on the time-dependent diffusion equation with a periodic boundary.
    Uses CUDA to parallelize the computation.
    inputs:
        c_old (DeviceNDArray) - the current state of the lattice
        c_new (DeviceNDArray) - the next state of the lattice
        N (int) - the size of the lattice
        dtdx2 (float) - the update factor
        D (float) - the diffusion constant through the medium
        Db (float) - the diffusion constant through the spore barrier
        spore_idx (tuple) - the indices of the spore location
    """
    i, j = cuda.grid(2)

    if i >= c_old.shape[0] or j >= c_old.shape[1]:
        return

    center = c_old[i, j]
    bottom = c_old[(i - 1) % N, j]
    top = c_old[(i + 1) % N, j]
    left = c_old[i, (j - 1) % N]
    right = c_old[i, (j + 1) % N]

    Ddtdx20 = D * dtdx2
    Ddtdx21 = D * dtdx2
    Ddtdx22 = D * dtdx2
    Ddtdx23 = D * dtdx2
    
    if i == spore_idx[0] and j == spore_idx[1]:
        Ddtdx20 = Db * dtdx2
        Ddtdx21 = Db * dtdx2
        Ddtdx22 = Db * dtdx2
        Ddtdx23 = Db * dtdx2
    elif i == spore_idx[0] - 1 and j == spore_idx[1]:
        Ddtdx21 = Db * dtdx2
    elif i == spore_idx[0] + 1 and j == spore_idx[1]:
        Ddtdx20 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] - 1:
        Ddtdx23 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] + 1:
        Ddtdx22 = Db * dtdx2

    diff_sum = Ddtdx20 * bottom + Ddtdx21 * top + Ddtdx22 * left + Ddtdx23 * right
    c_new[i, j] = center + diff_sum - (Ddtdx20 + Ddtdx21 + Ddtdx22 + Ddtdx23) * center


@cuda.jit()
def update_GPU_3D(c_old, c_new, N, dtdx2, D, Db, spore_idx):
    """
    Update the concentration of a 3D lattice point based on the time-dependent diffusion equation with a periodic boundary.
    Uses CUDA to parallelize the computation.
    inputs:
        c_old (DeviceNDArray) - the current state of the lattice
        c_new (DeviceNDArray) - the next state of the lattice
        N (int) - the size of the lattice
        dtdx2 (float) - the update factor
        D (float) - the diffusion constant through the medium
        Db (float) - the diffusion constant through the spore barrier
        spore_idx (tuple) - the indices of the spore location
    """
    i, j, k = cuda.grid(3)

    if i >= c_old.shape[0] or j >= c_old.shape[1] or k >= c_old.shape[2]:
        return
    
    # if k == 0 or k == c_old.shape[2] - 1:
    #     return

    center = c_old[i, j, k]
    bottom = c_old[(i - 1) % N, j, k]
    top = c_old[(i + 1) % N, j, k]
    left = c_old[i, (j - 1) % N, k]
    right = c_old[i, (j + 1) % N, k]
    front = c_old[i, j, (k - 1) % N]
    back = c_old[i, j, (k + 1) % N]

    # Neumann boundary at top and bottom
    # if k == 0:
    #     front = c_old[i, j, 1]
    # elif k == c_old.shape[2] - 1:
    #     back = c_old[i, j, c_old.shape[2] - 2]

    Ddtdx20 = D * dtdx2
    Ddtdx21 = D * dtdx2
    Ddtdx22 = D * dtdx2
    Ddtdx23 = D * dtdx2
    Ddtdx24 = D * dtdx2
    Ddtdx25 = D * dtdx2
    
    if i == spore_idx[0] and j == spore_idx[1] and k == spore_idx[2]:
        Ddtdx20 = Db * dtdx2
        Ddtdx21 = Db * dtdx2
        Ddtdx22 = Db * dtdx2
        Ddtdx23 = Db * dtdx2
        Ddtdx24 = Db * dtdx2
        Ddtdx25 = Db * dtdx2
    elif i == spore_idx[0] - 1 and j == spore_idx[1] and k == spore_idx[2]:
        Ddtdx21 = Db * dtdx2
    elif i == spore_idx[0] + 1 and j == spore_idx[1] and k == spore_idx[2]:
        Ddtdx20 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] - 1 and k == spore_idx[2]:
        Ddtdx23 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] + 1 and k == spore_idx[2]:
        Ddtdx22 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] and k == spore_idx[2] - 1:
        Ddtdx25 = Db * dtdx2
    elif i == spore_idx[0] and j == spore_idx[1] and k == spore_idx[2] + 1:
        Ddtdx24 = Db * dtdx2

    diff_sum = Ddtdx20 * bottom + Ddtdx21 * top + Ddtdx22 * left + Ddtdx23 * right + Ddtdx24 * front + Ddtdx25 * back
    c_new[i, j, k] = center + diff_sum - (Ddtdx20 + Ddtdx21 + Ddtdx22 + Ddtdx23 + Ddtdx24 + Ddtdx25) * center


# @cuda.jit()
# def update_GPU_3D_periodic_spores(c_old, c_new, N, dtdx2, D, Db, spore_spacing):
#     """
#     Update the concentration of a 3D lattice point based on the time-dependent diffusion equation
#     with a periodic boundary and spores spaced regularly in all dimensions.
#     Uses CUDA to parallelize the computation.
#     inputs:
#         c_old (DeviceNDArray) - the current state of the lattice
#         c_new (DeviceNDArray) - the next state of the lattice
#         N (int) - the size of the lattice
#         dtdx2 (float) - the update factor
#         D (float) - the diffusion constant through the medium
#         Db (float) - the diffusion constant through the spore barrier
#         spore_spacing (int) - the spacing between spores
#     """

#     i, j, k = cuda.grid(3)

#     if i >= c_old.shape[0] or j >= c_old.shape[1] or k >= c_old.shape[2]:
#         return
    
#     center = c_old[i, j, k]
#     bottom = c_old[(i - 1) % N, j, k]
#     top = c_old[(i + 1) % N, j, k]
#     left = c_old[i, (j - 1) % N, k]
#     right = c_old[i, (j + 1) % N, k]
#     front = c_old[i, j, (k - 1) % N]
#     back = c_old[i, j, (k + 1) % N]

#     # Neumann boundary at top and bottom
#     # if neumann:
#     #     if k == 0:
#     #         front = c_old[i, j, 1]
#     #     elif k == H - 1:
#     #         back = c_old[i, j, H - 2]

#     Ddtdx20 = D * dtdx2
#     Ddtdx21 = D * dtdx2
#     Ddtdx22 = D * dtdx2
#     Ddtdx23 = D * dtdx2
#     Ddtdx24 = D * dtdx2
#     Ddtdx25 = D * dtdx2

#     if i % spore_spacing == 0 and j % spore_spacing == 0 and k % spore_spacing == 0:
#         Ddtdx20 = Db * dtdx2
#         Ddtdx21 = Db * dtdx2
#         Ddtdx22 = Db * dtdx2
#         Ddtdx23 = Db * dtdx2
#         Ddtdx24 = Db * dtdx2
#         Ddtdx25 = Db * dtdx2
#     elif i % spore_spacing == spore_spacing - 1 and j % spore_spacing == 0 and k % spore_spacing == 0:
#         Ddtdx21 = Db * dtdx2
#     elif i % spore_spacing == 1 and j % spore_spacing == 0 and k % spore_spacing == 0:
#         Ddtdx20 = Db * dtdx2
#     elif i % spore_spacing == 0 and j % spore_spacing == spore_spacing - 1 and k % spore_spacing == 0:
#         Ddtdx23 = Db * dtdx2
#     elif i % spore_spacing == 0 and j % spore_spacing == 1 and k % spore_spacing == 0:
#         Ddtdx22 = Db * dtdx2
#     elif i % spore_spacing == 0 and j % spore_spacing == 0 and k % spore_spacing == spore_spacing - 1:
#         Ddtdx25 = Db * dtdx2
#     elif i % spore_spacing == 0 and j % spore_spacing == 0 and k % spore_spacing == 1:
#         Ddtdx24 = Db * dtdx2

#     diff_sum = Ddtdx20 * bottom + Ddtdx21 * top + Ddtdx22 * left + Ddtdx23 * right + Ddtdx24 * front + Ddtdx25 * back
#     c_new[i, j, k] = center + diff_sum - (Ddtdx20 + Ddtdx21 + Ddtdx22 + Ddtdx23 + Ddtdx24 + Ddtdx25) * center


@cuda.jit()
def update_GPU_3D_periodic_spores_bottom(c_old, c_new, H, dtdx2, D, Db, N):
    """
    Update the concentration of a 3D lattice point based on the time-dependent diffusion equation
    with a periodic boundary and spores spaced regularly in all dimensions.
    Uses CUDA to parallelize the computation.
    inputs:
        c_old (DeviceNDArray) - the current state of the lattice
        c_new (DeviceNDArray) - the next state of the lattice
        H (int) - the height of the lattice
        dtdx2 (float) - the update factor
        D (float) - the diffusion constant through the medium
        Db (float) - the diffusion constant through the spore barrier
        N (int) - the size of the lattice
    """

    i, j, k = cuda.grid(3)

    if i >= c_old.shape[0] or j >= c_old.shape[1] or k >= c_old.shape[2]:
        return
    
    center = c_old[i, j, k]
    bottom = c_old[(i - 1) % N, j, k]
    top = c_old[(i + 1) % N, j, k]
    left = c_old[i, (j - 1) % N, k]
    right = c_old[i, (j + 1) % N, k]
    front = c_old[i, j, (k - 1) % H]
    back = c_old[i, j, (k + 1) % H]

    # Neumann boundary at top and bottom
    if k == 0:
        front = c_old[i, j, 1]
    elif k == H - 1:
        back = c_old[i, j, H - 2]

    Ddtdx20 = D * dtdx2
    Ddtdx21 = D * dtdx2
    Ddtdx22 = D * dtdx2
    Ddtdx23 = D * dtdx2
    Ddtdx24 = D * dtdx2
    Ddtdx25 = D * dtdx2
    
    if i == N // 2 and j == N // 2 and k == 0:
        Ddtdx20 = Db * dtdx2
        Ddtdx21 = Db * dtdx2
        Ddtdx22 = Db * dtdx2
        Ddtdx23 = Db * dtdx2
        Ddtdx24 = Db * dtdx2
        Ddtdx25 = Db * dtdx2
    elif i == N // 2 - 1 and j == N // 2 and k == 0:
        Ddtdx21 = Db * dtdx2
    elif i == N // 2 + 1 and j == N // 2 and k == 0:
        Ddtdx20 = Db * dtdx2
    elif i == N // 2 and j == N // 2 - 1 and k == 0:
        Ddtdx23 = Db * dtdx2
    elif i == N // 2 and j == N // 2 + 1 and k == 0:
        Ddtdx22 = Db * dtdx2
    elif i == N // 2 and j == N // 2 and k == 1:
        Ddtdx24 = Db * dtdx2

    diff_sum = Ddtdx20 * bottom + Ddtdx21 * top + Ddtdx22 * left + Ddtdx23 * right + Ddtdx24 * front + Ddtdx25 * back
    c_new[i, j, k] = center + diff_sum - (Ddtdx20 + Ddtdx21 + Ddtdx22 + Ddtdx23 + Ddtdx24 + Ddtdx25) * center


@cuda.reduce
def max_reduce(a, b):
    """
    Find the maximum of two values.
    """
    if a > b:
        return a
    else:
        return b


def diffusion_time_dependent_GPU(c_init, t_max, D=1.0, Db=1.0, Ps=1.0, dt=0.005, dx=5, n_save_frames=100,
                                 spore_idx=(None, None, None), c_thresholds=None, bottom_arrangement=False, c_cutoff=None):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation.
    inputs:
        c_init (numpy.ndarray) - the initial state of the lattice
        t_max (int) - a maximum number of iterations
        D (float) - the diffusion constant; defaults to 1
        Db (float) - the diffusion constant through the spore; defaults to 1
        Ps (float) - the permeation constant through the spore barrier; defaults to 1
        dt (float) - timestep; defaults to 0.001
        dx (float) - spatial increment; defaults to 0.005
        n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100
        spore_idx (tuple) - the indices of the spore location; defaults to (None, None)
        c_thresholds (float) - threshold values for the concentration; defaults to None
        bottom_arrangement (bool) - whether the spores are arranged at the bottom of the lattice; defaults to False
        c_cutoff (float): the concentration threshold at which to terminate the simulation; defaults to None
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time
    """

    assert c_init.ndim == 2 or c_init.ndim == 3, 'input array must be 2- or 3-dimensional'

    # Determine dimensionality of the lattice
    dims = c_init.ndim
    N = c_init.shape[0]
    if dims == 2:
        H = 0
        update_func = update_GPU_2D
        size_ref = N
        spore_ref = spore_idx
        print("2D simulation")
    else:
        H = c_init.shape[2]
        if bottom_arrangement:
            update_func = update_GPU_3D_periodic_spores_bottom
            size_ref = H
            spore_ref = N - 1
            print("3D simulation with 2D periodic spores")
        else:
            update_func = update_GPU_3D
            size_ref = N
            spore_ref = spore_idx
            print("3D simulation with 3D periodic spores")

    # Save update factor
    dtdx2 = dt / (dx ** 2)
    
    # Correction factor for permeation
    if Db is None:
        if dims == 2:
            Db = Ps
        else:
            Db = Ps * dx
    # elif dims == 3:
    #     Db *= dx
    print(f"Using D = {D}, Db = {Db}, Ps = {Ps}")

    if  D * dtdx2 > 0.5:
        print("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation.")

    if  Db * dtdx2 > 0.5:
        print("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation.")

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))
    print(f"Simulation running for {n_frames} steps on a lattice of size {np.array(c_init.shape) * dx} microns.")

    # Array for storing lattice states
    if dims == 2:
        c_evolution = np.zeros((n_save_frames + 1, N, N))
    else:
        c_evolution = np.zeros((n_save_frames + 1, N, N, H))
    times = np.zeros(n_save_frames + 1)
    save_interval = np.floor(n_frames / n_save_frames)
    save_ct = 0
    
    # Array for storing times at which thresholds are reached
    if type(c_thresholds) == np.ndarray:
        times_thresh = np.zeros(c_thresholds.shape)
    else:
        times_thresh = None
    thresh_ct = 0

    # Initialise lattice states
    c_A_gpu = cuda.to_device(c_init)
    c_B_gpu = cuda.to_device(np.zeros_like(c_init))

    if dims == 2:
        kernel_blocks, kernel_threads = invoke_smart_kernel_2D(N)
    else:
        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(N)

    # Run simulation
    for t in range(n_frames):

        # print(f"Frame {t} of {n_frames}", end="\r")

        # Save frame
        if t % save_interval == 0:
            c_evolution[save_ct] = c_A_gpu.copy_to_host()
            times[save_ct] = t * dt
            save_ct += 1
        
        update_func[kernel_blocks, kernel_threads](c_A_gpu, c_B_gpu, size_ref, dtdx2, D, Db, spore_ref)

        # Synchronize the GPU to ensure the kernel has finished
        cuda.synchronize()
        
        c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu

        # Save time if threshold is reached
        if c_thresholds is not None:
            if thresh_ct < times_thresh.shape[0] and times_thresh[thresh_ct] == 0 and max_reduce(c_A_gpu.ravel()) < c_thresholds[thresh_ct]:
                times_thresh[thresh_ct] = t * dt
                thresh_ct += 1
        
        # Check if concentration has reached the cutoff
        if c_cutoff is not None:
            if max_reduce(c_A_gpu.ravel()) < c_cutoff:
                # print(f"Concentration has reached {c_cutoff} at frame {t}.")
                break

    # Save final frame
    c_evolution[(save_ct, ...)] = c_A_gpu.copy_to_host()
    times[save_ct] = t_max

    return c_evolution, times, times_thresh