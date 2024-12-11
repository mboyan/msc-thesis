import numpy as np
from numba import jit, cuda, float32

# ===== ANALYTICAL SOLUTIONS =====
def permeation_time_dependent_analytical(c_in, c_out, t, Ps, A, V, alpha=1.0):
    """
    Compute the concentration of a solute in a spore given the initial and external concentrations.
    inputs:
        c_in (float) - the initial concentration of the solute;
        c_out (float) - the external concentration of the solute;
        t (float) - time;
        Ps (float) - the spore membrane permeation constant;
        A (float) - the surface area of the spore;
        V (float) - the volume of the spore;
        alpha (float) - permeable fraction of the area; defaults to 1.
    """
    tau = V / (alpha * A * Ps)
    c = c_out - (c_out - c_in) * np.exp(-t / tau)
    return c

def diffusion_time_dependent_analytical_src(c_init, D, time, vol):
    """
    Compute the analytical solution of the time-dependent diffusion equation at the source.
    inputs:
        c_init (float) - the initial concentration;
        D (float) - the diffusion constant;
        time (float) - the time at which the concentration is to be computed;
        vol (float) - the volume of the initial concentration cell.
    """
    return vol * c_init / np.power(4*np.pi*D*time, 1.5)

def compute_permeation_constant(c_in, c_out, c_0, t, A, V, alpha=1.0):
    """
    Compute the permeation constant P_s given the constant external concentration,
    the initial concentration of the solute, the concentration of the solute at time t
    and the surface area and volume of the spore.
    inputs:
        c_in (float) - the initial concentration of the solute;
        c_out (float) - the external concentration of the solute;
        c_0 (float) - the concentration of the solute at time t;
        t (float) - time;
        A (float) - the surface area of the spore;
        V (float) - the volume of the spore;
        alpha (float) - permeable fraction of the area; defaults to 1.
    """
    Ps = V / (alpha * A * t) * np.log((c_out - c_0) / (c_out - c_in))
    return Ps


# ===== NUMERICAL SOLUTIONS =====
def invoke_smart_kernel(size, threads_per_block=(8, 8, 8)):
    """
    Invoke a kernel with the appropriate number of blocks and threads per block.
    """
    blocks_per_grid = [(size + (tpb - 1)) // tpb for tpb in threads_per_block]
    return tuple(blocks_per_grid), tuple(threads_per_block)


@cuda.jit()
def update_GPU(c_old, c_new, N, dtdx2, D, Db, spore_idx):
    """
    Update the concentration of a lattice point based on the time-dependent diffusion equation with a periodic boundary.
    inputs:
        c_old (DeviceNDArray) - the current state of the lattice;
        c_new (DeviceNDArray) - the next state of the lattice;
        dtdx2 (float) - the update factor;
        D (float) - the diffusion constant through the medium;
        Db (float) - the diffusion constant through the spore barrier;
        spore_idx (tuple) - the indices of the spore location.
    """
    i, j, k = cuda.grid(3)

    if i >= c_old.shape[0] or j >= c_old.shape[1] or k >= c_old.shape[2]:
        return
    
    if k == 0 or k == c_old.shape[2] - 1:
        return

    center = c_old[i, j, k]
    bottom = c_old[(i - 1) % N, j, k]
    top = c_old[(i + 1) % N, j, k]
    left = c_old[i, (j - 1) % N, k]
    right = c_old[i, (j + 1) % N, k]
    front = c_old[i, j, (k - 1) % N]
    back = c_old[i, j, (k + 1) % N]

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
    

@cuda.reduce
def max_reduce(a, b):
    """
    Find the maximum of two values.
    """
    if a > b:
        return a
    else:
        return b


def diffusion_time_dependent_GPU(c_init, t_max, D=1.0, Ps=1.0, dt=0.001, dx=0.005, n_save_frames=100, spore_idx=(None, None, None), c_thresholds=None):
    """
    Compute the evolution of a square lattice of concentration scalars
    based on the time-dependent diffusion equation.
    inputs:
        c_init (numpy.ndarray) - the initial state of the lattice;
        t_max (int) - a maximum number of iterations;
        D (float) - the diffusion constant; defaults to 1;
        Ps (float) - the permeation constant through the spore barrier; defaults to 1;
        dt (float) - timestep; defaults to 0.001;
        dx (float) - spatial increment; defaults to 0.005;
        n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100;
        spore_idx (tuple) - the indices of the spore location; defaults to (None, None);
        c_thresholds (float) - threshold values for the concentration; defaults to None.
    outputs:
        u_evolotion (numpy.ndarray) - the states of the lattice at all moments in time.
    """

    assert c_init.ndim == 3, 'input array must be 3-dimensional'
    assert c_init.shape[0] == c_init.shape[1] == c_init.shape[2], 'lattice must have equal size along each dimension'

    # Determine number of lattice rows/columns
    N = c_init.shape[0]

    # Save update factor
    dtdx2 = dt / (dx ** 2)
    Db = Ps * dx

    if  D * dtdx2 > 0.5:
        print("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation.")

    if  Db * dtdx2 > 0.5:
        print("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation.")

    # Determine number of frames
    n_frames = int(np.floor(t_max / dt))
    print(f"Simulation running for {n_frames} steps on a lattice of size {np.array(c_init.shape) * dx} microns.")

    # Array for storing lattice states
    c_evolution = np.zeros((n_save_frames + 1, N, N, N))
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

    kernel_blocks, kernel_threads = invoke_smart_kernel(N)

    for t in range(n_frames):

        print(f"Frame {t} of {n_frames}", end="\r")

        # Save frame
        if t % save_interval == 0:
            c_evolution[save_ct] = c_A_gpu.copy_to_host()
            times[save_ct] = t * dt
            save_ct += 1
        
        update_GPU[kernel_blocks, kernel_threads](c_A_gpu, c_B_gpu, N, dtdx2, D, Db, spore_idx)

        # Synchronize the GPU to ensure the kernel has finished
        cuda.synchronize()
        
        c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu

        # Save time if threshold is reached
        if c_thresholds is not None:
            if thresh_ct < times_thresh.shape[0] and times_thresh[thresh_ct] == 0 and max_reduce(c_A_gpu.ravel()) < c_thresholds[thresh_ct]:
                times_thresh[thresh_ct] = t * dt
                thresh_ct += 1

    # Save final frame
    c_evolution[save_ct, :, :, :] = c_A_gpu.copy_to_host()
    times[save_ct] = t_max

    return c_evolution, times, times_thresh