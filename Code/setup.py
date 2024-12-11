import numpy as np
import conversions as conv

def setup_lattice(N, spores_x, spores_y, spores_z, c_spore_init=1):
    """
    Set up the lattice with spores at the given locations.
    inputs:
        N (int): the size of the lattice
        spores_x (numpy array): the x coordinates of the spores
        spores_y (numpy array): the y coordinates of the spores
        spores_z (numpy array): the z coordinates of the spores
        c_spore_init (float): the initial concentration at the spores
    """

    assert spores_x.size == spores_y.size == spores_z.size, "spores_x, spores_y, and spores_z must have the same size"

    c_lattice = np.zeros((N+1, N+1, N+1), dtype=np.float64)
    c_lattice[spores_x, spores_y, spores_z] = c_spore_init

    return c_lattice

def populate_spore_grid_coords(N, dx, spore_density, bottom_only=False):
    """
    Generate spore grid coordinates for a given spore density.
    inputs:
        N (int): the size of the lattice
        dx (float): the lattice spacing in micrometers
        spore_density (float): the density of spores in micrometers^-3
        bottom_only (bool): whether to only populate the bottom layer of the lattice
    """

    # Calculate the number of spores to place
    V_grid = N**3 * dx**3
    n_spores = spore_density * V_grid
    if bottom_only:
        V_occupied = N**2 * dx**3
        assert np.sqrt(n_spores) % 1 == 0, "number of spores must be a perfect square if bottom_only is True"
        n_spores_1D = np.sqrt(n_spores)
    else:
        V_occupied = V_grid
        assert np.cbrt(n_spores) % 1 == 0, "number of spores must be a perfect cube if bottom_only is False"
        n_spores_1D = np.cbrt(n_spores)

    L = N * dx
    print(L)
    spore_spacing = L / n_spores_1D

    print(f"Populating volume of {V_occupied} micrometers^3 with {n_spores} spores, {n_spores_1D} spores per dimension")
    print(f"Spore spacing: {spore_spacing} micrometers")

    # Generate the spore grid coordinates
    spores_x = np.arange(0, N+1, spore_spacing / dx)
    spores_y = np.arange(0, N+1, spore_spacing / dx)
    spores_z = np.arange(0, N+1, spore_spacing / dx)

    return spores_x, spores_y, spores_z
