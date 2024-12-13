import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_spore_positions(N, spores_x, spores_y, spores_z, dx, title=None):
    """
    Plot the spore positions in 3D.
    inputs:
        N (int): the size of the lattice
        spores_x (numpy array): the x coordinates of the spores
        spores_y (numpy array): the y coordinates of the spores
        spores_z (numpy array): the z coordinates of the spores
        dx (float): the lattice spacing in micrometers
        title (str): the title of the plot
    """
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(spores_x * dx, spores_y * dx, spores_z * dx, marker='.')
    ax.set_xlim(0, N * dx)
    ax.set_ylim(0, N * dx)
    ax.set_zlim(0, N * dx)
    ax.set_xlabel('$x$ $[\mu m]$')
    ax.set_ylabel('$y$ $[\mu m]$')
    ax.set_zlabel('$z$ $[\mu m]$')
    if title:
        ax.set_title(title)
    plt.show()
