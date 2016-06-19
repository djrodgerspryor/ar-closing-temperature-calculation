import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from itertools import product, repeat, chain
import numpy as np
import scipy
from math import ceil, sqrt

def plot_1d(
        x,
        y,
        x_label=None,
        y_label=None,
        title=None,
        ax=None,
        block=False):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ax.plot(x, y)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    plt.show(block=block)


def plot_2d(
        data,
        x_label=None,
        y_label=None,
        z_label=None,
        title=None,
        extent=None,
        ax=None,
        block=False,
        central_colour_value=None,
        colour_range_offset=None,
        cmap='seismic'):
    """
        Ensures that colour range is symmetric around the given central_colour_value.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ### Process args ###
    if central_colour_value is None:
        central_colour_value = np.mean(data)

    if colour_range_offset is None:
        colour_range_offset = np.max(
            np.abs(
                data - central_colour_value
            )
        )

    colour_range = (
        central_colour_value - colour_range_offset,
        central_colour_value + colour_range_offset
    )


    ### Plot ###
    img = ax.imshow(data, extent=extent)

    ### Format plot ###
    img.set_cmap(cmap)

    img.set_clim(*colour_range)

    # Colourbar
    cbar = fig.colorbar(img)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if z_label is not None:
        cbar.set_label(z_label)

    if title is not None:
        ax.set_title(title)

    plt.show(block=block)

    return img