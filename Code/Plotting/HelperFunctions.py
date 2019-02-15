import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.patheffects as path_effects
import plotly.io as pio
import plotly as py
import plotly.graph_objs as go
import scipy
import peakutils
from scipy.optimize import curve_fit
import h5py
import matplotlib
import shutil
import imageio
import webbrowser
import scipy


def stylize(fig, x_label=None, y_label=None, title=None, xscale='linear',
            yscale='linear', xlim=None, ylim=None, grid=False,
            colorbar=False, legend=False, legend_loc=1,
            tight_layout=True):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    return fig


def set_figure_properties(fig, suptitle='', height=None, width=None):
    fig.suptitle(suptitle, x=0.5, y=1.08)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    plt.tight_layout()
    return fig


def get_duration(ce):
    start_time = ce.head(1)['Time'].values[0]
    end_time = ce.tail(1)['Time'].values[0]
    return (end_time - start_time) * 62.5e-9
