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
#import peakutils
from scipy.optimize import curve_fit
import h5py
import matplotlib
import shutil
import imageio
import webbrowser
import scipy

# =============================================================================
# Stylizing the plot properties
# =============================================================================

def stylize(fig, x_label=None, y_label=None, title=None, xscale='linear',
            yscale='linear', xlim=None, ylim=None, grid=False,
            colorbar=False, legend=False, legend_loc=1,
            tight_layout=True, locs_x=None, ticks_x=None):
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
    if (locs_x is not None) and (ticks_x is not None):
        plt.xticks(locs_x, ticks_x)
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


# =============================================================================
# Coordinate reconstruction
# =============================================================================


def get_detector_mappings():
    # Declare corner offsets
    offset_1 = {'x': -0.907574, 'y': -3.162949,  'z': 5.384863}
    offset_2 = {'x': -1.246560, 'y': -3.161484,  'z': 5.317432}
    offset_3 = {'x': -1.579114, 'y': -3.164503,  'z': 5.227986}
    # Calculate angles
    corners = {'ESS_2': {1: [-1.579114, -3.164503, 5.227986],
                         2: [-1.252877, -3.162614, 5.314108]},
               'ESS_1': {3: [-1.246560, -3.161484, 5.317432],
                         4: [-0.916552, -3.160360, 5.384307]},
               'ILL':   {5: [-0.907574, -3.162949, 5.384863],
                         6: [-0.575025, -3.162578, 5.430037]}
               }
    ILL_C = corners['ILL']
    ESS_1_C = corners['ESS_1']
    ESS_2_C = corners['ESS_2']
    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2]) /
                        (ILL_C[6][0]-ILL_C[5][0])
                        )
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2]) /
                        (ESS_1_C[4][0]-ESS_1_C[3][0])
                        )
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2]) /
                        (ESS_2_C[2][0]-ESS_2_C[1][0])
                        )
    # Initiate detector mappings
    ILL_mapping = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    ESS_CLB_mapping = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    ESS_PA_mapping = create_ess_channel_to_coordinate_map(theta_3, offset_3)
    return [ILL_mapping, ESS_CLB_mapping, ESS_PA_mapping]


def create_ess_channel_to_coordinate_map(theta, offset):
    # Import voxel coordinates
    file_path = '../Tables/Coordinates_MG_SEQ_ESS.xlsx'
    matrix = pd.read_excel(file_path).values
    coordinates = matrix[1:801]
    # Initate empty matrix that will hold coordinate mapping
    ess_ch_to_coord = np.empty((3, 124, 80), dtype='object')
    coordinate = {'x': -1, 'y': -1, 'z': -1}
    axises = ['x', 'y', 'z']
    # Fill coordinate mapping
    for i, row in enumerate(coordinates):
        grid_ch = i // 20 + 80
        for j, col in enumerate(row):
            module = j // 12
            layer = (j // 3) % 4
            wire_ch = (19 - (i % 20)) + (layer * 20)
            coordinate_count = j % 3
            coordinate[axises[coordinate_count]] = col
            if coordinate_count == 2:
                # Assign coordinates
                x = coordinate['x']
                y = coordinate['y']
                z = coordinate['z']
                # Convert from [mm] to [m]
                x = x/1000
                y = y/1000
                z = z/1000
                # Shift to match internal and external coordinate system
                z, x, y = x, y, z
                # Apply rotation
                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                # Apply translation
                x += offset['x']
                y += offset['y']
                z += offset['z']
                # Insert value in mapping vector
                ess_ch_to_coord[module, grid_ch, wire_ch] = {'x': x,
                                                             'y': y,
                                                             'z': z}
                # Reset temporary coordinate
                coordinate = {'x': -1, 'y': -1, 'z': -1}
    return ess_ch_to_coord


def create_ill_channel_to_coordinate_map(theta, offset):
    # Spacing between voxels
    WireSpacing = 10      # [mm]
    LayerSpacing = 23.5   # [mm]
    GridSpacing = 23.5    # [mm]
    # Offset from corner
    x_offset = 46.514     # [mm]
    y_offset = 37.912     # [mm]
    z_offset = 37.95      # [mm]
    # Fill ill_ch_to_coord mapping
    ill_ch_to_coord = np.empty((3, 120, 80), dtype='object')
    for Bus in range(0, 3):
        for GridChannel in range(80, 120):
            for WireChannel in range(0, 80):
                    # Calculate coordinates
                    x = (WireChannel % 20)*WireSpacing + x_offset
                    y = ((WireChannel // 20)*LayerSpacing
                         + (Bus*4*LayerSpacing)
                         + y_offset
                         )
                    z = ((GridChannel-80)*GridSpacing) + z_offset
                    # Convert from [mm] to [m]
                    x = x/1000
                    y = y/1000
                    z = z/1000
                    # Shift to match internal and external coordinate system
                    z, x, y = x, y, z
                    # Apply rotation
                    x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                    # Apply translation
                    x += offset['x']
                    y += offset['y']
                    z += offset['z']
                    # Insert value in mapping
                    ill_ch_to_coord[Bus, GridChannel, WireChannel] = {'x': x,
                                                                      'y': y,
                                                                      'z': z}
    return ill_ch_to_coord


# =============================================================================
# Detector border lines
# =============================================================================


def initiate_detector_border_lines(detector_vec):
    # Initiate all pairs of corners were lines will g
    pairs_ESS = [[[80, 0], [80, 60]],
                 [[80, 0], [80, 19]],
                 [[80, 79], [80, 60]],
                 [[80, 79], [80, 19]],
                 [[119, 0], [119, 60]],
                 [[119, 0], [119, 19]],
                 [[119, 79], [119, 60]],
                 [[119, 79], [119, 19]],
                 [[80, 0], [119, 0]],
                 [[80, 19], [119, 19]],
                 [[80, 60], [119, 60]],
                 [[80, 79], [119, 79]]
                 ]
    pairs_ILL = [[[80, 0, 0], [80, 60, 2]],
                 [[80, 0, 0], [80, 19, 0]],
                 [[80, 79, 2], [80, 60, 2]],
                 [[80, 79, 2], [80, 19, 0]],
                 [[119, 0, 0], [119, 60, 2]],
                 [[119, 0, 0], [119, 19, 0]],
                 [[119, 79, 2], [119, 60, 2]],
                 [[119, 79, 2], [119, 19, 0]],
                 [[80, 0, 0], [119, 0, 0]],
                 [[80, 19, 0], [119, 19, 0]],
                 [[80, 60, 2], [119, 60, 2]],
                 [[80, 79, 2], [119, 79, 2]]
                 ]
    # For each of the pairs, create a plotly trace with the plot
    b_traces = []
    for bus in range(3, 9):
        detector = detector_vec[bus//3]
        for pair in pairs_ESS:
            x_vec = []
            y_vec = []
            z_vec = []
            for loc in pair:
                gCh = loc[0]
                wCh = loc[1]
                coord = detector[bus % 3, gCh, wCh]
                x_vec.append(coord['x'])
                y_vec.append(coord['y'])
                z_vec.append(coord['z'])
            b_trace = go.Scatter3d(x=z_vec,
                                   y=x_vec,
                                   z=y_vec,
                                   mode='lines',
                                   line=dict(color='rgba(0, 0, 0, 0.5)',
                                             width=5
                                             )
                                   )
            b_traces.append(b_trace)

    detector = detector_vec[0]
    for pair in pairs_ILL:
        x_vec = []
        y_vec = []
        z_vec = []
        for loc in pair:
            gCh = loc[0]
            wCh = loc[1]
            bus = loc[2]
            coord = detector[bus % 3, gCh, wCh]
            x_vec.append(coord['x'])
            y_vec.append(coord['y'])
            z_vec.append(coord['z'])
        b_trace = go.Scatter3d(x=z_vec,
                               y=x_vec,
                               z=y_vec,
                               mode='lines',
                               line=dict(color='rgba(0, 0, 0, 0.5)',
                                         width=5
                                         )
                               )
        b_traces.append(b_trace)

    return b_traces


# =============================================================================
# Coordinate rotation
# =============================================================================


def get_new_x(x, y, theta):
    return np.cos(np.arctan(y/x)+theta)*np.sqrt(x**2 + y**2)


def get_new_y(x, y, theta):
    return np.sin(np.arctan(y/x)+theta)*np.sqrt(x**2 + y**2)


def flip_bus(bus):
    flip_bus_dict = {0: 2, 1: 1, 2: 0}
    return flip_bus_dict[bus]


def flip_wire(wCh):
        if 0 <= wCh <= 19:
            wCh += 60
        elif 20 <= wCh <= 39:
            wCh += 20
        elif 40 <= wCh <= 59:
            wCh -= 20
        elif 60 <= wCh <= 79:
            wCh -= 60
        return wCh

# =============================================================================
# General helper functions
# =============================================================================

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def get_calibration(temp_calibration, E_i):
        calibrations = ['High_Resolution', 'High_Flux', 'RRM']
        choices = ['High Resolution (HR)', 'High Flux (HF)', 'Rate Repetition Mode (RRM)']
        calibration_dict = {}
        for choice, calibration in zip(choices, calibrations):
            calibration_dict.update({choice: calibration})
        if temp_calibration in choices:
            mode = calibration_dict[temp_calibration]
            return 'Van__3x3_' + mode + '_Calibration_' + str(E_i)
        else:
            return temp_calibration


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def Gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def Linear(x, k, m):
    return k*x+m


def calculate_peak_norm(bin_centers, hist, left_edge, right_edge,
                        background_level):
    area = sum(hist[left_edge:right_edge])
    bins_under_peak = abs(right_edge - 1 - left_edge)
    area_peak = area - background_level * bins_under_peak
    return area_peak


def get_background_level(dE_hist, bin_centers, left_back_idx, right_back_idx):
    background_level = (dE_hist[left_back_idx] + dE_hist[right_back_idx])/2
    return background_level


def Gaussian_fit(bin_centers, dE_hist, p0):
    number_sigmas_peak = 9
    number_sigmas_background = 9
    center_idx = len(bin_centers)//2
    zero_idx = find_nearest(bin_centers[center_idx-20:center_idx+20], 0)
    zero_idx += center_idx - 20
    fit_bins = np.arange(zero_idx-40, zero_idx+40+1, 1)
    Max = max(dE_hist[fit_bins])
    popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[fit_bins],
                                        dE_hist[fit_bins], p0=p0
                                        )
    sigma = abs(popt[2])
    x0 = popt[1]
    # Get peak limits
    left_idx = find_nearest(bin_centers,
                            x0 - number_sigmas_peak*sigma
                            )
    right_idx = find_nearest(bin_centers,
                             x0 + number_sigmas_peak*sigma
                             )
    # Get background limits and background level
    left_back_idx = find_nearest(bin_centers,
                                 x0 - number_sigmas_background*sigma
                                 )
    right_back_idx = find_nearest(bin_centers,
                                  x0 + number_sigmas_background*sigma
                                  )
    left_back_idx_2 = find_nearest(bin_centers,
                                   x0 + 13*sigma
                                   )
    right_back_idx_2 = find_nearest(bin_centers,
                                    x0 + 18*sigma
                                    )
    back_indices = np.arange(left_back_idx_2, right_back_idx_2+1, 1)
    background_level = sum(dE_hist[back_indices])/len(back_indices)
    #background_level = get_background_level(dE_hist, bin_centers,
    #                                        left_back_idx, right_back_idx
    #                                        )
    FWHM = 2*np.sqrt(2*np.log(2))*sigma
    area = calculate_peak_norm(bin_centers, dE_hist, left_idx, right_idx,
                               background_level)
    x_gaussian = bin_centers[fit_bins]
    y_gaussian = Gaussian(x_gaussian, popt[0], popt[1], popt[2])
    return (area, FWHM, y_gaussian, x_gaussian, Max, popt, left_idx, right_idx,
            background_level, back_indices)


def get_peak_area_and_FWHM(bin_centers, dE_hist, calibration, p0, measurement,
                           window):
    # Import width of peak
    sigmas_peak = get_sigmas_peak(calibration)
    condition_1 = calibration == 'Van__3x3_High_Resolution_Calibration_2.0'
    condition_2 = measurement == 'MG'
    condition_3 = (window.ILL.isChecked() and window.ESS_CLB.isChecked())
    if condition_1 and condition_2 and condition_3:
        sigmas_peak = 0.4
    # Fit data to gaussian
    area, FWHM, p0 = Gaussian_fit_energy_transfer(bin_centers, dE_hist, p0,
                                                  sigmas_peak, calibration,
                                                  measurement)
    return area, FWHM, p0


def calculate_peak_area(dE_hist, background, left_idx, right_idx):
    signal = np.array(dE_hist[left_idx:right_idx])
    background = np.array(background[left_idx:right_idx])
    return sum(signal - background)


def get_background(dE_hist, bin_centers, left_idx, right_idx):
    x_l = bin_centers[left_idx]
    y_l = dE_hist[left_idx]
    x_r = bin_centers[right_idx]
    y_r = dE_hist[right_idx]
    par_back = np.polyfit([x_l, x_r], [y_l, y_r], deg=1)
    f_back = np.poly1d(par_back)
    return f_back(bin_centers)


def Gaussian_fit_energy_transfer(bin_centers, dE_hist, p0, sigmas_peak,
                                 calibration, measurement):
    colors = {'MG': 'red', 'He3': 'blue'}
    labels = {'MG': 'Multi-Grid', 'He3': '$^3$He-tubes'}
    center_idx = len(bin_centers)//2
    zero_idx = find_nearest(bin_centers[center_idx-20:center_idx+20], 0)
    zero_idx += center_idx - 20
    fit_bins = np.arange(zero_idx-40, zero_idx+40+1, 1)
    Max = max(dE_hist[fit_bins])
    popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[fit_bins],
                                        dE_hist[fit_bins], p0=p0)
    sigma = abs(popt[2])
    x0 = popt[1]
    # Get peak limits
    left_idx = find_nearest(bin_centers,
                            x0 - sigmas_peak*sigma
                            )
    right_idx = find_nearest(bin_centers,
                             x0 + sigmas_peak*sigma
                             )
    # Get background level
    background = get_background(dE_hist, bin_centers, left_idx, right_idx)
    # Calculate area
    area = calculate_peak_area(dE_hist, background, left_idx, right_idx)
    # Calculate FWHM
    FWHM, fwhm_x, fwhm_y = get_FWHM(bin_centers, dE_hist, left_idx, right_idx)
    # Save information about peak in a separate folder, to cross-check
    x_gaussian = bin_centers[fit_bins]
    y_gaussian = Gaussian(x_gaussian, popt[0], popt[1], popt[2])
    # Save area calculation
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(bin_centers, dE_hist, '-', color=colors[measurement],
             label=labels[measurement], zorder=5)
    plt.plot(x_gaussian, y_gaussian, '-', color='purple', label='Gaussian_fit',
             zorder=5)
    plt.fill_between(bin_centers[left_idx:right_idx],
                     dE_hist[left_idx:right_idx],
                     background[left_idx:right_idx],
                     facecolor=colors[measurement],
                     label='Elastic peak area',
                     alpha=0.9, zorder=2)
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Normalized counts')
    plt.yscale('log')
    plt.ylim([0.000001, Max*1.2])
    plt.title('%s' % (labels[measurement]))
    plt.legend(loc=1)
    # Get path
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname,
                                 '../../Results/dE_sanity_check/%s/area'
                                 % measurement)
    fig.savefig(output_folder + calibration + '.pdf')
    plt.close()
    # Save FWHM calculation
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(bin_centers, dE_hist, '-', color=colors[measurement],
             label=labels[measurement], zorder=5)
    plt.plot(fwhm_x, fwhm_y, color='green', label='FWHM')
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Normalized counts')
    plt.yscale('log')
    plt.ylim([1, Max*1.2])
    plt.title('%s' % (labels[measurement]))
    plt.legend(loc=1)
    # Get path
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname,
                                 '../../Results/dE_sanity_check/%s/FWHM'
                                 % measurement)
    fig.savefig(output_folder + calibration + '.pdf')
    plt.close()
    return area, FWHM, popt


def get_sigmas_peak(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/' + 'MG_peak_area.xlsx')
    matrix = pd.read_excel(path).values
    sigmas_table = {}
    for row in matrix:
        sigmas_table.update({str(row[0]): row[1]})
    return sigmas_table[calibration]


def get_FWHM(bin_centers, dE_hist, left_edge, right_edge):
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    # Calculate background level
    x_l = bin_centers[left_edge]
    y_l = dE_hist[left_edge]
    x_r = bin_centers[right_edge]
    y_r = dE_hist[right_edge]
    par_back = np.polyfit([x_l, x_r], [y_l, y_r], deg=1)
    f_back = np.poly1d(par_back)
    xx_back = np.linspace(x_l, x_r, 100)
    yy_back = f_back(xx_back)
    # Calculate Half-Maximum
    maximum = max(dE_hist[left_edge:right_edge])
    peak = np.where(dE_hist[left_edge:right_edge] == maximum)
    peak = peak[len(peak)//2][len(peak)//2]

    M = dE_hist[left_edge:right_edge][peak]
    xM = bin_centers[left_edge:right_edge][peak]
    noise_level = yy_back[find_nearest(xx_back, xM)]
    HM = (M-noise_level)/2 + noise_level
    # Calculate FWHM
    left_idx = find_nearest(dE_hist[left_edge:left_edge+peak], HM)
    right_idx = find_nearest(dE_hist[left_edge+peak:right_edge], HM)
    # Find which two points to interpolate
    if dE_hist[left_edge+left_idx] > HM:
        sl = [-1, 0]
    else:
        sl = [0, 1]
    if dE_hist[left_edge+peak+right_idx] < HM:
        rl = [-1, 0]
    else:
        rl = [0, 1]
    left_x = [bin_centers[left_edge+left_idx+sl[0]],
              bin_centers[left_edge+left_idx+sl[1]]]
    left_y = [dE_hist[left_edge+left_idx+sl[0]],
              dE_hist[left_edge+left_idx+sl[1]]]
    right_x = [bin_centers[left_edge+peak+right_idx+rl[0]],
               bin_centers[left_edge+peak+right_idx+rl[1]]]
    right_y = [dE_hist[left_edge+peak+right_idx+rl[0]],
               dE_hist[left_edge+peak+right_idx+rl[1]]]
    # Interpolate
    par_left = np.polyfit(left_x, left_y, deg=1)
    f_left = np.poly1d(par_left)
    par_right = np.polyfit(right_x, right_y, deg=1)
    f_right = np.poly1d(par_right)
    xx_left = np.linspace(left_x[0], left_x[1], 100)
    xx_right = np.linspace(right_x[0], right_x[1], 100)
    yy_left = f_left(xx_left)
    yy_right = f_right(xx_right)
    left_idx = find_nearest(yy_left, HM)
    right_idx = find_nearest(yy_right, HM)

    L = xx_left[left_idx]
    R = xx_right[right_idx]
    FWHM = R - L

    return (FWHM, [xx_left[left_idx], xx_right[right_idx]],
            [yy_left[left_idx], yy_right[right_idx]])





# =============================================================================
# Import helper functions
# =============================================================================


def import_C4H2I2S_MG_files():
    MG_files = ["C4H2I2S_21meV.h5",
                "['mvmelst_1361.mvmelst', '...'].h5",
                "['mvmelst_1260.mvmelst', '...'].h5",
                "['mvmelst_1266.mvmelst', '...'].h5",
                "['mvmelst_1301.mvmelst', '...'].h5",
                "['mvmelst_1319.mvmelst', '...'].h5",
                "['mvmelst_1334.mvmelst', '...'].h5",
                "['mvmelst_1348.mvmelst', '...'].h5"
                ]
    return MG_files


def import_C4H2I2S_He3_files():
    He3_files = [['169112', '169113', '169114', '169115', '169116', '169117'],
                 ['169118', '169119', '169120', '169121', '169122', '169123'],
                 ['169124', '169125', '169126', '169127', '169128', '169129'],
                 ['169130', '169131', '169132', '169133', '169134', '169135'],
                 ['169136', '169137', '169138', '169139', '169140', '169141'],
                 ['169142', '169143', '169144', '169145', '169146', '169147'],
                 ['169148', '169149', '169150', '169151', '169152', '169153'],
                 ['169154', '169155', '169156', '169157', '169158', '169159']
                 ]
    return He3_files



def import_efficiency_correction():
    def angstrom_to_meV(a):
        return (9.045 ** 2) / (a ** 2)
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/pressure_wavelengthdat.txt')
    angstrom_vs_correction_table = np.loadtxt(path, delimiter=',')
    size = len(angstrom_vs_correction_table)
    angstrom_vs_correction = np.array([np.zeros(size), np.zeros(size),
                                       np.zeros(size)]) 
    for i, row in enumerate(angstrom_vs_correction_table):
        angstrom_vs_correction[0][i] = angstrom_to_meV(row[0])
        angstrom_vs_correction[1][i] = row[0]
        angstrom_vs_correction[2][i] = row[1]
    return angstrom_vs_correction


def import_efficiency_theoretical():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/65xyscan.txt')
    eff_a_meV_table = np.loadtxt(path, delimiter=',')
    print(eff_a_meV_table)
    size = len(eff_a_meV_table)
    meV_vs_eff = np.array([np.zeros(size), np.zeros(size)])
    for i, row in enumerate(eff_a_meV_table):
        meV_vs_eff[0][i] = row[0]
        meV_vs_eff[1][i] = row[1]
    return meV_vs_eff


def import_MG_coincident_events(input_path):
    return pd.read_hdf(input_path, 'coincident_events')


def import_MG_Ei(input_path):
    return pd.read_hdf(input_path, 'E_i')['E_i'].iloc[0]


def import_MG_calibration(input_path):
    return pd.read_hdf(input_path, 'calibration')['calibration'].iloc[0]


def import_MG_measurement_time(input_path):
    return pd.read_hdf(input_path,
                       'measurement_time')['measurement_time'].iloc[0]


def load_He3_h5(calibration):
    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../../Clusters/He3/')
    path = folder + calibration + '.h5'
    return pd.read_hdf(path, calibration)


def import_T0_table():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/' + 'T0_vs_Energy.xlsx')
    matrix = pd.read_excel(path).values
    t0_table = {}
    for row in matrix:
        t0_table.update({str(row[0]): row[1]})
    return t0_table


def get_t_off_He3(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/He3_offset.xlsx')
    matrix = pd.read_excel(path).values
    He3_offset_table = {}
    for row in matrix:
        He3_offset_table.update({row[0]: row[2]})
    offset = float(He3_offset_table[calibration])
    return offset


def get_t_off_table():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/' + 'time_offset.xlsx')
    matrix = pd.read_excel(path).values
    t_off_table = {}
    for row in matrix:
        t_off_table.update({row[0]: row[1]})
    return t_off_table


def get_t_off_MG(calibration):
    t_off_table = get_t_off_table()
    return t_off_table[calibration]


def get_He3_duration(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname,
                        '../../Tables/experiment_log.xlsx')
    matrix = pd.read_excel(path).values
    measurement_table = {}
    for row in matrix:
        measurement_table.update({row[1]: row[5]})
    He3_measurement_time = measurement_table[calibration]
    return He3_measurement_time


def find_He3_measurement_id(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../Tables/experiment_log.xlsx')
    matrix = pd.read_excel(path).values
    measurement_table = {}
    for row in matrix:
        measurement_table.update({row[1]: row[0]})
    return measurement_table[calibration]


def import_He3_coordinates():
    # Declare paths
    dirname = os.path.dirname(__file__)
    he_folder = os.path.join(dirname, '../../Tables/Helium3_coordinates/')
    az_path = he_folder + '145160_azimuthal.txt'
    dis_path = he_folder + '145160_distance.txt'
    pol_path = he_folder + '145160_polar.txt'
    # Import data
    az = np.loadtxt(az_path)
    dis = np.loadtxt(dis_path)
    pol = np.loadtxt(pol_path)
    # Convert to cartesian coordinates
    x = dis*np.sin(pol * np.pi/180)*np.cos(az * np.pi/180)
    y = dis*np.sin(az * np.pi/180)*np.sin(pol * np.pi/180)
    z = dis*np.cos(pol * np.pi/180)
    return x, y, z



def get_charge(calibration):
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, '../../Tables/Charge_normalisation.xlsx')
    matrix = pd.read_excel(path).values
    charge_norm_table = {}
    for i, row in enumerate(matrix):
        charge_norm_table.update({row[0]: [row[2], row[5], row[7], row[8]]})
    no_glitch_time = charge_norm_table[calibration][0]
    total_time = charge_norm_table[calibration][1]
    MG_charge = charge_norm_table[calibration][2]
    He3_charge = charge_norm_table[calibration][3]
    return (MG_charge * (no_glitch_time/total_time)), He3_charge


# =============================================================================
# Energy transfer helper functions
# =============================================================================

def get_T0(calibration, energy):
    T0_table = import_T0_table()
    return T0_table[calibration]


def get_dE_He3(E_i, ToF, d, T_0, t_off):
    # Declare parameters
    L_1 = 20.01                                # Target-to-sample distance
    m_n = 1.674927351e-27                      # Neutron mass [kg]
    meV_to_J = 1.60218e-19 * 0.001             # Convert meV to J
    J_to_meV = 6.24150913e18 * 1000            # Convert J to meV
    # Calculate dE
    E_i_J = E_i * meV_to_J                     # Convert E_i from meV to J
    v_i = np.sqrt((E_i_J*2)/m_n)               # Get velocity of E_i
    t_1 = (L_1 / v_i) + T_0 * 1e-6             # Use velocity to find t_1
    ToF_real = ToF * 1e-6 + (t_off * 1e-6)     # Time from source to detector
    t_f = ToF_real - t_1                       # Time from sample to detector
    E_J = (m_n/2) * ((d/t_f) ** 2)             # Energy E_f in Joule
    E_f = E_J * J_to_meV                       # Convert to meV
    return (E_i - E_f), t_f


def get_raw_He3_dE(calibration, E_i):
    # Import data
    df = load_He3_h5(calibration)
    offset = 39936 - 2 * 1024
    # Filter so that we do not include the events from the edges
    x_he, __, __ = import_He3_coordinates()
    # Declare surplus area because of deactivated tubes
    surplus1 = np.arange(0, 1024, 1)
    surplus2 = np.arange(37888//2, 39934//2, 1)
    surplus3 = np.arange(77824//2, 79870//2, 1)
    surplus4 = np.arange(82944//2, 83966//2, 1)
    surplus5 = np.arange(62080//2, 62462//2, 1)
    # Declare surplus area because of shielded end pixels
    surplus6 = np.arange(0, len(x_he), 64)
    surplus7 = np.arange(1, len(x_he), 64)
    surplus8 = np.arange(2, len(x_he), 64)
    surplus9 = np.arange(3, len(x_he), 64)
    surplus10 = np.arange(60, len(x_he), 64)
    surplus11 = np.arange(61, len(x_he), 64)
    surplus12 = np.arange(62, len(x_he), 64)
    surplus13 = np.arange(63, len(x_he), 64)
    # Declare surplus area from removed banks around beam dump
    surplus14 = np.arange(60416//2, 62078//2, 1)
    surplus15 = np.arange(62464//2, 67582//2, 1)
    # Concatenate all pixels
    surplus = np.concatenate((surplus1, surplus2, surplus3, surplus4,
                              surplus5, surplus6, surplus7, surplus8,
                              surplus9, surplus10, surplus11, surplus12,
                              surplus13, surplus14, surplus15), axis=None)
    # Filter data
    df = df[~((df.pixel-offset)//2).isin(surplus)]
    # Get dE
    T_0 = get_T0(calibration, E_i) * np.ones(len(df.ToF))
    t_off = get_t_off_He3(calibration) * np.ones(len(df.ToF))
    E_i = E_i * np.ones(len(df.ToF))
    dE, t_f = get_dE_He3(E_i, df.ToF, df.distance, T_0, t_off)
    df_temp_raw = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE_raw = df_temp_raw[df_temp_raw['t_f'] > 0].dE
    return dE_raw


def get_dE_MG(E_i, ToF, d, T_0, t_off, frame_shift):
    # Declare parameters
    L_1 = 20.01                                # Target-to-sample distance
    m_n = 1.674927351e-27                      # Neutron mass [kg]
    meV_to_J = 1.60218e-19 * 0.001             # Convert meV to J
    J_to_meV = 6.24150913e18 * 1000            # Convert J to meV
    # Calculate dE
    E_i_J = E_i * meV_to_J                     # Convert E_i from meV to J
    v_i = np.sqrt((E_i_J*2)/m_n)               # Get velocity of E_i
    t_1 = (L_1 / v_i) + T_0 * 1e-6             # Use velocity to find t_1
    ToF_real = ToF * 62.5e-9 + (t_off * 1e-6)  # Time from source to detector
    ToF_real += frame_shift                    # Apply frame-shift
    t_f = ToF_real - t_1                       # Time from sample to detector
    E_J = (m_n/2) * ((d/t_f) ** 2)             # Energy E_f in Joule
    E_f = E_J * J_to_meV                       # Convert to meV
    return (E_i - E_f), t_f


def get_MG_dE(df, calibration, E_i):
    t_off = np.ones(df.shape[0]) * get_t_off_MG(calibration)
    T_0 = np.ones(df.shape[0]) * get_T0(calibration, E_i)
    frame_shift = np.ones(df.shape[0]) * get_frame_shift(E_i)
    E_i = np.ones(df.shape[0]) * E_i
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE_MG(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    return dE


def get_frame_shift(E_i):
    frame_shifts = {2: 2 * (16666.66666e-6) - 0.0004475,
                    3: 2 * (16666.66666e-6) - 0.00800875,
                    4: 2 * (16666.66666e-6) - 0.0125178125,
                    5: 2 * (16666.66666e-6) - 0.015595,
                    6: (16666.66666e-6) - 0.001190399,
                    7: (16666.66666e-6) - 0.002965625,
                    8: (16666.66666e-6) - 0.0043893,
                    9: (16666.66666e-6) - 0.0055678125,
                    10: (16666.66666e-6) - 0.0065653125,
                    12: (16666.66666e-6) - 0.00817125,
                    14: (16666.66666e-6) - 0.00942,
                    15: (16666.66666e-6) - 0.009948437499999999,
                    16: (16666.66666e-6) - 0.01042562499,
                    18: (16666.66666e-6) - 0.011259375,
                    20: (16666.66666e-6) - 0.011965,
                    21: (16666.66666e-6) - 0.01227875,
                    25: (16666.66666e-6) - 0.013340625,
                    30: (16666.66666e-6) - 0.01435625,
                    32: (16666.66666e-6) - 0.014646875,
                    34: (16666.66666e-6) - 0.015009375,
                    35: (16666.66666e-6) - 0.01514625,
                    40: (16666.66666e-6) - 0.0157828125,
                    40.8: (16666.66666e-6) - 0.015878125,
                    48: (16666.66666e-6) - 0.0165909375
                    }
    if E_i in frame_shifts.keys():
        frame_shift = frame_shifts[E_i]
    else:
        frame_shift = 0
    return frame_shift


def import_HR_energies():
    HR_energies = np.array([2.0070418096,
                            3.0124122859,
                            4.018304398,
                            5.02576676447,
                            5.63307336334,
                            7.0406141592,
                            8.04786448037,
                            9.0427754509,
                            10.0507007198,
                            12.0647960483,
                            19.9019141333,
                            16.0973007945,
                            18.1003686861,
                            20.1184539648,
                            25.1618688243,
                            30.2076519655,
                            35.2388628217,
                            40.2872686153,
                            50.3603941793,
                            60.413447821,
                            70.4778157835,
                            80.4680371063,
                            90.4435536331,
                            100.413326074,
                            120.22744903,
                            139.795333256,
                            159.332776731,
                            178.971175232,
                            198.526931374,
                            222.999133573,
                            247.483042439,
                            271.986770107,
                            296.478093005
                            ])
    return HR_energies


def import_HF_energies():
    HF_energies = np.array([17.4528845174,
                            20.1216525977,
                            24.9948712594,
                            31.7092863506,
                            34.0101890432,
                            40.8410134518,
                            48.0774091652,
                            60.0900313977,
                            70.0602511267,
                            79.9920242035,
                            89.9438990322,
                            99.7962684685,
                            119.378824234,
                            138.763366168,
                            158.263398719,
                            177.537752942,
                            196.786207914,
                            221.079908375,
                            245.129939925,
                            269.278234529,
                            293.69020718,
                            341.776302631,
                            438.115942632,
                            485.795356795,
                            581.684376285,
                            677.286322624,
                            771.849709682,
                            866.511558326,
                            959.894393204,
                            1193.72178898,
                            1425.05415048,
                            1655.36691639,
                            1883.3912789,
                            2337.09815735,
                            2786.40707554,
                            3232.25185586])
    return HF_energies


def get_detector(window):
    ILL = window.ILL.isChecked()
    ESS_CLB = window.ESS_CLB.isChecked()
    ESS_PA = window.ESS_PA.isChecked()
    if ILL and not any([ESS_CLB, ESS_PA]):
        detector = ' - ILL'
    elif ESS_CLB and not any([ILL, ESS_PA]):
        detector = ' - ESS (Coated Long Blades)'
    elif ESS_PA and not any([ESS_CLB, ILL]):
        detector = ' - ESS (Pure Aluminum)'
    else:
        detector = ''
    return detector

                                
# =============================================================================
# Filters
# =============================================================================


def detector_filter(df, detector):
    if detector == 'ILL':
        df = df[(df.Bus <= 2) & (df.Bus >= 0)]
    elif detector == 'ESS_CLB':
        df = df[(df.Bus <= 5) & (df.Bus >= 3)]
    elif detector == 'ESS_PA':
        df = df[(df.Bus <= 8) & (df.Bus >= 6)]
    return df


def filter_ce_clusters(window, ce):
    # Declare parameters
    parameters = {'wM': [window.wM_min.value(),
                         window.wM_max.value(),
                         window.wM_filter.isChecked()],
                  'gM': [window.gM_min.value(),
                         window.gM_max.value(),
                         window.gM_filter.isChecked()],
                  'ceM': [window.ceM_min.value(),
                          window.ceM_max.value(),
                          window.ceM_filter.isChecked()],
                  'wADC': [window.wADC_min.value(),
                           window.wADC_max.value(),
                           window.wADC_filter.isChecked()],
                  'gADC': [window.gADC_min.value(),
                           window.gADC_max.value(),
                           window.gADC_filter.isChecked()],
                  'ToF': [window.ToF_min.value() / (62.5e-9 * 1e6),
                          window.ToF_max.value() / (62.5e-9 * 1e6),
                          window.ToF_filter.isChecked()],
                  'Time': [float(window.Time_min.text()),
                           float(window.Time_max.text()),
                           window.Time_filter.isChecked()],
                  'Bus': [window.module_min.value(),
                          window.module_max.value(),
                          window.module_filter.isChecked()],
                  'wire': [window.wire_min.value(),
                           window.wire_max.value(),
                           window.wire_filter.isChecked()],
                  'gCh': [window.grid_min.value() + 80 - 1,
                          window.grid_max.value() + 80 - 1,
                          window.grid_filter.isChecked()]

                  }
    # Only include the filters that we want to use
    ce_red = ce
    for parameter, (min_val, max_val, filter_on) in parameters.items():
        if filter_on:
            if parameter == 'wire':
                ce_red = ce_red[(((ce_red.wCh >= min_val - 1) &
                                  (ce_red.wCh <= max_val - 1)) |
                                 ((ce_red.wCh >= min_val + 20 - 1) &
                                  (ce_red.wCh <= max_val + 20 - 1)) |
                                 ((ce_red.wCh >= min_val + 40 - 1) &
                                  (ce_red.wCh <= max_val + 40 - 1)) |
                                 ((ce_red.wCh >= min_val + 60 - 1) &
                                  (ce_red.wCh <= max_val + 60 - 1)))
                                ]
            else:
                ce_red = ce_red[(ce_red[parameter] >= min_val) &
                                (ce_red[parameter] <= max_val)]


    # Filter on event values
    #ce_filtered = ce[(ce.ceM >= window.ceM_min.value()) &
    #                 (ce.ceM <= window.ceM_max.value()) &
    #                 (ce.wM >= window.wM_min.value()) &
    #                 (ce.wM <= window.wM_max.value()) &
    #                 (ce.gM >= window.gM_min.value()) &
    #                 (ce.gM <= window.gM_max.value()) &
    #                 (ce.wADC >= window.wADC_min.value()) &
    #                 (ce.wADC <= window.wADC_max.value()) &
    #                 (ce.gADC >= window.gADC_min.value()) &
    #                 (ce.gADC <= window.gADC_max.value()) &
    #                 (ce.ToF * 62.5e-9 * 1e6 >= ToF_min) &
    #                 (ce.ToF * 62.5e-9 * 1e6 <= ToF_max) &
    #                 (ce.Time >= timestamp_min) &
    #                 (ce.Time <= timestamp_max) &
    #                 (ce.Bus >= window.module_min.value()) &
    #                 (ce.Bus <= window.module_max.value()) &
    #                 (((ce.gCh >= window.grid_min.value() + 80 - 1)
    #                   ) &
    #                  ((ce.gCh <= window.grid_max.value() + 80 - 1)
    #                   )
    #                  ) &
    #                 (((ce.wCh >= window.wire_min.value() - 1) &
    #                   (ce.wCh <= window.wire_max.value() - 1)) |
    #                  ((ce.wCh >= window.wire_min.value() + 20 - 1) &
    #                   (ce.wCh <= window.wire_max.value() + 20 - 1)) |
    #                  ((ce.wCh >= window.wire_min.value() + 40 - 1) &
    #                   (ce.wCh <= window.wire_max.value() + 40 - 1)) |
    #                  ((ce.wCh >= window.wire_min.value() + 60 - 1) &
    #                   (ce.wCh <= window.wire_max.value() + 60 - 1))
    #                  )]
    # Filter away middle grids
    ce_red = ce_red[(ce_red.gCh <= window.lowerStartGrid.value() + 80 - 1) |
                    (ce_red.gCh >= window.upperStartGrid.value() + 80 - 1)]
    # Filter on detectors used in analysis
    ce_red = remove_modules(ce_red, window)
    # UNCOMMENT THIS IF YOU WOULD LIKE TO REMOVE THE SHADOWED AREA
    #gChs = [117, 118, 119]
    #wCh_mins = [16, 10, 6]
    #wCh_maxs = [20, 20, 20]
    #for gCh, wCh_min, wCh_max in zip(gChs, wCh_mins, wCh_maxs):
    #    ce_filtered = ce_filtered[~((((ce_filtered.wCh >= wCh_min - 1) &
    #                                (ce_filtered.wCh <= wCh_max - 1)) |
    #                               ((ce_filtered.wCh >= wCh_min + 20 - 1) &
    #                                (ce_filtered.wCh <= wCh_max + 20 - 1)) |
    #                               ((ce_filtered.wCh >= wCh_min + 40 - 1) &
    #                                (ce_filtered.wCh <= wCh_max + 40 - 1)) |
    #                               ((ce_filtered.wCh >= wCh_min + 60 - 1) &
    #                                (ce_filtered.wCh <= wCh_max + 60 - 1))
    #                                 ) &
    #                              (ce_filtered.gCh == gCh + 80 - 1)
    #                              )
    #                              ]
    return ce_red


def get_modules_to_include(window):
    detectors = ['ILL', 'ESS_CLB', 'ESS_PA']
    detectors = {'ILL': {'isChecked': window.ILL.isChecked(),
                         'modules': [0, 1, 2]},
                 'ESS_CLB': {'isChecked': window.ESS_CLB.isChecked(),
                             'modules': [3, 4, 5]},
                 'ESS_PA': {'isChecked': window.ESS_PA.isChecked(),
                            'modules': [6, 7, 8]}
                 }
    modules_to_include = []
    for detector in detectors.keys():
        if detectors[detector]['isChecked'] is True:
            modules_temp = detectors[detector]['modules']
            modules_to_include.extend(modules_temp)
    return modules_to_include


def remove_modules(df, window):
    modules = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    modules_to_include = get_modules_to_include(window)
    for module in modules:
        if module not in modules_to_include:
            df = df[df.Bus != module]
    return df

# =============================================================================
# Area and solid angle calculations
# =============================================================================

def get_multi_grid_area_and_solid_angle(window, calibration, Ei):
    #def get_solid_angle(l, b, d):
    #    return 4*np.arcsin((l*b)/(np.sqrt((l**2 + 4*d**2)*(b**2 + 4*d**2))))

    # Declare parameters (voxel area is in m^2)
    MG_area = 0
    MG_projected_area = 0
    MG_solid_angle = 0
    voxel_area = 0.0005434375
    l = 0.0225
    b = 0.0225
    detectors = get_detector_mappings()
    # Get amount of surface to include
    modules_to_include = get_modules_to_include(window)
    module_to_exclude = get_module_to_exclude(calibration, Ei)
    condition_1 = (module_to_exclude is not None)
    condition_2 = (module_to_exclude in modules_to_include)
    if condition_1 and condition_2:
        idx = modules_to_include.index(module_to_exclude)
        modules_to_include.pop(idx)
    grids = get_grids(window)
    # Iterate through surface and calculate area and solid angle
    for module in modules_to_include:
        detector = detectors[module//3]
        wires = get_wires(module)
        print(wires)
        print(grids)
        print('Number wires: %d' % len(wires))
        print('Number grids: %d' % len(grids))
        counts = 0
        for grid in grids:
            for wire in wires:
                # Extract coordinates
                vox_coordinate = detector[module % 3, grid, wire]
                x_vox = vox_coordinate['x']
                y_vox = vox_coordinate['y']
                z_vox = vox_coordinate['z']
                x_vox, y_vox, z_vox = z_vox, x_vox, y_vox
                # Do calculations
                theta = np.arctan(abs(z_vox/x_vox))
                d = np.sqrt(x_vox ** 2 + y_vox ** 2 + z_vox ** 2)
                projected_area = voxel_area * np.cos(theta)
                MG_area += voxel_area
                MG_projected_area += projected_area
                #l_proj = l * np.cos(theta)
                MG_solid_angle += (projected_area / (d ** 2))  # get_solid_angle(l_proj, b, d)
                counts += 1
        print('Loops: %d' % counts)
    print('MG area')
    print(MG_area)
    return MG_area, MG_solid_angle


def get_wires(module):
    if module == 2:
        wires = [0, 20, 40]
    elif module == 3:
        wires = [0, 20, 60]
    elif module == 8:
        wires = [0, 40, 60]
    else:
        wires = [0, 20, 40, 60]
    return wires


def get_grids(window):
    # Get parameters
    start = window.grid_min.value() + 80 - 1
    stop = window.grid_max.value() + 80 - 1
    intersect_start = window.lowerStartGrid.value() + 80 - 1
    intersect_stop = window.upperStartGrid.value() + 80 - 1
    # Declare grids
    grids_temp = np.arange(start, stop+1, 1)
    if intersect_start < intersect_stop:
        intersection = np.arange(intersect_start+1, intersect_stop, 1)
    else:
        intersection = []
    grids = [grid for grid in grids_temp if grid not in intersection]
    return grids


def get_module_to_exclude(calibration, Ei):
    isHighFlux = (calibration[0:30] == 'Van__3x3_High_Flux_Calibration')
    isHighResolution = not isHighFlux
    if isHighFlux and Ei < 450:
        module_to_exclude = 4
    elif isHighResolution and Ei > 50:
        module_to_exclude = 4
    else:
        module_to_exclude = None
    return module_to_exclude


def get_He3_tubes_area_and_solid_angle():
    def get_He3_area(i):
        # Declare pixels sizes
        lower_small = [32256, 33279]
        upper_small = [31232, 32255]
        normal_pixel_size = 0.0254 * 0.01875
        lower_small_pixel_size = 0.0254 * 0.00521
        upper_small_pixel_size = 0.0254 * 0.0040482
        area = normal_pixel_size
        if lower_small[0] <= i <= lower_small[1]:
            area = lower_small_pixel_size
        elif upper_small[0] <= i <= upper_small[1]:
            area = upper_small_pixel_size
        return area
    # Import data
    x_he, y_he, z_he = import_He3_coordinates()
    # Declare surplus area because of deactivated tubes
    surplus1 = np.arange(0, 1024, 1)
    surplus2 = np.arange(37888//2, 39934//2, 1)
    surplus3 = np.arange(77824//2, 79870//2, 1)
    surplus4 = np.arange(82944//2, 83966//2, 1)
    surplus5 = np.arange(62080//2, 62462//2, 1)
    # Declare surplus area because of shielded end pixels
    surplus6 = np.arange(0, len(x_he), 64)
    surplus7 = np.arange(1, len(x_he), 64)
    surplus8 = np.arange(2, len(x_he), 64)
    surplus9 = np.arange(3, len(x_he), 64)
    surplus10 = np.arange(60, len(x_he), 64)
    surplus11 = np.arange(61, len(x_he), 64)
    surplus12 = np.arange(62, len(x_he), 64)
    surplus13 = np.arange(63, len(x_he), 64)
    # Declare surplus area from removed banks around beam dump
    surplus14 = np.arange(60416//2, 62078//2, 1)
    surplus15 = np.arange(62464//2, 67582//2, 1)
    # Concatenate all pixels
    surplus = np.concatenate((surplus1, surplus2, surplus3, surplus4,
                              surplus5, surplus6, surplus7, surplus8, surplus9,
                              surplus10, surplus11, surplus12, surplus13,
                              surplus14, surplus15), axis=None)
    # Declare MG-sized fragmet
    #MG_sized_fragment = np.arange(32768//2, 36862//2, 1)
    #end_pixels = np.concatenate((surplus6, surplus7, surplus8, surplus9,
    #                             #surplus12, surplus13,
    #                             #surplus14, surplus15,
    #                             #surplus16, surplus17,
    #                            ), axis=None)
    # Declare parameters
    He3_area = 0
    He3_projected_area = 0
    He3_solid_angle = 0
    x_he, y_he, z_he = z_he, x_he, y_he
    # Loop over all pixels
    for pixel_id, (x, y, z) in enumerate(zip(x_he, y_he, z_he)):
        theta = np.arctan(abs(z/x))
        phi = np.arctan(abs(y/x))
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        area = get_He3_area(pixel_id)
        projected_area = area * np.cos(theta)
        if pixel_id in surplus:
            pass
        else:
            He3_area += area
            He3_projected_area += projected_area
            He3_solid_angle += (projected_area / (d ** 2))
    print('He3 area')
    print(He3_area)
    print('He3 solid')
    print(He3_solid_angle)
    return He3_area, He3_solid_angle


# =============================================================================
# Export data to text
# =============================================================================


def export_ToF_histograms_to_text(calibration, MG_bin_centers, He3_bin_centers,
                                  MG_hist_normalized, He3_hist, MG_back_hist):
    dir_name = os.path.dirname(__file__)
    MG_path = os.path.join(dir_name,
                           '../../Results/Histograms/MG/ToF/MG_%s_meV.txt'
                           % calibration
                           )
    He3_path = os.path.join(dir_name,
                            '../../Results/Histograms/He3/ToF/He3_%s_meV.txt'
                            % calibration
                            )
    MG_dict = {'ToF [um]': MG_bin_centers,
               'Signal [Normalized Counts]': MG_hist_normalized,
               'Background estimation [Normalized counts]': MG_back_hist
               }
    He3_dict = {'ToF [um]': He3_bin_centers,
                'Signal [Normalized Counts]': He3_hist
                }
    MG_df = pd.DataFrame(MG_dict)
    He3_df = pd.DataFrame(He3_dict)
    MG_df.to_csv(MG_path, index=None, sep=' ', mode='w', encoding='ascii')
    He3_df.to_csv(He3_path, index=None, sep=' ', mode='w', encoding='ascii')


def export_dE_histograms_to_text(calibration, bin_centers, MG_dE_hist,
                                 He3_dE_hist):
    dir_name = os.path.dirname(__file__)
    MG_path = os.path.join(dir_name,
                           '../../Results/Histograms/MG/dE/MG_%s_meV.txt'
                           % calibration)
    He3_path = os.path.join(dir_name,
                            '../../Results/Histograms/He3/dE/He3_%s_meV.txt'
                            % calibration)
    MG_dict = {'dE [meV]': bin_centers,
               'Normalized Counts': MG_dE_hist,
               }
    He3_dict = {'dE [meV]': bin_centers,
                'Normalized Counts': He3_dE_hist
                }
    MG_df = pd.DataFrame(MG_dict)
    He3_df = pd.DataFrame(He3_dict)
    MG_df.to_csv(MG_path, index=None, sep=' ', mode='w', encoding='ascii')
    He3_df.to_csv(He3_path, index=None, sep=' ', mode='w', encoding='ascii')


# =============================================================================
# Get information for each energy
# =============================================================================


def get_ToF_intervals():
    intervals_He3 = [[45000, 48000],
                     [38000, 41000],
                     [21000, 23000],
                     [30000, 32000],
                     [27000, 31000],
                     [25000, 30000],
                     [13000, 16000],
                     [23000, 27000],
                     [10e3, 15e3],
                     [11e3, 13e3],
                     [8e3, 12e3],
                     [8e3, 11e3],
                     [6e3, 11e3],
                     [6e3, 10e3],
                     [5e3, 8e3],
                     [6e3, 8e3],
                     [5e3, 6e3],
                     [5e3, 7e3],
                     [4e3, 7e3],
                     [13e3, 16e3],
                     [12e3, 16e3],
                     [11e3, 16e3],
                     [11e3, 16e3],
                     [10e3, 16e3],
                     [10e3, 16e3],
                     [10e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 16e3],
                     [8e3, 12e3],  # High Flux measurements from here
                     [8e3, 10e3],
                     [8e3, 9e3],
                     [7e3, 8e3],
                     [6e3, 7e3],
                     [5e3, 7e3],
                     [6e3, 7e3],
                     [15e3, 16e3],
                     [15e3, 16e3],
                     [13e3, 16e3],
                     [12e3, 16e3],
                     [12e3, 16e3],
                     [12e3, 16e3],
                     [10e3, 16e3],
                     [11e3, 12e3],
                     [10e3, 12e3],
                     [10e3, 12e3],
                     [9e3, 13e3],
                     [8e3, 13e3],
                     [8e3, 12e3],
                     [8e3, 11e3],
                     [8e3, 10e3],
                     [8e3, 9e3],
                     [6e3, 10e3],
                     [6e3, 8e3],
                     [6e3, 8e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3],
                     [13e3, 16e3]
                     ]
    intervals_MG = intervals_He3
    intervals_MG_back = [[0, 12500]] * len(intervals_He3)
    all_intervals = np.concatenate((intervals_MG,
                                    intervals_MG_back,
                                    intervals_He3), axis=1)
    return all_intervals


def get_all_calibrations():
    HR_calibrations = ["Van__3x3_High_Resolution_Calibration_2.0",
                       "Van__3x3_High_Resolution_Calibration_3.0",
                       "Van__3x3_High_Resolution_Calibration_4.0",
                       "Van__3x3_High_Resolution_Calibration_5.0",
                       "Van__3x3_High_Resolution_Calibration_6.0",
                       "Van__3x3_High_Resolution_Calibration_7.0",
                       "Van__3x3_High_Resolution_Calibration_8.0",
                       "Van__3x3_High_Resolution_Calibration_9.0",
                       "Van__3x3_High_Resolution_Calibration_10.0",
                       "Van__3x3_High_Resolution_Calibration_12.0",
                       "Van__3x3_High_Resolution_Calibration_14.0",
                       "Van__3x3_High_Resolution_Calibration_16.0",
                       "Van__3x3_High_Resolution_Calibration_18.0",
                       "Van__3x3_High_Resolution_Calibration_20.0",
                       "Van__3x3_High_Resolution_Calibration_25.0",
                       "Van__3x3_High_Resolution_Calibration_30.0",
                       "Van__3x3_High_Resolution_Calibration_35.0",
                       "Van__3x3_High_Resolution_Calibration_40.0",
                       "Van__3x3_High_Resolution_Calibration_50.0",
                       "Van__3x3_High_Resolution_Calibration_60.0",
                       "Van__3x3_High_Resolution_Calibration_70.0",
                       "Van__3x3_High_Resolution_Calibration_80.0",
                       "Van__3x3_High_Resolution_Calibration_90.0",
                       "Van__3x3_High_Resolution_Calibration_100.0",
                       "Van__3x3_High_Resolution_Calibration_120.0",
                       "Van__3x3_High_Resolution_Calibration_140.0",
                       "Van__3x3_High_Resolution_Calibration_160.0",
                       "Van__3x3_High_Resolution_Calibration_180.0",
                       "Van__3x3_High_Resolution_Calibration_200.0",
                       "Van__3x3_High_Resolution_Calibration_225.0",
                       "Van__3x3_High_Resolution_Calibration_250.0",
                       "Van__3x3_High_Resolution_Calibration_275.0",
                       "Van__3x3_High_Resolution_Calibration_300.0"
                       ]
    HF_calibrations = ["Van__3x3_High_Flux_Calibration_15.0",
                       "Van__3x3_High_Flux_Calibration_20.0",
                       "Van__3x3_High_Flux_Calibration_25.0",
                       "Van__3x3_High_Flux_Calibration_32.0",
                       "Van__3x3_High_Flux_Calibration_34.0",
                       "Van__3x3_High_Flux_Calibration_40.8",
                       "Van__3x3_High_Flux_Calibration_48.0",
                       "Van__3x3_High_Flux_Calibration_60.0",
                       "Van__3x3_High_Flux_Calibration_70.0",
                       "Van__3x3_High_Flux_Calibration_80.0",
                       "Van__3x3_High_Flux_Calibration_90.0",
                       "Van__3x3_High_Flux_Calibration_100.0",
                       "Van__3x3_High_Flux_Calibration_120.0",
                       "Van__3x3_High_Flux_Calibration_140.0",
                       "Van__3x3_High_Flux_Calibration_160.0",
                       "Van__3x3_High_Flux_Calibration_180.0",
                       "Van__3x3_High_Flux_Calibration_200.0",
                       "Van__3x3_High_Flux_Calibration_225.0",
                       "Van__3x3_High_Flux_Calibration_250.0",
                       "Van__3x3_High_Flux_Calibration_275.0",
                       "Van__3x3_High_Flux_Calibration_300.0",
                       "Van__3x3_High_Flux_Calibration_350.0",
                       # "Van__3x3_High_Flux_Calibration_400.0",
                       "Van__3x3_High_Flux_Calibration_450.0",
                       "Van__3x3_High_Flux_Calibration_500.0",
                       "Van__3x3_High_Flux_Calibration_600.0",
                       "Van__3x3_High_Flux_Calibration_700.0",
                       "Van__3x3_High_Flux_Calibration_800.0",
                       "Van__3x3_High_Flux_Calibration_900.0",
                       "Van__3x3_High_Flux_Calibration_1000.0",
                       "Van__3x3_High_Flux_Calibration_1250.0",
                       "Van__3x3_High_Flux_Calibration_1500.0",
                       "Van__3x3_High_Flux_Calibration_1750.0",
                       "Van__3x3_High_Flux_Calibration_2000.0",
                       "Van__3x3_High_Flux_Calibration_2500.0",
                       "Van__3x3_High_Flux_Calibration_3000.0",
                       "Van__3x3_High_Flux_Calibration_3500.0"
                       ]
    return HR_calibrations, HF_calibrations


def get_all_energies():
    HR_energies = np.array([2.0070418096,
                            3.0124122859,
                            4.018304398,
                            5.02576676447,
                            5.63307336334,
                            7.0406141592,
                            8.04786448037,
                            9.0427754509,
                            10.0507007198,
                            12.0647960483,
                            19.9019141333,
                            16.0973007945,
                            18.1003686861,
                            20.1184539648,
                            25.1618688243,
                            30.2076519655,
                            35.2388628217,
                            40.2872686153,
                            50.3603941793,
                            60.413447821,
                            70.4778157835,
                            80.4680371063,
                            90.4435536331,
                            100.413326074,
                            120.22744903,
                            139.795333256,
                            159.332776731,
                            178.971175232,
                            198.526931374,
                            222.999133573,
                            247.483042439,
                            271.986770107,
                            296.478093005
                            ])
    HF_energies = np.array([17.4528845174,
                            20.1216525977,
                            24.9948712594,
                            31.7092863506,
                            34.0101890432,
                            40.8410134518,
                            48.0774091652,
                            60.0900313977,
                            70.0602511267,
                            79.9920242035,
                            89.9438990322,
                            99.7962684685,
                            119.378824234,
                            138.763366168,
                            158.263398719,
                            177.537752942,
                            196.786207914,
                            221.079908375,
                            245.129939925,
                            269.278234529,
                            293.69020718,
                            341.776302631,
                            438.115942632,
                            485.795356795,
                            581.684376285,
                            677.286322624,
                            771.849709682,
                            866.511558326,
                            959.894393204,
                            1193.72178898,
                            1425.05415048,
                            1655.36691639,
                            1883.3912789,
                            2337.09815735,
                            2786.40707554,
                            3232.25185586])
    return HR_energies, HF_energies


def get_energy(element):
    start = element.find('Calibration_')+len('Calibration_')
    stop = element.find('_meV')
    return float(element[start:stop])


def get_chopper_setting(calibration):
    is_high_flux = (calibration[0:30] == 'Van__3x3_High_Flux_Calibration')
    if is_high_flux:
        chopper_setting = 'HF'
    else:
        chopper_setting = 'HR'
    return chopper_setting


def append_folder_and_files(folder, files):
    folder_vec = np.array(len(files)*[folder])
    return np.core.defchararray.add(folder_vec, files)

















