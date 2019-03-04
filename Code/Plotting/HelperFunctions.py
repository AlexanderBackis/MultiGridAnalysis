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
    WireSpacing = 10     # [mm]
    LayerSpacing = 23.5   # [mm]
    GridSpacing = 23.5   # [mm]
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
            return calibration


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
    background_level = get_background_level(dE_hist, bin_centers,
                                            left_back_idx, right_back_idx
                                            )
    FWHM = 2*np.sqrt(2*np.log(2))*sigma
    area = calculate_peak_norm(bin_centers, dE_hist, left_idx, right_idx,
                               background_level)
    x_gaussian = bin_centers[fit_bins]
    y_gaussian = Gaussian(x_gaussian, popt[0], popt[1], popt[2])
    return area, FWHM, y_gaussian, x_gaussian, Max, popt, left_idx, right_idx, background_level


def detector_filter(df, detector):
    if detector == 'ILL':
        df = df[(df.Bus <= 2) & (df.Bus >= 0)]
    elif detector == 'ESS_CLB':
        df = df[(df.Bus <= 5) & (df.Bus >= 3)]
    elif detector == 'ESS_PA':
        df = df[(df.Bus <= 8) & (df.Bus >= 6)]
    return df


# =============================================================================
# Import helper functions
# =============================================================================


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
    df = load_He3_h5(calibration)
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

# =============================================================================
# Filters
# =============================================================================


def filter_ce_clusters(window, ce):
    ce_filtered = ce[(ce.wM >= window.wM_min.value()) &
                     (ce.wM <= window.wM_max.value()) &
                     (ce.gM >= window.gM_min.value()) &
                     (ce.gM <= window.gM_max.value()) &  
                     (ce.wADC >= window.wADC_min.value()) &
                     (ce.wADC <= window.wADC_max.value()) &
                     (ce.gADC >= window.gADC_min.value()) &
                     (ce.gADC <= window.gADC_max.value()) &
                     (ce.ToF * 62.5e-9 * 1e6 >= window.ToF_min.value()) &
                     (ce.ToF * 62.5e-9 * 1e6 <= window.ToF_max.value()) &
                     (ce.Bus >= window.module_min.value()) &
                     (ce.Bus <= window.module_max.value()) &
                     (((ce.gCh >= window.grid_min.value() + 80 - 1) &
                       (ce.gCh <= window.lowerStartGrid.value() + 80 - 1)
                       ) |
                      ((ce.gCh <= window.grid_max.value() + 80 - 1) &
                       (ce.gCh >= window.upperStartGrid.value() + 80 - 1)
                       )
                      ) &
                     (((ce.wCh >= window.wire_min.value() - 1) &
                       (ce.wCh <= window.wire_max.value() - 1)) |
                      ((ce.wCh >= window.wire_min.value() + 20 - 1) &
                       (ce.wCh <= window.wire_max.value() + 20 - 1)) |
                      ((ce.wCh >= window.wire_min.value() + 40 - 1) &
                       (ce.wCh <= window.wire_max.value() + 40 - 1)) |
                      ((ce.wCh >= window.wire_min.value() + 60 - 1) &
                       (ce.wCh <= window.wire_max.value() + 60 - 1))
                      )
                     ]
    return ce_filtered


