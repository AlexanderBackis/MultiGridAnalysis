# =======  LIBRARIES  ======= #
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly as py
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio
import os

from Plotting.HelperFunctions import get_duration, set_figure_properties
from Plotting.HelperFunctions import stylize, filter_ce_clusters
from Plotting.HelperFunctions import get_detector_mappings, flip_bus, flip_wire
from Plotting.HelperFunctions import (initiate_detector_border_lines,
                                      get_multi_grid_area_and_solid_angle,
                                      import_MG_calibration, mkdir_p,
                                      import_MG_coincident_events)

# =============================================================================
# Coincidence Histogram (2D)
# =============================================================================


def Coincidences_2D_plot(ce, data_sets, module_order, window):
    def plot_2D_bus(fig, sub_title, ce, vmin, vmax, duration):
        h, *_ = plt.hist2d(ce.wCh, ce.gCh, bins=[80, 40],
                           range=[[-0.5, 79.5], [79.5, 119.5]],
                           vmin=vmin, vmax=vmax, norm=LogNorm(), cmap='jet')
        xlabel = 'Wire [Channel number]'
        ylabel = 'Grid [Channel number]'
        fig = stylize(fig, xlabel, ylabel, title=sub_title, colorbar=True)
        plt.colorbar()
        return fig, h

    # Filter clusters
    ce = filter_ce_clusters(window, ce)
    # Declare parameters (added with condition if empty array)
    if ce.shape[0] != 0:
        duration = window.measurement_time
        vmin = 1
        vmax = ce.shape[0] // 4500 + 5
    else:
        duration = 1
        vmin = 1
        vmax = 1
    title = 'Coincident events (2D)\nData set(s): %s' % data_sets
    height = 12
    width = 14
    # Ensure only coincident events are plotted
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    # Retrieve text path
    dir_name = os.path.dirname(__file__)
    folder_path = os.path.join(dir_name,
                               '../../text/%s/Coincidences_2D/' % window.data_sets)
    mkdir_p(folder_path)
    # Plot data and save matrices in text
    fig = plt.figure()
    for i, bus in enumerate(module_order):
        ce_bus = ce[ce.Bus == bus]
        number_events = ce_bus.shape[0]
        events_per_s = round(number_events/duration, 4)
        sub_title = ('Bus %d\n(%d events, %f events/s)' % (bus, number_events,
                                                           events_per_s)
                     )
        plt.subplot(3, 3, i+1)
        fig, h = plot_2D_bus(fig, sub_title, ce_bus, vmin, vmax, duration)
        # Save matrix in text
        path = folder_path + 'Bus_' + str(bus) + '.txt'
        if window.createText.isChecked():
            np.savetxt(path, h, fmt="%d", delimiter=",")
    fig = set_figure_properties(fig, title, height, width)
    return fig


# =============================================================================
# Coincidence Histogram (3D)
# =============================================================================

def Coincidences_3D_plot(df, data_sets, window):
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Perform initial filters
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    df = filter_ce_clusters(window, df)
    if data_sets == 'mvmelst_039.zip':
        df = df[df.Time < 1.5e12]
    # Initiate 'voxel_id -> (x, y, z)'-mapping
    detector_vec = get_detector_mappings()
    # Initiate border lines
    b_traces = initiate_detector_border_lines(detector_vec)
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wCh', 'gCh', 'Bus']].values,
                              bins=(80, 40, 9),
                              range=((0, 80), (80, 120), (0, 9))
                              )
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    detector_names = ['ILL', 'ESS_CLB', 'ESS_PA']
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 9):
                detector = detector_vec[bus//3]
                over_min = H[wCh, gCh-80, bus] > min_count
                under_max = H[wCh, gCh-80, bus] <= max_count
                if over_min and under_max:
                    coord = detector[flip_bus(bus % 3), gCh, flip_wire(wCh)]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    loc += 1
                    labels.append('Detector: ' + detector_names[(bus//3)]
                                  + '<br>'
                                  + 'Module: ' + str(bus) + '<br>'
                                  + 'WireChannel: ' + str(wCh) + '<br>'
                                  + 'GridChannel: ' + str(gCh) + '<br>'
                                  + 'Counts: ' + str(H[wCh, gCh-80, bus])
                                  )
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[2],
                               y=hist[0],
                               z=hist[1],
                               mode='markers',
                               marker=dict(size=5,
                                           color=np.log10(hist[3]),
                                           colorscale='Jet',
                                           opacity=1,
                                           colorbar=dict(thickness=20,
                                                         title='log10(counts)'
                                                         ),
                                           #cmin=0,
                                           #cmax=2.5
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=1,
                                 specs=[[{'is_3d': True}]]
                                 )
    # Insert histogram
    fig.append_trace(MG_3D_trace, 1, 1)
    # Insert vessel borders
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 1)
    # Assign layout with axis labels, title and camera angle
    a = 1
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                  )
    fig['layout']['scene1']['xaxis'].update(title='z [m]')
    fig['layout']['scene1']['yaxis'].update(title='x [m]')
    fig['layout']['scene1']['zaxis'].update(title='y [m]')
    fig['layout'].update(title='Coincidences (3D)<br>' + str(data_sets))
    fig['layout']['scene1']['camera'].update(camera)
    fig.layout.showlegend = False
    # If in plot He3-tubes histogram, return traces, else save HTML and plot
    if data_sets == '':
        return b_traces, hist[0], hist[1], hist[2], np.log10(hist[3])
    else:
        py.offline.plot(fig,
                        filename='../Results/HTML_files/Ce3Dhistogram.html',
                        auto_open=True)
        pio.write_image(fig, '../Results/HTML_files/Ce3Dhistogram.pdf')


# =============================================================================
# Coincidence Histogram (Front, Top, Side)
# =============================================================================

def Coincidences_Front_Top_Side_plot(df, data_sets, module_order,
                                     number_of_detectors, window):
    # Ensure we only plot coincident events
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    df = filter_ce_clusters(window, df)
    # Define figure and set figure properties
    fig = plt.figure()
    title = ('Coincident events (Front, Top, Side)' +
             '\nData set(s): %s' % data_sets
             )
    height = 4
    width = 14
    fig = set_figure_properties(fig, title, height, width)
    if df.shape[0] != 0:
        vmin = 1
        vmax = df.shape[0] // 200 + 5
    else:
        vmin = 1
        vmax = 1
    # Retrieve text path
    dir_name = os.path.dirname(__file__)
    folder_path = os.path.join(dir_name,
                               '../../text/%s/Projections_2D/' % window.data_sets)
    mkdir_p(folder_path)
    # Plot front view
    plt.subplot(1, 3, 1)
    __, h_front = plot_2D_Front(module_order, df, fig, number_of_detectors, vmin, vmax)
    path_front = folder_path + 'Front.txt'
    if window.createText.isChecked():
        np.savetxt(path_front, h_front, fmt="%d", delimiter=",")
    # Plot top view
    plt.subplot(1, 3, 2)
    __, h_top = plot_2D_Top(module_order, df, fig, number_of_detectors, vmin, vmax)
    path_top = folder_path + 'Top.txt'
    if window.createText.isChecked():
        np.savetxt(path_top, h_top, fmt="%d", delimiter=",")
    # Plot side view
    plt.subplot(1, 3, 3)
    __, h_side = plot_2D_Side(module_order, df, fig, number_of_detectors, vmin, vmax)
    path_side = folder_path + 'Side.txt'
    if window.createText.isChecked():
        np.savetxt(path_side, h_side, fmt="%d", delimiter=",")
    return fig


# =============================================================================
# Coincidence Histogram - Front
# =============================================================================


def plot_2D_Front(bus_vec, df, fig, number_of_detectors, vmin, vmax):
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    h, *_ = plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1,
                       df_tot.gCh,
                       bins=[12*number_of_detectors + 8, 40],
                       range=[[0.5, 12*number_of_detectors + 0.5 + 8],
                              [0.5, 40.5]
                              ],
                       norm=LogNorm(), cmap='jet', vmin=vmin, vmax=vmax
                       )
    title = 'Front view'
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    xlabel = 'Layer'
    ylabel = 'Grid'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True,
                  locs_x=locs_x, ticks_x=ticks_x)
    plt.colorbar()
    return fig, h

# =============================================================================
# Coincidence Histogram - Top
# =============================================================================


def plot_2D_Top(bus_vec, df, fig, number_of_detectors, vmin, vmax):
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_tot = pd.concat([df_tot, df_clu])
    h, *_ = plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1,
                       df_tot['wCh'] % 20 + 1,
                       bins=[12*number_of_detectors + 8, 20],
                       range=[[0.5, 12*number_of_detectors + 0.5 + 8],
                              [0.5, 20.5]
                              ],
                       norm=LogNorm(), cmap='jet', vmin=vmin, vmax=vmax
                       )
    title = 'Top view'
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    xlabel = 'Layer'
    ylabel = 'Wire'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True,
                  locs_x=locs_x, ticks_x=ticks_x)
    plt.colorbar()
    return fig, h


# =============================================================================
# Coincidence Histogram - Side
# =============================================================================


def plot_2D_Side(bus_vec, df, fig, number_of_detectors, vmin, vmax):

    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    h, *_ = plt.hist2d(df_tot['wCh'] % 20 + 1, df_tot['gCh'],
                       bins=[20, 40],
                       range=[[0.5, 20.5], [0.5, 40.5]],
                       norm=LogNorm(),
                       cmap='jet', vmin=vmin, vmax=vmax
                       )

    title = 'Side view'
    xlabel = 'Wire'
    ylabel = 'Grid'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True)
    plt.colorbar()
    return fig, h


# =============================================================================
# Coincidence Histogram (2D) - Iterate through all energies
# =============================================================================


def Coincidences_2D_plot_all_energies(window):
    def get_energy(element):
        start = element.find('Calibration_')+len('Calibration_')
        stop = element.find('_meV')
        return float(element[start:stop])

    def append_folder_and_files(folder, files):
        folder_vec = np.array(len(files)*[folder])
        return np.core.defchararray.add(folder_vec, files)
    # Declare all input-paths
    dir_name = os.path.dirname(__file__)
    HF_folder = os.path.join(dir_name, '../../Clusters/MG/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder)
                         if file[-3:] == '.h5'])
    HF_files_sorted = sorted(HF_files, key=lambda element: get_energy(element))
    Van_3x3_HF_clusters = append_folder_and_files(HF_folder, HF_files_sorted)
    HR_folder = os.path.join(dir_name, '../../Clusters/MG/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder)
                         if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: get_energy(element))
    Van_3x3_HR_clusters = append_folder_and_files(HR_folder, HR_files_sorted)
    input_paths = np.concatenate((Van_3x3_HR_clusters, Van_3x3_HF_clusters),
                                 axis=None)
    # Declare output folder
    output_folder = os.path.join(dir_name, '../../Results/2D_coincidences/')
    # Iterate through all data
    module_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for input_path in input_paths:
        # Import meta data
        calibration = import_MG_calibration(input_path)
        # Import data
        df_MG = import_MG_coincident_events(input_path)
        df_MG_reduced = filter_ce_clusters(window, df_MG)
        # Plot Coincidences 2D histogram
        fig = Coincidences_2D_plot(df_MG_reduced, calibration, module_order,
                                   window)
        # Save figure
        fig.savefig(output_folder + calibration + '.pdf', bbox_inches='tight')
        plt.close()







