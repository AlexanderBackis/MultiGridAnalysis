# =======  LIBRARIES  ======= #
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly as py
import numpy as np
import plotly.graph_objs as go
import pandas as pd

from Plotting.HelperFunctions import get_duration, set_figure_properties
from Plotting.HelperFunctions import stylize
from Plotting.HelperFunctions import get_detector_mappings, flip_bus, flip_wire
from Plotting.HelperFunctions import initiate_detector_border_lines

# =============================================================================
# Coincidence Histogram (2D)
# =============================================================================


def Coincidences_2D_plot(ce, data_sets, module_order):
    def plot_2D_bus(fig, sub_title, ce, vmin, vmax, duration):
        plt.hist2d(ce.wCh, ce.gCh, bins=[80, 40],
                   range=[[-0.5, 79.5], [79.5, 119.5]],
                   vmin=vmin, vmax=vmax, norm=LogNorm(), cmap='jet')
        xlabel = 'Wire [Channel number]'
        ylabel = 'Grid [Channel number]'
        fig = stylize(fig, xlabel, ylabel, title=sub_title, colorbar=True)
        return fig

    fig = plt.figure()
    duration = get_duration(ce)
    title = 'Coincident events (2D)\nData set(s): %s' % data_sets
    height = 12
    width = 14
    # Ensure only coincident events are plotted
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    vmin = 1
    vmax = ce.shape[0] // 4500
    for i, bus in enumerate(module_order):
        ce_bus = ce[ce.Bus == bus]
        number_events = ce_bus.shape[0]
        events_per_s = round(number_events/duration, 4)
        sub_title = ('Bus %d\n(%d events, %f events/s)' % (bus, number_events,
                                                           events_per_s)
                     )
        plt.subplot(3, 3, i+1)
        fig = plot_2D_bus(fig, sub_title, ce_bus, vmin, vmax, duration)
    fig = set_figure_properties(fig, title, height, width)
    return fig


# =============================================================================
# Coincidence Histogram (3D)
# =============================================================================

def Coincidences_3D_plot(df, data_sets):
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Perform initial filters
    df = df[(df.wCh != -1) & (df.gCh != -1)]
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
                                  + 'Counts: ' + str(H[wCh, gCh-80, bus]))
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
    a = 0.92
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


# =============================================================================
# Coincidence Histogram (Front, Top, Side)
# =============================================================================

def Coincidences_Front_Top_Side_plot(df, data_sets, module_order,
                                     number_of_detectors):
    # Ensure we only plot coincident events
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    # Define figure and set figure properties
    fig = plt.figure()
    title = ('Coincident events (Front, Top, Side)' +
             '\nData set(s): %s' % data_sets
             )
    height = 4
    width = 14
    fig = set_figure_properties(fig, title, height, width)
    # Plot front view
    plt.subplot(1, 3, 1)
    plot_2D_Front(module_order, df, fig, number_of_detectors)
    # Plot top view
    plt.subplot(1, 3, 2)
    plot_2D_Top(module_order, df, fig, number_of_detectors)
    # Plot side view
    plt.subplot(1, 3, 3)
    plot_2D_Side(module_order, df, fig, number_of_detectors)
    return fig


def plot_2D_Front(bus_vec, df, fig, number_of_detectors):
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1,
               df_tot.gCh,
               bins=[12*number_of_detectors + 8, 40],
               range=[[0.5, 12*number_of_detectors + 0.5 + 8],
                      [0.5, 40.5]
                      ],
               norm=LogNorm(), cmap='jet'
               )
    title = 'Front view'
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    xlabel = 'Layer'
    ylabel = 'Grid'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True,
                  locs_x=locs_x, ticks_x=ticks_x)


def plot_2D_Top(bus_vec, df, fig, number_of_detectors):
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_tot = pd.concat([df_tot, df_clu])
    plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1,
               df_tot['wCh'] % 20 + 1,
               bins=[12*number_of_detectors + 8, 20],
               range=[[0.5, 12*number_of_detectors + 0.5 + 8],
                      [0.5, 20.5]
                      ],
               norm=LogNorm(), cmap='jet')
    title = 'Top view'
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    xlabel = 'Layer'
    ylabel = 'Wire'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True,
                  locs_x=locs_x, ticks_x=ticks_x)
    return fig


def plot_2D_Side(bus_vec, df, fig, number_of_detectors):

    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    plt.hist2d(df_tot['wCh'] % 20 + 1, df_tot['gCh'],
               bins=[20, 40],
               range=[[0.5, 20.5], [0.5, 40.5]],
               norm=LogNorm(),
               cmap='jet'
               )

    title = 'Side view'
    xlabel = 'Wire'
    ylabel = 'Grid'
    fig = stylize(fig, xlabel, ylabel, title=title, colorbar=True)
    return fig






