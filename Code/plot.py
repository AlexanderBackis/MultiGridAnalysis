#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:18:00 2018

@author: alexanderbackis
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# =======  LIBRARIES  ======= #
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
from scipy.optimize import curve_fit
import h5py
import matplotlib
import shutil
import imageio
import webbrowser
import scipy
from Plotting.HelperFunctions import get_detector_mappings
#from Plotting.Coincidences import Coincidences_3D_plot
matplotlib.use('MacOSX')

    
    
# =============================================================================
# PHS (1D)
# =============================================================================
        

def PHS_1D_plot(df, data_sets, window=None):
    if window is not None:
        number_bins = int(window.tofBins.text())
    else:
        number_bins = 300
    fig = plt.figure()
    #plt.subplot(1, 3, 1)
    name = 'PHS (1D)\nData set(s): ' + str(data_sets)
    height = 5
    width = 10
    fig.set_figheight(height)
    fig.set_figwidth(width)
    fig.suptitle(data_sets, y=1.01)
    fig.suptitle(name)
    #plt.grid(True, which='major', zorder=0)
    #plt.grid(True, which='minor', linestyle='--', zorder=0)
    #e_single = df[(df.Channel == window.Channel.value()) & (df.Bus == window.Module.value())]
    #plt.hist(e_single.ADC, bins=number_bins, range=[0, 4400], histtype='step',
    #         color='black', zorder=5)
    #plt.xlabel("Charge [ADC channels]")
    #plt.ylabel("Intensity [Counts]")
    #plt.yscale('log')
    #plt.title('Bus: %d, Channel: %d' % (window.Module.value(), window.Channel.value()))
    plt.subplot(1, 2, 1)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    e_wires = df[(df.Channel <= 79)]
    plt.hist(e_wires.ADC, bins=number_bins, range=[0, 4400], histtype='step',
             color='black', zorder=5)
    plt.xlabel("Charge [ADC channels]")
    plt.ylabel("Intensity [Counts]")
    plt.yscale('log')
    plt.title('Wires')
    plt.subplot(1, 2, 2)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    e_wires = df[(df.Channel >= 80)]
    plt.hist(e_wires.ADC, bins=number_bins, range=[0, 4400], histtype='step',
             color='black', zorder=5)
    plt.xlabel("Charge [ADC channels]")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.title('Grids')
    return fig

# =============================================================================
# PHS (2D)
# =============================================================================

def PHS_2D_plot(df, data_sets, bus_vec, number_of_detectors):
    fig = plt.figure()
    name = name = 'PHS (2D)\nData set(s): ' + str(data_sets)
    buses_per_row = None
    number_of_detectors = None
    if len(bus_vec) < 3:
        buses_per_row = len(bus_vec)
        number_of_detectors = 1
        figwidth = 14 * len(bus_vec) / 3
    else:
        number_of_detectors = np.ceil(len(bus_vec) / 3)
        buses_per_row = 3
        figwidth = 14
     
    fig.suptitle(name + '\n\n', x=0.5, y=1.08)
    fig.set_figheight(4 * number_of_detectors)
    fig.set_figwidth(figwidth)
    vmin = 1
    vmax = df.shape[0] // (len(bus_vec) * 100)
    for loc, bus in enumerate(bus_vec):
        plot_PHS_bus(df, bus, loc, number_of_detectors, fig,
                     buses_per_row, vmin, vmax)        
    plt.tight_layout()
    return fig


def plot_PHS_bus(df, bus, loc, number_of_detectors, fig, buses_per_row,
                 vmin, vmax):
    df_red = df[df.Bus == bus]
    plt.subplot(1*number_of_detectors, buses_per_row, loc+1)
    plt.hist2d(df_red.Channel, df_red.ADC, bins=[120, 120], norm=LogNorm(),
               range=[[-0.5, 119.5], [0, 4400]], vmin=vmin, vmax=vmax,
               cmap='jet')
    plt.ylabel("Charge [ADC channels]")
    plt.xlabel("Channel [a.u.]")
    plt.colorbar()
    name = ('Bus ' + str(bus) + ', ' + str(df_red.shape[0]) + ' events\n' + 
            'Wire events = ' + str(df_red[df_red.Channel < 80].shape[0]) + 
            ', Grid events = ' + str(df_red[df_red.Channel >= 80].shape[0]))
    plt.title(name)


# =============================================================================
# PHS (Wires vs Grids)
# =============================================================================

def charge_scatter(df, bus, number_of_detectors, loc, fig, buses_per_row,
                   vmin, vmax):
    name = 'Bus ' + str(bus) + '\n(' + str(df.shape[0]) + ' events)'
    plt.subplot(number_of_detectors, buses_per_row, loc+1)
    plt.hist2d(df.wADC, df.gADC, bins=[200, 200],
               norm=LogNorm(), range=[[0, 5000], [0, 5000]],
               vmin=vmin, vmax=vmax, cmap='jet', zorder=5)
    plt.xlabel("wADC [ADC channels]")
    plt.ylabel("gADC [ADC channels]")
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)

    plt.colorbar()
    plt.title('PHS: wires vs grids')
    

def PHS_wires_vs_grids_plot(df, data_sets, module_order,
                            number_of_detectors):
    fig = plt.figure()
    name = name = 'PHS (Wires vs Grids)\nData set(s): ' + str(data_sets)
    buses_per_row = None
    number_of_detectors = None
    if len(module_order) < 3:
        buses_per_row = len(module_order)
        number_of_detectors = 1
        figwidth = 14 * len(module_order) / 3
    else:
        number_of_detectors = np.ceil(len(module_order) / 3)
        buses_per_row = 3
        figwidth = 14
    fig.suptitle(name, x=0.5, y=1.12)
    fig.set_figheight(4 * number_of_detectors)
    fig.set_figwidth(figwidth)
    vmin = 1
    vmax = df.shape[0] // (len(module_order) * 100)
    for loc, bus in enumerate(module_order):
        df_clu = df[df.Bus == bus]
        charge_scatter(df_clu, bus, number_of_detectors, loc, fig,
                       buses_per_row, vmin, vmax)
    plt.tight_layout()
    return fig


# =============================================================================
# Plot all PHS
# =============================================================================

def plot_all_PHS():
    # Find all paths
    dir_name = os.path.dirname(__file__)
    HF_folder = os.path.join(dir_name, '../Clusters/MG_new/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder) if file[-3:] == '.h5'])
    Van_3x3_HF_clusters = np.core.defchararray.add(np.array(len(HF_files)*[HF_folder]), HF_files)
    HR_folder = os.path.join(dir_name, '../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder) if file[-3:] == '.h5'])
    Van_3x3_HR_clusters = np.core.defchararray.add(np.array(len(HR_files)*[HR_folder]), HR_files)
    input_paths = np.concatenate((Van_3x3_HR_clusters, Van_3x3_HF_clusters), axis=None)
    # Declare parameters
    module_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    number_of_detectors = 3
    window = None
    for input_path in input_paths:
        # Import data
        e = pd.read_hdf(input_path, 'events')
        ce = pd.read_hdf(input_path, 'coincident_events')
        calibration = pd.read_hdf(input_path, 'calibration')['calibration'].iloc[0]
        # Produce histograms
        fig1 = PHS_wires_vs_grids_plot(ce, calibration, module_order, number_of_detectors)
        fig2 = PHS_2D_plot(e, calibration, module_order, number_of_detectors)
        fig3 = PHS_1D_plot(e, calibration, window)
        # Define output paths
        output_path_1 = os.path.join(dir_name, '../Results/PHS/PHS_1D/%s.pdf' % calibration)
        output_path_2 = os.path.join(dir_name, '../Results/PHS/PHS_2D/%s.pdf' % calibration)
        output_path_3 = os.path.join(dir_name, '../Results/PHS/PHS_wires_vs_grids/%s.pdf' % calibration)
        # Save histograms
        fig3.savefig(output_path_1, bbox_inches='tight')
        fig2.savefig(output_path_2, bbox_inches='tight')
        fig1.savefig(output_path_3, bbox_inches='tight')




     
# =============================================================================
# Coincidence Histogram (2D)
# =============================================================================


def Coincidences_2D_plot(df, data_sets, module_order, number_of_detectors):
    fig = plt.figure()
    if data_sets == "['mvmelst_039.mvmelst']":
        df = df[df.Time < 1.5e12]
        df = df[df.d != -1]

    start_time = df.head(1)['Time'].values[0]
    end_time = df.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9

    name = 'Coincident events (2D)\nData set(s): ' + str(data_sets)
    buses_per_row = None
    number_of_detectors = None
    if len(module_order) < 3:
        buses_per_row = len(module_order)
        number_of_detectors = 1
        figwidth = 14 * len(module_order) / 3
    else:
        number_of_detectors = np.ceil(len(module_order) / 3)
        buses_per_row = 3
        figwidth = 14
    fig.suptitle(name, x=0.5, y=1.09)
    fig.set_figheight(4 * number_of_detectors)
    fig.set_figwidth(figwidth)
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    vmin = 1
    vmax = df.shape[0] // (len(module_order) * 500)
    for loc, bus in enumerate(module_order):
        df_bus = df[df.Bus == bus]
        plot_2D_bus(df_bus, bus, number_of_detectors, loc, fig, buses_per_row,
                    vmin, vmax, duration)
    plt.tight_layout()
    fig.show()


def plot_2D_bus(df, bus, number_of_detectors, loc, fig, buses_per_row,
                vmin, vmax, duration):
    plt.subplot(number_of_detectors, buses_per_row, loc+1)
    plt.hist2d(df.wCh, df.gCh, bins=[80, 40],
               range=[[-0.5, 79.5], [79.5, 119.5]],
               vmin=vmin, vmax=vmax, norm=LogNorm(), cmap='jet')
    plt.xlabel("Wire [Channel number]")
    plt.ylabel("Grid [Channel number]")
    plt.colorbar()
    name = ('Bus ' + str(bus) + '\n(' + str(df.shape[0]) + ' events, '
            + str(round(df.shape[0]/duration, 4)) + ' events/s)')
    plt.title(name)

# =============================================================================
# Coincidence Histogram (3D)
# ============================================================================= 
    
def Coincidences_3D_plot(df, data_sets):
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Perform initial filters
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    # Declare offsets
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
    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2])/(ILL_C[6][0]-ILL_C[5][0]))
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2])/(ESS_1_C[4][0]-ESS_1_C[3][0]))
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2])/(ESS_2_C[2][0]-ESS_2_C[1][0]))
    # Initiate detector mappings
    detector_1 = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    detector_2 = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    detector_3 = create_ess_channel_to_coordinate_map(theta_3, offset_3)
    detector_vec = [detector_1, detector_2, detector_3]
    # Initiate border lines   
    pairs = [[[80, 0], [80, 60]],
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
    
    b_traces = []
    for bus in range(3, 9):
        detector = detector_vec[bus//3]
        for pair in pairs:
            x_vec = []
            y_vec = []
            z_vec = []
            for loc in pair:
                gCh = loc[0]
                wCh = loc[1]
                coord = detector[bus%3, gCh, wCh]
                x_vec.append(coord['x'])
                y_vec.append(coord['y'])
                z_vec.append(coord['z'])
                
      
            b_trace = go.Scatter3d(x=z_vec,
                                   y=x_vec,
                                   z=y_vec,
                                   mode='lines',
                                   line = dict(
                                                color='rgba(0, 0, 0, 0.5)',
                                                width=5)
                                    )
            b_traces.append(b_trace)
       
    detector = detector_vec[0]
    pairs_2 = [[[80, 0, 0], [80, 60, 2]],
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
    for pair in pairs_2:
        x_vec = []
        y_vec = []
        z_vec = []
        for loc in pair:
            gCh = loc[0]
            wCh = loc[1]
            bus = loc[2]
            coord = detector[bus%3, gCh, wCh]
            x_vec.append(coord['x'])
            y_vec.append(coord['y'])
            z_vec.append(coord['z'])
                
      
        b_trace = go.Scatter3d(x=z_vec,
                               y=x_vec,
                               z=y_vec,
                               mode='lines',
                               line = dict(
                                       color='rgba(0, 0, 0, 0.5)',
                                       width=5
                                        )
                                )
        b_traces.append(b_trace)
        
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wCh','gCh', 'Bus']].values,
                              bins=(80, 40, 9), 
                              range=((0, 80), (80, 120), (0,9))
                             )
    # Insert results into an array
    flip_bus = {0: 2, 1: 1, 2: 0}
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
    
    maximum_voxel = None
    max_hist = 0
    hist = [[], [], [], []]
    loc = 0
    labels = []
    detector_names = ['ILL', 'ESS_CLB', 'ESS_PA']
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 9):
                detector = detector_vec[bus//3]
                if H[wCh, gCh-80, bus] > min_count and H[wCh, gCh-80, bus] <= max_count:
                    coord = detector[flip_bus[bus%3], gCh, flip_wire(wCh)]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    if H[wCh, gCh-80, bus] > max_hist:
                    	max_hist = H[wCh, gCh-80, bus]
                    	maximum_voxel = [bus, wCh, gCh]
                    loc = loc + 1
                    labels.append('Detector: ' + detector_names[(bus//3)] + '<br>'
                              + 'Module: ' + str(bus) + '<br>'
                              + 'WireChannel: ' + str(wCh) + '<br>'
                              + 'GridChannel: ' + str(gCh) + '<br>'
                              + 'Counts: ' + str(H[wCh, gCh-80, bus]))

                        
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[2],
                               y=hist[0],
                               z=hist[1],
                               mode='markers',
                               marker=dict(
                                       size=5,
                                       color = np.log10(hist[3]),
                                       colorscale = 'Jet',
                                       opacity=1,
                                       colorbar=dict(thickness=20,
                                                     title = 'log10(counts)'
                                                     ),
                                       ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
                        
    color_lim_trace = go.Scatter3d(x=[5.35],
                                   y=[-0.9],
                                   z=[-3.07],
                                   mode='markers',
                                   marker=dict(
                                           size=5,
                                           color = 'rgb(255,255,255)',
                                           opacity=1,
                                           line = dict(
                                                   color = 'rgb(255,255,255)',
                                                   width = 1
                                                   )
                                        ),
                                    )
                                       
    
                                     
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=1,
                                 specs=[ 
                                    [{'is_3d': True}]
                                    ]
                                 )
  
    fig.append_trace(MG_3D_trace, 1, 1)
    #fig.append_trace(color_lim_trace, 1, 1)
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 1)
  
    a = 0.92
    camera = dict(
                 up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0),
                 eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                 )
    fig['layout']['scene1']['xaxis'].update(title='z [m]') # range=[5.28, 5.66]
    fig['layout']['scene1']['yaxis'].update(title='x [m]') # range=[-1.64, -0.6]
    fig['layout']['scene1']['zaxis'].update(title='y [m]') # range=[-3.13, -2.2]
    fig['layout'].update(title='Coincidences (3D)<br>' + str(data_sets))
    fig['layout']['scene1']['camera'].update(camera)
    fig.layout.showlegend = False
    if data_sets == '':
        print(maximum_voxel)
        return b_traces, hist[0], hist[1], hist[2], np.log10(hist[3]), maximum_voxel
    else:
        py.offline.plot(fig, filename='../Results/HTML_files/Ce3Dhistogram.html', 
                        auto_open=True)

    
# =============================================================================
# Coincidence Histogram (Front, Top, Side)
# =============================================================================


def Coincidences_Front_Top_Side_plot(df, data_sets, module_order,  
                                     number_of_detectors):
    name = 'Coincident events (Front, Top, Side)\nData set(s): ' + str(data_sets)
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(14)
    fig.suptitle(name, x=0.5, y=1.08)

    plt.subplot(1, 3, 1)
    plot_2D_Front(module_order, df, fig, number_of_detectors)
    plt.subplot(1, 3, 2)
    plot_2D_Top(module_order, df, fig, number_of_detectors)
    plt.subplot(1, 3, 3)
    plot_2D_Side(module_order, df, fig, number_of_detectors)
    
    plt.tight_layout()
    fig.show()

    
def plot_2D_Front(bus_vec, df, fig, number_of_detectors):
    name = 'Front view'
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1, df_tot.gCh,
               bins=[12*number_of_detectors + 8, 40],
               range=[[0.5, 12*number_of_detectors + 0.5 + 8], [0.5, 40.5]],
               norm=LogNorm(), cmap='jet'
               )
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Grid")
    plt.xticks(locs_x, ticks_x)
    plt.title(name)
    

def plot_2D_Top(bus_vec, df, fig, number_of_detectors):
    name = 'Top view'
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]
        df_clu['wCh'] += (80 * i) + (i // 3) * 80
        df_tot = pd.concat([df_tot, df_clu])  
    plt.hist2d(np.floor(df_tot['wCh'] / 20).astype(int) + 1, df_tot['wCh'] % 20 + 1, 
               bins=[12*number_of_detectors + 8, 20], 
               range=[[0.5, 12*number_of_detectors + 0.5 + 8], [0.5, 20.5]], 
               norm=LogNorm(), cmap='jet')
    locs_x = [1, 12, 17, 28, 33, 44]
    ticks_x = [1, 12, 13, 25, 26, 38]
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Wire")
    plt.xticks(locs_x, ticks_x)
    plt.title(name)
    

def plot_2D_Side(bus_vec, df, fig, number_of_detectors):
    name = 'Side view'
    df_tot = pd.DataFrame()
    for i, bus in enumerate(bus_vec):
        df_clu = df[df.Bus == bus]    
        df_clu['gCh'] += (-80 + 1)
        df_tot = pd.concat([df_tot, df_clu])
    plt.hist2d(df_tot['wCh'] % 20 + 1, df_tot['gCh'],
               bins=[20, 40], range=[[0.5,20.5],[0.5,40.5]], 
               norm=LogNorm(), cmap='jet')
    plt.colorbar()
    plt.xlabel("Wire")
    plt.ylabel("Grid")
    plt.title(name)
    
    
# =============================================================================
# Multiplicity
# =============================================================================       
    
def plot_2D_multiplicity(df, number_of_detectors, bus, loc,
                         fig, buses_per_row, vmin, vmax):
    df_clu = df[df.Bus == bus]
    m_range = [0, 5, 0, 5]

    plt.subplot(number_of_detectors, buses_per_row, loc+1)
    hist, xbins, ybins, im = plt.hist2d(df_clu.wM, df_clu.gM, 
                                        bins=[m_range[1]-m_range[0]+1, m_range[3]-m_range[2]+1], 
                                        range=[[m_range[0], m_range[1]+1], [m_range[2], m_range[3]+1]],
                                        norm=LogNorm(),
                                        vmin=vmin,
                                        vmax=vmax,
                                        cmap='jet')
    tot = df_clu.shape[0]
    font_size = 12
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5,ybins[i]+0.5, 
                         str(format(100*(round((hist[j,i]/tot),3)),'.1f')) + 
                         "%", color="w", ha="center", va="center", 
                         fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])
        
    ticks_x = np.arange(m_range[0],m_range[1]+1,1)
    locs_x = np.arange(m_range[0] + 0.5, m_range[1]+1.5,1)
    ticks_y = np.arange(m_range[2],m_range[3]+1,1)
    locs_y = np.arange(m_range[2] + 0.5, m_range[3]+1.5,1)
    
    plt.xticks(locs_x, ticks_x)
    plt.yticks(locs_y, ticks_y)
    plt.xlabel("Wire Multiplicity")
    plt.ylabel("Grid Multiplicity")

    plt.colorbar()
    plt.tight_layout()
    name = 'Bus ' + str(bus) + '\n(' + str(df_clu.shape[0]) + ' events)' 
    plt.title('Multiplicity distribution')


def Multiplicity_plot(df, data_sets, module_order, 
                      number_of_detectors):
    name = 'Multiplicity\nData set(s): ' + str(data_sets)
    fig = plt.figure()
    
    buses_per_row = None
    number_of_detectors = None
    if len(module_order) < 3:
        buses_per_row = len(module_order)
        number_of_detectors = 1
        figwidth = 14 * len(module_order) / 3
    else:
        number_of_detectors = np.ceil(len(module_order) / 3)
        buses_per_row = 3
        figwidth = 14

    fig.suptitle(name, x=0.5, y=1.08)
    fig.set_figheight(4*number_of_detectors)
    fig.set_figwidth(figwidth)
    vmin = 1
    vmax = df.shape[0] // (len(module_order) * 1)
    for loc, bus in enumerate(module_order):
        plot_2D_multiplicity(df, number_of_detectors, bus, loc,
                             fig, buses_per_row, vmin, vmax)

    plt.tight_layout()
    fig.show()
    

# =============================================================================
# ToF histogram
# =============================================================================
    
def ToF_plot(df, data_sets, calibration, E_i, measurement_time, isCLB, isPureAl,
             window, interval=None):
    name = 'ToF\n' + calibration
    df = filter_ce_clusters(window, df)
    df = df[df.d != -1]
    fig = plt.figure()
    rnge = [window.ToF_min.value(), window.ToF_max.value()]
    number_bins = 100
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    MG_over_He3 = 0
    MG_over_MG_back = 0
    if window.compare_he3.isChecked():
        if isPureAl:
            df = df[(df.Bus >= 6) & (df.Bus <= 8)]
        elif isCLB:
            df = df[(df.Bus >= 3) & (df.Bus <= 5)]
        df_He3 = load_He3_h5(calibration)
        #T_0 = get_T0(calibration, E_i)
        frame_shift = get_frame_shift(E_i) * 1e6
        He3_hist, He3_bins = np.histogram(df_He3.ToF, bins=number_bins)
        MG_hist, MG_bins = np.histogram(df.ToF * 62.5e-9 * 1e6 + frame_shift,
                                        bins=number_bins)
        He3_bin_centers = 0.5 * (He3_bins[1:] + He3_bins[:-1])
        MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
        # Find He3 duration
        dir_name = os.path.dirname(__file__)
        m_id = str(find_He3_measurement_id(calibration))
        raw_path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
        file = h5py.File(raw_path, 'r')
        He3_measurement_time = file['entry']['duration'].value
        area_time_norm = get_area_time_norm(calibration, E_i, isCLB, isPureAl,
                                            measurement_time,
                                            He3_measurement_time)
        # Plot data
        plt.plot(MG_bin_centers, MG_hist/area_time_norm, color='red', label='Multi-Grid', zorder=3)
        plt.plot(He3_bin_centers, He3_hist, color='blue', label='$^3$He-tubes', zorder=4)
        # Get background
        back_hist, back_bins, MG_over_He3, MG_over_MG_back, error_MG_over_He3 = get_ToF_background(window, calibration, E_i, isCLB, isPureAl,
                                                                                                   frame_shift, measurement_time, number_bins,
                                                                                                   He3_measurement_time, interval, df_He3,
                                                                                                   area_time_norm, df)
        plt.plot(back_bins, back_hist, color='green', label='MG background', zorder=5)
        if interval is not None:
            plt.axvline(x=interval[0], color='black', label='Flat background estimation')
            plt.axvline(x=interval[1], color='black', label=None)
        plt.yscale('log')
        plt.legend()

        # Export histograms to text-files
        dir_name = os.path.dirname(__file__)
        MG_path = os.path.join(dir_name, '../Results/Histograms/MG/ToF/MG_%s_meV.txt' % calibration)
        He3_path = os.path.join(dir_name, '../Results/Histograms/He3/ToF/He3_%s_meV.txt' % calibration)
        MG_dict = {'ToF [um]': MG_bin_centers,
                   'Signal [Normalized Counts]': MG_hist/area_time_norm,
                   'Background estimation [Normalized counts]': back_hist
                    }
        He3_dict = {'dE [meV]': He3_bin_centers,
                    'Signal [Normalized Counts]': He3_hist
                    }
        MG_df = pd.DataFrame(MG_dict)
        He3_df = pd.DataFrame(He3_dict)
        MG_df.to_csv(MG_path, index=None, sep=' ', mode='w', encoding='ascii')
        He3_df.to_csv(He3_path, index=None, sep=' ', mode='w', encoding='ascii')

    else:
        df = filter_ce_clusters(window, df)
        number_bins = int(window.tofBins.text())
        hist, bins, patches = plt.hist(df.ToF * 62.5e-9 * 1e6,
                                   	   bins=number_bins, range=rnge,
                                       log=True, color='darkviolet', zorder=3,
                                       alpha=0.4,
                                       )
        hist, bins, patches = plt.hist(df.ToF * 62.5e-9 * 1e6, bins=number_bins,
                                   	   range=rnge,
                                       log=True, color='black', zorder=4,
                                       histtype='step', label='MG'
                                       )
        name = 'ToF\n' + data_sets
    plt.xlabel('ToF [$\mu$s]')
    plt.ylabel('Counts')
    plt.title(name)
    return fig, MG_over_He3, MG_over_MG_back, error_MG_over_He3

def plot_all_energies_ToF(window, isCLB, isPureAl):
    # Declare ToF-intervals (us) for background comparison
    intervals_He3 = np.array([[45000, 48000],
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
                    [8e3, 12e3],  #HF
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
                    ])
    # Declare input-folders
    #dir_name = os.path.dirname(__file__)
    #HR_in = os.path.join(dir_name, '../Clusters/MG/HF_HR_clusters/V_HR/')
    #HF_in = os.path.join(dir_name, '../Clusters/MG/HF_HR_clusters/V_HF/')
    #HR_out = os.path.join(dir_name, '../Results/ToF/HR/')
    #HF_out = os.path.join(dir_name, '../Results/ToF/HF/')
    back_HR = [[], [], []]
    back_HF = [[], [], []]
    MG_over_MG_back_HR = []
    MG_over_MG_back_HF = []
    # Find all paths
    dir_name = os.path.dirname(__file__)
    HF_folder = os.path.join(dir_name, '../Clusters/MG_new/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder) if file[-3:] == '.h5'])
    HF_files_sorted = sorted(HF_files, key=lambda element: float(element[element.find('Calibration_')+len('Calibration_'):element.find('_meV')]))
    Van_3x3_HF_clusters = np.core.defchararray.add(np.array(len(HF_files)*[HF_folder]), HF_files_sorted)
    HR_folder = os.path.join(dir_name, '../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder) if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: float(element[element.find('Calibration_')+len('Calibration_'):element.find('_meV')]))
    Van_3x3_HR_clusters = np.core.defchararray.add(np.array(len(HR_files)*[HR_folder]), HR_files_sorted)
    input_paths = np.concatenate((Van_3x3_HR_clusters, Van_3x3_HF_clusters), axis=None)
    output_folder = os.path.join(dir_name, '../Results/ToF/')
    for input_path, interval in zip(input_paths, intervals_He3):
        df = pd.read_hdf(input_path, 'coincident_events')
        E_i = pd.read_hdf(input_path, 'E_i')['E_i'].iloc[0]
        calibration = pd.read_hdf(input_path, 'calibration')['calibration'].iloc[0]
        print(calibration)
        data_sets = pd.read_hdf(input_path, 'data_set')['data_set'].iloc[0]
        measurement_time = pd.read_hdf(input_path, 'measurement_time')['measurement_time'].iloc[0]
        fig, MG_over_He3, MG_over_MG_back, error_MG_over_He3 = ToF_plot(df, data_sets, calibration, E_i, measurement_time,
                                                                        isCLB, isPureAl, window, interval)
        fig.savefig(output_folder + 'ToF' + calibration + '.pdf')
        # Calculate error

        if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
            back_HF[0].append(E_i)
            back_HF[1].append(MG_over_He3)
            back_HF[2].append(error_MG_over_He3)
            MG_over_MG_back_HF.append(MG_over_MG_back)
        else:
            back_HR[0].append(E_i)
            back_HR[1].append(MG_over_He3)
            back_HR[2].append(error_MG_over_He3)
            MG_over_MG_back_HR.append(MG_over_MG_back)
        plt.close()

    print('MG over MG back HF')
    print(MG_over_MG_back_HF)
    print('MG over MG back HR')
    print(MG_over_MG_back_HR)



    fig = plt.figure()
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.errorbar(back_HR[0], MG_over_MG_back_HR, back_HR[2],
                 color='blue', ecolor='blue', capsize=5,
                 label='V_3x3_HR', zorder=5, fmt='o', linestyle='-'
                 )
    plt.errorbar(back_HF[0], MG_over_MG_back_HF, back_HF[2],
                 color='red', ecolor='red', capsize=5,
                 label='V_3x3_HF', zorder=5, fmt='o', linestyle='-'
                 )
    print('Average HR: %f' % (sum(back_HR[1])/len(back_HR[1])))
    print('Average HF: %f' % (sum(back_HF[1])/len(back_HF[1])))
    plt.xlabel('Energy [meV]')
    plt.ylabel('Background fraction (MG/$^3$He-tubes)')
    plt.title('Comparison between background in MG and $^3$He-tubes')
    plt.xscale('log')
    plt.legend()
    fig.show()

        
def get_ToF_background(window, calibration, E_i, isCLB, isPureAluminium,
                       frame_shift, measurement_time, number_bins,
                       He3_measurement_time, He3_interval, df_He3,
                       norm_MG_signal,
                       df_MG):
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, '../Clusters/MG/Background.h5')
    df = pd.read_hdf(path, 'coincident_events')
    df = filter_ce_clusters(window, df)
    df = df[df.Time < 1.5e12]
    df = df[df.d != -1]
    modules_to_exclude = []
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        if E_i < 450:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
    else:
        if E_i > 50:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])

    for bus in modules_to_exclude:
        df = df[df.Bus != bus]
    
    # He3 area 
    He3_area = 23.457226948780608
    # Multi-Grid area
    voxel_area = 0.0005434375
    MG_area_wire_row = voxel_area * 35
    MG_area_module = MG_area_wire_row * 4
    MG_area_detector = MG_area_module * 3
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        if E_i < 450:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row) - MG_area_module) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row - MG_area_module) * (34/40)

        else:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row)) * 34/40
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row) * 34/40

    else:
        if E_i > 50:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row) - MG_area_module) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row - MG_area_module) * (34/40)
        else:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row)) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row) * (34/40)
    
    if isPureAluminium:
        MG_area = MG_area_detector - MG_area_wire_row - voxel_area * 35 - (4 * 4 * voxel_area)  # Last part is because we remove the middle three grids and top grid


    area_frac = MG_area/He3_area
    time_frac = measurement_time/He3_measurement_time
    norm_time_area = area_frac * time_frac
    # Calculate background duration
    start_time = df.head(1)['Time'].values[0]
    end_time = df.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9
    # Calculate weights
    number_of_events = df.shape[0]
    events_per_s = number_of_events / duration
    events_s_norm = events_per_s / number_of_events
    weight = (1/norm_time_area) * events_s_norm * measurement_time
    # Histogram background
    MG_hist, MG_bins = np.histogram(df.ToF * 62.5e-9 * 1e6 + frame_shift,
                                    bins=number_bins)
    MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
    # Declare inteval in MG background
    MG_interval = [0, 12500]
    # Declare inter in MG from measurement data
    MG_meas_interval = He3_interval
    # Calculate background fraction
    if He3_interval is not None:
        counts_MG = df[((df.ToF * 62.5e-9 * 1e6) >= MG_interval[0]) &
                       ((df.ToF * 62.5e-9 * 1e6) <= MG_interval[1])
                       ].shape[0] / (MG_interval[1] - MG_interval[0])
        counts_He3 = df_He3[(df_He3.ToF >= He3_interval[0]) &
                            (df_He3.ToF <= He3_interval[1])
                            ].shape[0] / (He3_interval[1] - He3_interval[0])
        MG_over_He3 = (counts_MG / norm_time_area) / counts_He3
        # Calculate error
        a = df[((df.ToF * 62.5e-9 * 1e6) >= MG_interval[0]) &
               ((df.ToF * 62.5e-9 * 1e6) <= MG_interval[1])
               ].shape[0]
        b = df_He3[(df_He3.ToF >= He3_interval[0]) &
                   (df_He3.ToF <= He3_interval[1])
                   ].shape[0]
        db = np.sqrt(b)
        da = np.sqrt(a)
        error_MG_over_He3 = np.sqrt((da/a) ** 2 + (db/b) ** 2) * MG_over_He3






        # Calculate MG over MG back
        counts_MG_signal = df_MG[((df_MG.ToF * 62.5e-9 * 1e6 + frame_shift) >= MG_meas_interval[0]) &
                                 ((df_MG.ToF * 62.5e-9 * 1e6 + frame_shift) <= MG_meas_interval[1])
                                 ].shape[0]
        MG_over_MG_back = (counts_MG * weight) / (counts_MG_signal * (1/norm_time_area))

    return MG_hist*weight, MG_bin_centers, MG_over_He3, MG_over_MG_back, error_MG_over_He3

# =============================================================================
# Timestamp and Trigger
# =============================================================================

def Timestamp_plot(df, data_sets):
    name = 'Timestamp\nData set(s): ' + str(data_sets)
    fig = plt.figure()    
    event_number = np.arange(0, df.shape[0], 1)
    plt.title(name)
    plt.xlabel('Event number')
    plt.ylabel('Timestamp [TDC channels]')
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(event_number, df.Time, color='black', label='All events')
    glitches = df[(df.wM >= 80) & (df.gM >= 40)].Time
    plt.plot(glitches.index.tolist(), glitches, 'rx',
             label='Glitch events')
    plt.legend()
    plt.tight_layout()
    fig.show()
    


# =============================================================================
# E_i - E_f
# =============================================================================

def dE_plot(df, E_i, calibration, measurement_time,
            window, isCLB, isPA, number_bins=390):
    # Filter data
    df = df[df.d != -1]
    df = filter_ce_clusters(window, df)
    if isPA:
        df = df[(df.Bus >= 6) & (df.Bus <= 8)]
    elif isCLB:
        df = df[(df.Bus >= 3) & (df.Bus <= 5)]
    # Declare parameters
    T_0 = get_T0(calibration, E_i)
    t_off = get_t_off(calibration)
    t_off = get_t_off(calibration) * np.ones(df.shape[0])
    T_0 = get_T0(calibration, E_i) * np.ones(df.shape[0])
    frame_shift = get_frame_shift(E_i) * np.ones(df.shape[0])
    E_i = np.ones(df.shape[0]) * E_i
    ToF = df.ToF.values
    d = df.d.values
    # Calculate energy transfer
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp_new = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    df_temp_new = df_temp_new[df_temp_new['t_f'] > 0]
    hist_MG, bins = np.histogram(df_temp_new.dE, bins=number_bins, range=[-E_i[0], E_i[0]])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Calculate background
    hist_back = plot_dE_background_single(E_i[0], calibration,
                                          measurement_time, -E_i[0],
                                          E_i[0], window, isPA,
                                          number_bins, isCLB)
    # Plot
    fig = plt.figure()
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(bin_centers, hist_MG, color='black', label='Data', zorder=5)
    plt.plot(bin_centers, hist_back, color='green', label='Background estimation', zorder=5)
    plt.xlabel('E$_i$ - E$_f$ [meV]')
    plt.xlim([-E_i[0], E_i[0]])
    plt.ylabel('Counts')
    plt.yscale('log')    
    plt.title('Energy transfer\nData set(s): ' + calibration)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_dE_background_single(E_i, calibration, measurement_time, minimum,
                              maximum, window, isPureAluminium=False,
                              number_bins=390, isCLB=False):
    # Import background data
    dirname = os.path.dirname(__file__)
    clu_path = os.path.join(dirname, '../Clusters/MG/Background.h5')
    df = pd.read_hdf(clu_path, 'coincident_events')
    df = filter_ce_clusters(window, df)
    df = df[df.Time < 1.5e12]
    df = df[df.d != -1]
    modules_to_exclude = []
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        if E_i < 450:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])

    else:
        if E_i > 50:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])

    for bus in modules_to_exclude:
        df = df[df.Bus != bus]
    # Calculate background duration
    start_time = df.head(1)['Time'].values[0]
    end_time = df.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9
    # Calculate background
    t_off = get_t_off(calibration) * np.ones(df.shape[0])
    T_0 = get_T0(calibration, E_i) * np.ones(df.shape[0])
    frame_shift = get_frame_shift(E_i) * np.ones(df.shape[0])
    E_i = E_i * np.ones(df.shape[0])
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    # Calculate weights
    number_of_events = len(dE)
    events_per_s = number_of_events / duration
    events_s_norm = events_per_s / number_of_events
    weights = events_s_norm * measurement_time * np.ones(len(dE))
    # Histogram background
    dE_range = [minimum, maximum]
    MG_dE_hist, MG_bins = np.histogram(dE, bins=number_bins, range=dE_range,
                                       weights=weights)
    return MG_dE_hist

# =============================================================================
# Plot all dE
# =============================================================================

def plot_all_dE(isCLB, isPA, window, number_bins):
    # Find all paths
    dir_name = os.path.dirname(__file__)
    HF_folder = os.path.join(dir_name, '../Clusters/MG_new/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder) if file[-3:] == '.h5'])
    Van_3x3_HF_clusters = np.core.defchararray.add(np.array(len(HF_files)*[HF_folder]), HF_files)
    HR_folder = os.path.join(dir_name, '../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder) if file[-3:] == '.h5'])
    Van_3x3_HR_clusters = np.core.defchararray.add(np.array(len(HR_files)*[HR_folder]), HR_files)
    input_paths = np.concatenate((Van_3x3_HR_clusters, Van_3x3_HF_clusters), axis=None)
    # Iterate over all energies
    for input_path in input_paths:
        # Import data
        ce = pd.read_hdf(input_path, 'coincident_events')
        calibration = pd.read_hdf(input_path, 'calibration')['calibration'].iloc[0]
        measurement_time = pd.read_hdf(input_path, 'measurement_time')['measurement_time'].iloc[0]
        E_i = pd.read_hdf(input_path, 'E_i')['E_i'].iloc[0] 
        # Produce histograms
        fig = dE_plot(ce, E_i, calibration, measurement_time, window, isCLB, isPA, number_bins)
        # Define output paths
        output_path = os.path.join(dir_name, '../Results/Energy_transfer/%s.pdf' % calibration)
        # Save histograms
        fig.savefig(output_path, bbox_inches='tight')




# =============================================================================
# E_i - E_f (zoom peak)
# =============================================================================

def dE_plot_peak(df, data_sets, E_i, calibration, measurement_time,
                 scale_factor, df_back):
    T_0 = get_T0(calibration, E_i)
    t_off = get_t_off(calibration)
    # Keep only coincident events
    df = df[df.d != -1]
    dE_bins = 100
    dE_range = [-10*scale_factor, 10*scale_factor]
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    t_off = get_t_off(calibration) * np.ones(df.shape[0])
    T_0 = get_T0(calibration, E_i) * np.ones(df.shape[0])
    frame_shift = get_frame_shift(E_i) * np.ones(df.shape[0])
    E_i = E_i * np.ones(df.shape[0])
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp_new = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    df_temp_new = df_temp_new[df_temp_new['t_f'] > 0]
    hist_MG, bins = np.histogram(df_temp_new.dE, bins=dE_bins, range=dE_range)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Calculate background duration
    start_time = df_back.head(1)['Time'].values[0]
    end_time = df_back.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9
    # Calculate background
    t_off = get_t_off(calibration) * np.ones(df_back.shape[0])
    T_0 = get_T0(calibration, E_i[0]) * np.ones(df_back.shape[0])
    frame_shift = get_frame_shift(E_i[0]) * np.ones(df_back.shape[0])
    E_i = E_i[0] * np.ones(df_back.shape[0])
    ToF = df_back.ToF.values
    d = df_back.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    # Calculate weights
    number_of_events = len(dE)
    events_per_s = number_of_events / duration
    events_s_norm = events_per_s / number_of_events
    weights = events_s_norm * measurement_time * np.ones(len(dE))
    # Histogram background
    MG_back_hist, __ = np.histogram(dE, bins=dE_bins, range=dE_range, weights=weights)

    return hist_MG, bin_centers, MG_back_hist, 



# =============================================================================
# 13. Delta E
# =============================================================================
    
def dE_single(fig, name, df, data_set, E_i, sample, 
              left_edge=175, right_edge=220):
        df = filter_clusters(df)
        
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        bus_ranges = [[0,2], [3,5], [6,8]]
        color_vec = ['darkorange', 'magenta', 'blue']
        detectors = ['ILL', 'ESS_1', 'ESS_2']
        dE_bins = 400
        dE_range = [-E_i, E_i]
        
        name = ('13. ' + sample + ', ' + str(E_i) + 'meV')
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--',zorder=0)
                
        hist, bins, patches = plt.hist(df.dE, bins=dE_bins, range=dE_range, 
                                       log=LogNorm(), color='black', 
                                       histtype='step', zorder=2, 
                                       label='All detectors')
        
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        # Calculate background level
        x_l = bin_centers[left_edge]
        y_l = hist[left_edge]
        x_r = bin_centers[right_edge-1]
        y_r = hist[right_edge-1]
        par_back = np.polyfit([x_l, x_r], [y_l, y_r], deg=1)
        f_back = np.poly1d(par_back)
        xx_back = np.linspace(x_l, x_r, 100)
        yy_back = f_back(xx_back)
        
        plt.plot(xx_back, yy_back, 'orange', label='Background')
        
        bins_under_peak = abs(right_edge - 1 - left_edge)
        
        area_noise = ((abs(y_r - y_l) * bins_under_peak) / 2 
                      + bins_under_peak * y_l)
        
        area = sum(hist[left_edge:right_edge])
        peak_area = area - area_noise
        
        # Calculate HM
        peak = peakutils.peak.indexes(hist[left_edge:right_edge])
        plt.plot(bin_centers[left_edge:right_edge][peak],
                 hist[left_edge:right_edge][peak], 'bx', label='Maximum', 
                 zorder=5)
        M = hist[left_edge:right_edge][peak]
        xM = bin_centers[left_edge:right_edge][peak]
        print(xM)
        noise_level = yy_back[find_nearest(xx_back, xM)]
        print(noise_level)
        HM = (M-noise_level)/2 + noise_level

        # Calculate FWHM
        left_idx = find_nearest(hist[left_edge:left_edge+peak[0]], HM)
        right_idx = find_nearest(hist[left_edge+peak[0]:right_edge], HM)
        
        sl = []
        sr = []
        
        if hist[left_edge+left_idx] > HM:
            sl = [-1, 0]
        else:
            sl = [0, 1]
        
        if hist[left_edge+peak[0]+right_idx] < HM:
            rl = [-1, 0]
        else:
            rl = [0, 1]
        
        left_x = [bin_centers[left_edge+left_idx+sl[0]], 
                  bin_centers[left_edge+left_idx+sl[1]]]
        left_y = [hist[left_edge+left_idx+sl[0]], hist[left_edge+left_idx+sl[1]]]
        right_x = [bin_centers[left_edge+peak[0]+right_idx+rl[0]], 
                   bin_centers[left_edge+peak[0]+right_idx+rl[1]]]
        right_y = [hist[left_edge+peak[0]+right_idx+rl[0]], 
                   hist[left_edge+peak[0]+right_idx+rl[1]]]

        par_left = np.polyfit(left_x, left_y, deg=1)
        f_left = np.poly1d(par_left)
        par_right = np.polyfit(right_x, right_y, deg=1)
        f_right = np.poly1d(par_right)
        
        xx_left = np.linspace(left_x[0], left_x[1], 100)
        xx_right = np.linspace(right_x[0], right_x[1], 100)
        yy_left = f_left(xx_left)
        yy_right = f_right(xx_right)
        plt.plot(xx_left, yy_left, 'blue', label=None)
        plt.plot(xx_right, yy_right, 'blue', label=None)
        
        left_idx = find_nearest(yy_left, HM)
        right_idx = find_nearest(yy_right, HM)
        
        
        
        plt.plot([xx_left[left_idx], xx_right[right_idx]], 
                 [HM, HM], 'g', label='FWHM')
        
        L = xx_left[left_idx]
        R = xx_right[right_idx]
        FWHM = R - L
#        noise_level = ((hist[right_edge] + hist[left_edge])/2) 
#        peak_area = (area - 
#                     noise_level
#                     * (bin_centers[right_edge] - bin_centers[left_edge])
#                     )
        
        
        plt.text(0, 1, 'Area: ' + str(int(peak_area)) + ' [counts]' + '\nFWHM: ' + str(round(FWHM,3)) + '  [meV]', ha='center', va='center', 
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#        
#        plt.plot([bin_centers[left_edge], bin_centers[right_edge]], 
#                 [hist[left_edge], hist[right_edge]], 'bx', mew=3, ms=10,
#                 label='Peak edges', zorder=5)
                 
        plt.plot(bin_centers[left_edge:right_edge], hist[left_edge:right_edge],
                 'r.-', label='Peak')
        
        plt.legend(loc='upper left')
        plt.xlabel('$\Delta E$ [meV]')
        plt.ylabel('Counts')
        name = name + ', Histogram of $E_i$ - $E_f$'
        plt.title(name)
        
#        folder = get_output_path(data_set)
#        hist_path = folder + name + ', histogram.dat'
#        bins_path = folder + name + ', bins.dat'
#        np.savetxt(hist_path, hist, delimiter=",")
#        np.savetxt(bins_path, bins, delimiter=",")
            
        
        plot_path = get_plot_path(data_set) + name + '.pdf'

        
        return fig, plot_path
    
    
# =============================================================================
# 14. ToF vs d + dE
# =============================================================================
        
    
def ToF_vs_d_and_dE(fig, name, df, data_set, E_i, plot_separate):
        df = df[df.d != -1]
        df = df[df.tf > 0]
        df = df[(df.wADC > 300) & (df.gADC > 300)]
        bus_ranges = [[0,2], [3,5], [6,8]]
        
        name = ('14. Histogram of $E_i$ - $E_f$, Vanadium, ' + str(E_i) + 'meV')
        
        detectors = ['ILL', 'ESS (Natural Al)', 'ESS (Pure Al)']
            
        color_vec = ['darkorange', 'magenta', 'blue']
        
        dE_bins = 1000
        dE_range = [-E_i, E_i]
        
        ToF_bins = 200
        ToF_range = [0, 20e3]
        
        d_bins = 100
        d_range = [5.9, 6.5]
        
        tf_bins = 200
        tf_range = [0, 20e3]
        if plot_separate:
            fig.set_figheight(9)
            fig.set_figwidth(12)
            fig.suptitle(name, x=0.5, y=1.05)
            for i, bus_range in enumerate(bus_ranges):
#                # Plot ToF vs d
                title = detectors[i]
                plt.subplot(3, 3, i+1)
                bus_min = bus_range[0]
                bus_max = bus_range[1]
                df_temp = df[(df.Bus >= bus_min) & (df.Bus <= bus_max)]
                plt.hist2d(df_temp.tf * 1e6, df_temp.d,
                           bins = [ToF_bins, d_bins],
                           norm=LogNorm(), vmin=1, vmax=6e3, cmap='jet')
                plt.xlabel('$t_f$ [$\mu$s]')
                plt.ylabel('d [m]')
                plt.title(title + ', $t_f$ vs d')
                plt.colorbar()
                # Plot dE
                plt.subplot(3, 3, 3+i+1)
                plt.grid(True, which='major', zorder=0)
                plt.grid(True, which='minor', linestyle='--',zorder=0)
                plt.hist(df_temp.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                         color=color_vec[i], histtype='step', label=title, 
                         zorder=3)
                plt.hist(df.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                         color='black', histtype='step', label='All detectors', 
                         zorder=2)
                plt.legend(loc='upper left')
                plt.xlabel('$E_i$ - $E_f$ [meV]')
                plt.ylabel('Counts')
                plt.title(title)
                # Plot ToF
                plt.subplot(3, 3, 6+i+1)
                plt.grid(True, which='major', zorder=0)
                plt.grid(True, which='minor', linestyle='--',zorder=0)
                plt.hist(df_temp.ToF * 62.5e-9 * 1e6, bins=1000,
                         log=LogNorm(), 
                         color=color_vec[i], histtype='step', 
                         zorder=3)
                plt.xlabel('ToF [$\mu$s]')
                plt.ylabel('Counts')
                plt.title(title + ', Histogram of ToF')
        else:
            plt.grid(True, which='major', zorder=0)
            plt.grid(True, which='minor', linestyle='--',zorder=0)
            plt.xlabel('$E_i$ - $E_f$ [meV]')
            plt.ylabel('Counts')
            for i, bus_range in enumerate(bus_ranges):
                title = detectors[i]
                bus_min = bus_range[0]
                bus_max = bus_range[1]
                df_temp = df[(df.Bus >= bus_min) & (df.Bus <= bus_max)]
                plt.hist(df_temp.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                         color=color_vec[i], histtype='step', label=title, 
                         zorder=3)
            plt.hist(df.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                     color='black', histtype='step', label='All detectors', 
                     zorder=2)
            plt.legend(loc='upper left')
            plt.title(name)
        
        plt.tight_layout()
        plot_path = get_plot_path(data_set) + name + '.pdf'
                
                
                
        
        return fig, plot_path
    
# =============================================================================
# 15. Compare cold and thermal
# =============================================================================
        
def compare_cold_and_thermal(fig, name, data_set, E_i):
    dirname = os.path.dirname(__file__)
    clusters_folder = os.path.join(dirname, '../Clusters/')
    clu_files = os.listdir(clusters_folder)
    clu_files = [file for file in clu_files if file[-3:] == '.h5']

    
    name = ('15. Vanadium, ' + str(E_i) + 'meV\n Comparrison between cold and'
            + ' thermal')
    
    dE_bins = 400
    dE_range = [-E_i, E_i]
    
    print()
    print('**************** Choose cold data ***************')
    print('-------------------------------------------------')
    not_int = True
    not_in_range = True
    file_number = None
    while (not_int or not_in_range):
        for i, file in enumerate(clu_files):
            print(str(i+1) + '. ' + file)
    
        file_number = input('\nEnter a number between 1-' + 
                            str(len(clu_files)) + '.\n>> ')
    
        try:
            file_number = int(file_number)
            not_int = False
            not_in_range = (file_number < 1) | (file_number > len(clu_files))
        except ValueError:
            pass
    
        if not_int or not_in_range:
            print('\nThat is not a valid number.')
    
    clu_set = clu_files[int(file_number) - 1]
    clu_path_cold = clusters_folder + clu_set
    
    print()
    print('*************** Choose thermal data *************')
    print('-------------------------------------------------')
    not_int = True
    not_in_range = True
    file_number = None
    while (not_int or not_in_range):
        for i, file in enumerate(clu_files):
            print(str(i+1) + '. ' + file)
    
        file_number = input('\nEnter a number between 1-' + 
                            str(len(clu_files)) + '.\n>> ')
    
        try:
            file_number = int(file_number)
            not_int = False
            not_in_range = (file_number < 1) | (file_number > len(clu_files))
        except ValueError:
            pass
    
        if not_int or not_in_range:
            print('\nThat is not a valid number.')
    
    clu_set = clu_files[int(file_number) - 1]
    clu_path_thermal = clusters_folder + clu_set
    
    print('Loading...')
    tc = pd.read_hdf(clu_path_thermal, 'coincident_events')
    cc = pd.read_hdf(clu_path_cold, 'coincident_events')
    tc = filter_clusters(tc)
    cc = filter_clusters(cc)
    
    size_cold = cc.shape[0]
    size_thermal = tc.shape[0]
    

    plt.title(name)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--',zorder=0)
    
    plt.hist(tc.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                     color='red', histtype='step', density=True,
                     zorder=2, label='Thermal sample')
    
    plt.hist(cc.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                     color='blue', histtype='step', density=True,
                     zorder=2, label='Cold sample')
    
    plt.legend(loc='upper left')
    
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Normalized counts')
    
    plot_path = get_plot_path(data_set) + name + '.pdf'
    
    return fig, plot_path
    
# =============================================================================
# 16. Compare MG and Helium-tubes
# =============================================================================
    
def compare_MG_and_He3(fig, name, df, data_set, E_i, MG_offset, He3_offset,
                       only_pure_al, MG_l, MG_r):
    
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    
    name = ('16. Vanadium, ' + str(E_i) + 'meV\n Comparrison between He3 and'
            + ' Multi-Grid')
    
    use_peak_norm = True
    
    dirname = os.path.dirname(__file__)
    he_folder = os.path.join(dirname, '../Tables/Helium3_spectrum/')
    energies_path = he_folder + 'e_list_HR.dat'
    dE_path = he_folder + 'energy_HR.dat'
    hist_path = he_folder + 'histo_HR.dat'
    
    energies = np.loadtxt(energies_path)
    dE = np.loadtxt(dE_path, delimiter=',', ndmin=2)
    hist = np.loadtxt(hist_path, delimiter=',', ndmin=2)
    
    energy_dict = {}
    for i, energy in enumerate(energies):
        hist_dict = {'bins': dE[:, i], 'histogram': hist[:, i]}
        energy_dict.update({energy: hist_dict})
    
    dE_bins = 390
    dE_range = [-E_i, E_i]
    df = filter_clusters(df)
    MG_label = 'Multi-Grid'
    if only_pure_al:
        df = df[(df.Bus >= 6) | (df.Bus <= 8)]
        MG_label += ' (Pure Aluminium)'
        name += ' (Pure Aluminium)'
    hist, bins, patches = plt.hist(df.dE, bins=dE_bins, range=dE_range)
    plt.clf()
    plt.title(name)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--',zorder=0)
    
    binCenters = 0.5 * (bins[1:] + bins[:-1])
    norm_MG = 1 / sum(hist)
    if use_peak_norm:
        norm_MG = calculate_peak_norm(binCenters, hist, MG_l, MG_r)
        
    plt.plot(binCenters+MG_offset, hist / norm_MG, color='crimson', zorder=3, 
             label='Multi-Grid')
    
    data = energy_dict[E_i]
    print('Number of bins: ' + str(len(data['bins'])))
    norm_he3 = 1 / sum(data['histogram'])
    if use_peak_norm:
        He3_l = find_nearest(data['bins']+He3_offset, binCenters[MG_l]+MG_offset)
        He3_r = find_nearest(data['bins']+He3_offset, binCenters[MG_r]+MG_offset)
        norm_he3 = calculate_peak_norm(data['bins']+He3_offset, 
                                       data['histogram'], He3_l, He3_r)
    
        plt.plot([binCenters[MG_l]+MG_offset, data['bins'][He3_l]+He3_offset], 
                 [hist[MG_l]/norm_MG, data['histogram'][He3_l]/norm_he3], 'b-x', 
                 label='Peak edges', zorder=20)
        plt.plot([binCenters[MG_r]+MG_offset, data['bins'][He3_r]+He3_offset], 
                 [hist[MG_r]/norm_MG, data['histogram'][He3_r]/norm_he3], 'b-x', 
                 label=None, zorder=20)
    
    plt.plot(data['bins']+He3_offset, data['histogram'] / norm_he3, color='teal',
             label='He3', zorder=2)
    
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Normalized counts')
    plt.yscale('log')
    
    plt.legend(loc='upper left')
    
    text_string = r"$\bf{" + 'MultiGrid' + "}$" + '\n'
    text_string += 'Area: ' + str(int(norm_MG)) + ' [counts]\n'
    text_string += 'FWHM: ' + '\n'
    text_string += r"$\bf{" + 'He3' + "}$" + '\n'
    text_string += 'Area: ' + str(round(norm_he3,3)) + ' [counts]\n'
    text_string += 'FWHM: '
    
    
    plt.text(0.6*E_i, 0.04, text_string, ha='center', va='center', 
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10}, fontsize=8)
    
    plot_path = get_plot_path(data_set) + name + '.pdf'
    
    return fig, plot_path    

# =============================================================================
# 18. dE - loglog-plot
# =============================================================================   

def de_loglog(fig, name, df_vec, data_set, E_i_vec):
    
    dE_bins = 500
        
    name = ('18. $C_4 H_2 I_2 S$, Histogram of $E_i$ - $E_f$' )
    
    count = 0
    end = [440, 470, 465, 475, 470, 475, 470, 475]
    multi = [4, 3, 3, 1, 1, 1, 1, 0.5]
    hists = []
    bins_vec = []
    for df, E_i in zip(df_vec, E_i_vec):
        dE_range = [-E_i, E_i]
        df = filter_clusters(df)
        weights = np.ones(df['dE'].shape[0]) * 4 ** count
        hist, bins, __ = plt.hist(df.dE, bins=dE_bins, range=dE_range, log=LogNorm(), 
                 histtype='step', weights=weights, zorder=2, label=str(E_i) + ' meV')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        hists.append(hist)
        bins_vec.append(bin_centers)
        count += 1
    
    plt.clf()
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    count = 0
    for hist, bins in zip(hists, bins_vec):
        plt.plot(bins[:end[count]], hist[:end[count]]*multi[count], zorder=2, 
                 label=str(E_i_vec[count]) + ' meV')
        count += 1

    plt.legend(loc='lower left')
    plt.xlabel('$\Delta E$ [meV]')
    plt.ylabel('Intensity [a.u.]')
    plt.xlim(0.1, max(E_i_vec))
    plt.ylim(1e3, 5e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(name)
        
    plot_path = get_plot_path(data_set) + name + str(E_i_vec) + '.pdf'
        
    return fig, plot_path


# =============================================================================
# 19. Neutrons vs Gammas scatter plot
# ============================================================================= 
    
def neutrons_vs_gammas(fig, name, df, data_set, g_l, g_r, n_l, n_r):
    df = df[df.d != -1]
    df = df[(df.wM == 1) & (df.gM <= 5)]
    bins = 1000
    ToF_range = [0, 300000]
    noise_l = 600
    noise_r = 700
    print('Calculating...')
    thres_vec = np.arange(0, 1010, 10)
    n_vec = []
    g_vec = []
    for i, thres in enumerate(thres_vec):
        df_temp = df[(df.wADC >= thres) & (df.gADC >= thres)]
        hist, __, __ = plt.hist(df_temp.ToF, bins=bins, range=ToF_range)
        background_per_bin = sum(hist[noise_l:noise_r])/(noise_r - noise_l)
        print('Background per bin: ' + str(background_per_bin))
        n_counts = sum(hist[n_l:n_r]) - background_per_bin * (n_r - n_l)
        g_counts = sum(hist[g_l:g_r]) - background_per_bin * (g_r - g_l)
        n_vec.append(n_counts)
        g_vec.append(g_counts)
        plt.clf()
        percentage_finished = str(int(round((i/len(thres_vec))*100, 1))) + '%'
        print(percentage_finished)
    print('Done!')
        
    
    fig.set_figheight(4)
    fig.set_figwidth(12)
    fig.suptitle(name, x=0.5, y=1.05)
    
    plt.subplot(1, 3, 1)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--',zorder=0)
    hist, bins, __ = plt.hist(df.ToF, bins=bins, range=ToF_range,
                              log=LogNorm(),
                              color='black', histtype='step',
                              zorder=3, label='All events')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers[g_l:g_r], hist[g_l:g_r], 'r-', label='Gamma peak', 
             zorder=5, linewidth=3.0)
    plt.plot(bin_centers[n_l:n_r], hist[n_l:n_r], 'b-', label='Neutron peak', 
             zorder=5, linewidth=3.0)
    plt.plot(bin_centers[noise_l:noise_r], hist[noise_l:noise_r], 'g-', 
             label='Background', zorder=5, linewidth=3.0)
    plt.legend()
    plt.xlabel('ToF [TDC channels]')
    plt.ylabel('Couns')
    plt.title('ToF histogram')
    
    plt.subplot(1, 3, 2)
    plt.plot(thres_vec, n_vec, 'b-',label='Neutrons', linewidth=3.0)
    plt.plot(thres_vec, g_vec, 'r-',label='Gammas', linewidth=3.0)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Threshold [ADC channels]')
    plt.ylabel('Events [Counts]')
    plt.title('Neutrons and Gammas vs ADC threshold')
    
    plt.subplot(1, 3, 3)
    cm = plt.cm.get_cmap('jet')
    sc = plt.scatter(n_vec, g_vec, c=thres_vec, vmin=thres_vec[0],
                     vmax=thres_vec[-1], cmap=cm)
    plt.colorbar(sc)
    plt.xlabel('Neutron events [Counts]')
    plt.ylabel('Gamma events [Counts]')
    plt.title('Neutrons vs Gammas scatter map')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlim(1e5, 1e7)
#    plt.ylim(1e2, 1e4)

    plt.tight_layout()
    
    plot_path = get_plot_path(data_set) + name + '.pdf'
    
    return fig, plot_path

# =============================================================================
# Rate Repetition Mode
# =============================================================================
    
def RRM_plot(df, data_sets, border, E_i_vec, measurement_time, back_yes,
             window, isPureAluminium):
    fig = plt.figure()
    df = df[df.d != -1]
    name = 'RRM\nData set(s):' + str(data_sets)
    dE_bins = 390
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ToF_bins = 1000
    
    fig.suptitle(name, x=0.5, y=1.07)
    
    df_vec = [df[df.ToF * 62.5e-9 * 1e6 <= border],
              df[df.ToF * 62.5e-9 * 1e6  > border]]
    color_vec = ['red', 'blue']
    lim_ToF_vec = [[0, border], [border, np.inf]]
    
    for i, E_i in enumerate(E_i_vec):
        plt.subplot(2, 2, i+1)
        df_temp = df_vec[i]
        calibration = 'Van__3x3_RRM_Calibration_' + str(E_i)
        t_off = get_t_off(calibration) * np.ones(df_temp.shape[0])
        T_0 = get_T0(calibration, E_i) * np.ones(df_temp.shape[0])
        frame_shift = get_frame_shift(E_i) * np.ones(df_temp.shape[0])
        E_i = E_i * np.ones(df_temp.shape[0])
        ToF = df_temp.ToF.values
        d = df_temp.d.values
        dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
        df_temp_new = pd.DataFrame(data={'dE': dE, 't_f': t_f})
        df_temp_new = df_temp_new[df_temp_new['t_f'] > 0]
    
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        
        MG_norm = 1
        tot_norm = 1

        
        MG_dE_hist, bins = np.histogram(df_temp_new.dE, bins=dE_bins,
                                        range=[-E_i[0], E_i[0]])
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        MG_back_dE_hist = plot_dE_background(E_i[0], calibration, measurement_time,
                                             MG_norm, -E_i[0], E_i[0], back_yes,
                                             tot_norm, window, isPureAluminium,
                                             lim_ToF_vec[i])

        if back_yes:
            plt.plot(bin_centers, MG_back_dE_hist, color='green',
                     label='MG background',
                     zorder=5)
        else:
            MG_dE_hist = MG_dE_hist - MG_back_dE_hist
        
        plt.plot(bin_centers, MG_dE_hist, '-', color=color_vec[i],
                 label=str(E_i[0]) + ' meV')
                
        
        
        plt.xlabel('$\Delta$E [meV]')
        plt.yscale('log')
        plt.ylabel('Counts')
        title = 'Energy transfer, $E_i$ - $E_f$'
        if back_yes is False:
            title += '\n(Background subtracted)'
            
        plt.title(title)
        plt.legend(loc='upper left')
    
    plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--',zorder=0)
    df_temp = df[df.ToF * 62.5e-9 * 1e6 <= border]
    plt.hist(df_temp.ToF * 62.5e-9 * 1e6, bins=ToF_bins, range=[0, 17e3],
             log=LogNorm(), histtype='step', color='red', zorder=2, 
             label=str(float(E_i_vec[0])) + ' meV')
    df_temp = df[df.ToF * 62.5e-9 * 1e6 > border]
    plt.hist(df_temp.ToF * 62.5e-9 * 1e6, bins=ToF_bins, range=[0, 17e3],
             log=LogNorm(), histtype='step', color='blue', zorder=2, 
             label=str(float(E_i_vec[1])) + ' meV')
    plt.xlabel('ToF [$\mu$s]')
    plt.ylabel('Counts')
    plt.title('ToF histogram')
    plt.legend()
    plt.tight_layout()
    fig.show()


# =============================================================================
# Helium3 data
# =============================================================================
    
def plot_He3_data(df, data_sets, calibration, measurement_time, E_i, calcFWHM,
                  vis_help, back_yes, window, isPureAluminium=False,
                  isRaw=False, isFiveByFive=False, useGaussianFit=False,
                  p0=None, isCLB=False, isCorrected=False):

    print('CALIBRATION: %s' % calibration)

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def Gaussian_fit(bin_centers, dE_hist, p0):
        def Gaussian(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))#+k*x+m
        
        def Linear(x, k, m):
            return k*x+m
        

        center_idx = len(bin_centers)//2
        zero_idx = find_nearest(bin_centers[center_idx-20:center_idx+20], 0) + center_idx - 20
        fit_bins = np.arange(zero_idx-40, zero_idx+40+1, 1)
        Max = max(dE_hist[fit_bins])

        plt.axvline(x=bin_centers[fit_bins[0]], color='red')
        plt.axvline(x=bin_centers[fit_bins[-1]], color='red')
        print(p0)
        popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[fit_bins],
                                            dE_hist[fit_bins],
                                            p0=p0
                                            )
        print(popt)
        sigma = abs(popt[2])
       # k = popt[3]
       # m = popt[4]
        left_idx = find_nearest(bin_centers, bin_centers[zero_idx]  - 6*sigma)
        right_idx = find_nearest(bin_centers, bin_centers[zero_idx] + 6*sigma)
        plt.axvline(x=bin_centers[left_idx], color='purple')
        plt.axvline(x=bin_centers[right_idx], color='purple')
        print('Zero_idx: ' + str(zero_idx))
        print('Valye')
        print('left idx: ' + str(left_idx))
        print('right idx: ' + str(right_idx))

        FWHM = 2*np.sqrt(2*np.log(2))*sigma

        print('FWHM: ' + str(FWHM))

        #area = (sum(dE_hist[left_idx:right_idx]) 
        #        -sum(Linear(bin_centers[left_idx:right_idx], k, m))
        #        )
        area = calculate_peak_norm(bin_centers, dE_hist, left_idx, right_idx)
        x_gaussian = bin_centers[fit_bins]
        y_gaussian = Gaussian(x_gaussian, popt[0], popt[1], popt[2]) #, popt[3], popt[4])
        #y_linear = Linear(x_gaussian, k, m)

        return area, FWHM, y_gaussian, x_gaussian, Max, popt, left_idx, right_idx
    #plt.close()
    #fig_temp = plt.figure()
    #plt.grid(True, which='major', zorder=0)
    #plt.grid(True, which='minor', linestyle='--', zorder=0)
    #E_vs_correction = import_efficiency_correction()
    #plt.plot(E_vs_correction[0], E_vs_correction[1], 'rx', label="Matt's textfile")
    #plt.xlabel('E [meV]')
    #plt.ylabel('Correction')
    #plt.title('Energy vs efficiency correction')
    #plt.xscale('log')

    #real_x = [100, 160, 225, 300, 450, 700, 1000, 1750, 3000,
    #		  4, 6, 8, 10, 14, 20, 35, 60, 90, 140, 200, 275
    #		  ]
    #real_y = [1.24, 1.37, 1.49, 1.62, 1.84, 2.13, 2.42, 3, 3.72, 
    #		  1, 1, 1, 1, 1.01, 1.02, 1.07, 1.14, 1.21, 1.33, 1.45,
    #		  1.59]

    #plt.plot(real_x, real_y, 'bx', label="Corrected_peak/Uncorrected_peak")
    #plt.legend()
    #fig_temp.show()

    if isPureAluminium:
        df = df[(df.Bus <= 8) & (df.Bus >= 6)]

    if isCLB:
        df = df[(df.Bus <= 5) & (df.Bus >= 3)]
    
    fig = plt.figure()
    MG_SNR = 0
    He3_SNR = 0
    
    # Import He3 data
    measurement_id = find_He3_measurement_id(calibration)
    dirname = os.path.dirname(__file__)
    folder = '../Archive/2018_06_06_SEQ_He3/'
    nxs_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxs'
    nxspe_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxspe'
    nxs_path = os.path.join(dirname, folder + nxs_file)
    nxspe_path = os.path.join(dirname, folder + nxspe_file)
    nxs = h5py.File(nxs_path, 'r')
    nxspe = h5py.File(nxspe_path, 'r')
    # Extract values
    he3_bins = nxs['mantid_workspace_1']['event_workspace']['axis1'].value
    he3_min = he3_bins[0]
    he3_max = he3_bins[-1]
    He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
    He3_dE_hist = None
    if isRaw:
        df_He3 = load_He3_h5(calibration)
        T_0_raw = get_T0(calibration, E_i) * np.ones(len(df_He3.ToF))
        t_off_raw = get_t_off_He3(calibration) * np.ones(len(df_He3.ToF))
        E_i_raw = E_i * np.ones(len(df_He3.ToF))
        dE_raw, t_f_raw = get_dE_He3(E_i_raw, df_He3.ToF, df_He3.distance, T_0_raw, t_off_raw)
        df_temp_raw = pd.DataFrame(data={'dE': dE_raw, 't_f': t_f_raw})
        dE_raw = df_temp_raw[df_temp_raw['t_f'] > 0].dE
        He3_dE_hist, __ = np.histogram(dE_raw, bins=390, range=[he3_min, he3_max])
        Ei_temp = get_Ei(measurement_id)
        #for i, dE in enumerate(He3_bin_centers):
        #    E_temp = -(dE - Ei_temp)
            #print(E_temp)
            #closest_idx = find_nearest(E_vs_correction[0], E_temp)
            #He3_dE_hist[i] *= E_vs_correction[1][closest_idx] ## UNCOMMENT THIS LINE OFR efficency corr

    elif isCorrected:
        corrected_path = ''
        uncorrected_path = ''
        if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
            corrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_trueHighFlux/')
            file_name = 'van_eff_trueHighFlux_' + str(int(E_i)) + 'p00.nxspe'
            corrected_path = corrected_folder + file_name
            uncorrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_falseHighFlux/')
            uncorrected_file = 'van_eff_falseHighFlux_' + str(int(E_i)) + 'p00.nxspe'
            uncorrected_path = uncorrected_folder + uncorrected_file
        else:
            corrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_true/')
            file_name = 'van_eff_true_' + str(int(E_i)) + 'p00.nxspe'
            corrected_path = corrected_folder + file_name
            uncorrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_false/')
            uncorrected_file = 'van_eff_false_' + str(int(E_i)) + 'p00.nxspe'
            uncorrected_path = uncorrected_folder + uncorrected_file
        corrected_He3 = h5py.File(corrected_path, 'r')
        uncorrected_He3 = h5py.File(uncorrected_path, 'r')
        corrected_data = corrected_He3['data']['data']['data']
        uncorrected_data = uncorrected_He3['data']['data']['data']
        he3_bins = corrected_He3['data']['data']['energy']
        he3_min = he3_bins[0]
        he3_max = he3_bins[-1]
        He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
        #print(corrected_data)
        #print(He3_bin_centers)
        He3_dE_hist = np.zeros(len(He3_bin_centers), dtype='float')
        He3_dE_hist_uncorr = np.zeros(len(He3_bin_centers), dtype='float')
        for i, row in enumerate(corrected_data):
            if np.isnan(np.array(row)).any():
                print(i)
            else:
                He3_dE_hist += np.array(row)

        for i, row in enumerate(uncorrected_data):
            if np.isnan(np.array(row)).any():
                pass
            else:
                He3_dE_hist_uncorr += np.array(row)



    else:
        dE = nxs['mantid_workspace_1']['event_workspace']['tof'].value
        He3_dE_hist, __ = np.histogram(dE, bins=390, range=[he3_min, he3_max])
    
    He_duration = nxs['mantid_workspace_1']['logs']['duration']['value'].value[0]
    
    # Calculate MG spectrum
    df = df[df.d != -1]
    t_off = np.ones(df.shape[0]) * get_t_off(calibration)
    T_0 = np.ones(df.shape[0]) * get_T0(calibration, E_i)
    frame_shift = np.ones(df.shape[0]) * get_frame_shift(E_i)
    E_i = np.ones(df.shape[0]) * E_i #get_Ei(measurement_id)
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift) 
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    # Get MG dE histogram
    dE_bins = len(He3_bin_centers)
    dE_range = [he3_min, he3_max]
    MG_dE_hist, MG_bins = np.histogram(dE, bins=dE_bins, range=dE_range)
    MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
    # Get He3 offset
    He3_offset = 0
    if isRaw is False:
        He3_offset = get_He3_offset(calibration)
    # Use Gaussian fitting procedure instead, if selected
    if useGaussianFit:
        norm_MG, MG_FWHM, MG_gaussian, MG_fit_x, MG_MAX, p0, MG_left, MG_right = Gaussian_fit(MG_bin_centers,
                                                                                      		  MG_dE_hist, p0)
        norm_He3, He3_FWHM, He3_gaussian, He3_fit_x, He3_MAX, p0, He3_left, He3_right = Gaussian_fit(He3_bin_centers+He3_offset,
                                                                                            		 He3_dE_hist, p0)
    else:
        # Get MG and He3 peak edges
        MG_left, MG_right, He3_left, He3_right = get_peak_edges(calibration)
        if isRaw:
            MG_left, MG_right, He3_left, He3_right = get_peak_edges_raw(calibration)
        # Get MG and He3 normalisation
        norm_MG = calculate_peak_norm(MG_bin_centers, MG_dE_hist, MG_left,
                                      MG_right)
        norm_He3 = calculate_peak_norm(He3_bin_centers, He3_dE_hist, He3_left,
                                       He3_right)




    # Declare solid angle for Helium-3 and Multi-Grid
  #  He3_solid_angle = 0.7498023343722737
  #  MG_solid_angle = 0
  #  MG_solid_angle_tot = 0.01660177142644554
 #   MG_missing_solid_angle_1 = 0.0013837948633069277
 #   MG_missing_solid_angle_2 = 0.0018453301781999457
 #   if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
  #      if E_i[0] < 450:
  #          MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1
  #                            - MG_missing_solid_angle_2)
  #          if isCLB:
  #              MG_solid_angle = (0.005518217498193907 - 0.00046026495055297194
  #                                - 0.0018453301781999457)

 #       else:
 #           MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1)
 #           if isCLB:
 #               MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194

 #   else:
 #       if E_i[0] > 50:
 #           MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1
 #                             - MG_missing_solid_angle_2)
 #           if isCLB:
 #               MG_solid_angle = (0.005518217498193907 - 0.00046026495055297194
 #                                 - 0.0018453301781999457)
 #       else:
 #           MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1)
 #           if isCLB:
 #               MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194
 #   
 #   if isPureAluminium:
 #       MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194
    
    	

    # Get charge normalization
 #   charge_norm = get_charge_norm(calibration)
 #   proton_solid_norm = ((MG_solid_angle/He3_solid_angle)*charge_norm)
    
    # Calculate total normalization
    area_time_norm = get_area_time_norm(calibration, E_i[0], isCLB,
                                        isPureAluminium,
                                        measurement_time, He_duration)
    proton_solid_norm = get_charge_solid_norm(calibration, E_i[0], isCLB,
                                              isPureAluminium)
    
    

    if isFiveByFive:
        tot_norm = sum(MG_dE_hist)/sum(He3_dE_hist)
    
    # Plot background level
    hist_back = plot_dE_background(E_i[0], calibration, measurement_time,
                                   area_time_norm, he3_min, he3_max, back_yes,
                                   area_time_norm, window, isCLB=isCLB,
                                   isPureAluminium=isPureAluminium,
                                   numberBins=dE_bins)

    # Plot MG and He3
    if vis_help is not True:
        MG_dE_hist = MG_dE_hist/proton_solid_norm
        He3_dE_hist = He3_dE_hist

    if back_yes:
        pass
    else:
        hist_back = plot_dE_background(E_i[0], calibration, measurement_time,
                                       proton_solid_norm, he3_min, he3_max, back_yes,
                                       proton_solid_norm, window, isCLB=isCLB,
                                       isPureAluminium=isPureAluminium,
                                       numberBins=dE_bins)

        MG_dE_hist = MG_dE_hist - hist_back
        
    if isRaw:
        plt.plot(MG_bin_centers, MG_dE_hist, label='Multi-Grid', color='red')
        plt.plot(He3_bin_centers, He3_dE_hist, label='$^3$He-tubes', color='blue')
        if back_yes:
            pass
            #plt.plot(MG_bin_centers, hist_back, color='green', label='Background estimation', zorder=5)
    else:
        plt.plot(MG_bin_centers, MG_dE_hist, label='Multi-Grid', color='red', zorder=10)
        plt.plot(He3_bin_centers+He3_offset, He3_dE_hist, label='$^3$He tubes', color='blue')
        plt.plot(He3_bin_centers+He3_offset, He3_dE_hist_uncorr, label='$^3$He tubes (uncorrected)', color='orange')
        if back_yes:
            pass
        	#plt.plot(MG_bin_centers, hist_back, color='green', label='MG background', zorder=5)
    
    # Calculate FWHM
    if calcFWHM:
        if useGaussianFit:
            MG_FWHM = str(round(MG_FWHM, 4))
            He3_FWHM = str(round(He3_FWHM, 4))
        else:
            MG_FWHM, MG_SNR, MG_MAX = get_FWHM(MG_bin_centers, MG_dE_hist, MG_left, MG_right,
                                               vis_help, b_label='Background')
            MG_FWHM = str(round(MG_FWHM, 4))
            He3_FWHM, He3_SNR, He3_MAX = get_FWHM(He3_bin_centers, He3_dE_hist, He3_left,
                                                  He3_right, vis_help, b_label=None)
            He3_FWHM = str(round(He3_FWHM, 4))

    MG_peak_normalised = norm_MG/proton_solid_norm
    He3_peak_normalised = norm_He3
    MG_over_He3 = round(MG_peak_normalised/He3_peak_normalised, 4)
    MG_over_He3_max = 0
    if calcFWHM:
        MG_over_He3_max = round((MG_MAX/He3_MAX), 4)
    # Plot text box
    text_string = r"$\bf{" + '---MultiGrid---' + "}$" + '\n'
    text_string += 'Area: ' + str(round(MG_peak_normalised, 1)) + ' [counts]\n'
    text_string += 'FWHM: ' + MG_FWHM + ' [meV]\n'
    text_string += 'Duration: ' + str(round(measurement_time, 1)) + ' [s]\n'
    text_string += r"$\bf{" + '---He3---' + "}$" + '\n'
    text_string += 'Area: ' + str(round(He3_peak_normalised, 1)) + ' [counts]\n'
    text_string += 'FWHM: ' + He3_FWHM + ' [meV]\n'
    text_string += 'Duration: ' + str(round(He_duration, 1)) + '  [s]\n'
    text_string += r"$\bf{" + '---Comparison---' + "}$" + '\n'
    text_string += 'Area fraction: ' + str(MG_over_He3)

    #nearest_temp = find_nearest(He3_bin_centers+He3_offset, 0)
    #zero_val_He3_corr   = He3_dE_hist[nearest_temp]
    #zero_val_He3_uncorr = He3_dE_hist_uncorr[nearest_temp]

    #plt.plot([He3_bin_centers[nearest_temp], He3_bin_centers[nearest_temp]],
    #		 [zero_val_He3_corr, zero_val_He3_uncorr])

    #text_string += '\nEfficiency correction: %f' % round(zero_val_He3_corr/zero_val_He3_uncorr, 4)
    
    He3_hist_max = max(He3_dE_hist)
    MG_hist_max = max(MG_dE_hist)
    tot_max = max([He3_hist_max, MG_hist_max])
    plt.text(-0.7*E_i[0], tot_max * 0.07, text_string, ha='center', va='center', 
                bbox={'facecolor':'white', 'alpha':0.9, 'pad':10}, fontsize=6,
                zorder=50)
    
    # Visualize peak edges
    if useGaussianFit:
        plt.plot(MG_fit_x, MG_gaussian/proton_solid_norm, color='purple')
        plt.plot(He3_fit_x, He3_gaussian, color='orange')
        
        plt.plot([He3_bin_centers[He3_left]+He3_offset,
                  He3_bin_centers[He3_right]+He3_offset],
                 [He3_dE_hist[He3_left], He3_dE_hist[He3_right]], '-x',
                  color='black' ,
                  label='Peak edges', zorder=20)
        plt.plot([MG_bin_centers[MG_left], MG_bin_centers[MG_right]], 
                 [MG_dE_hist[MG_left], MG_dE_hist[MG_right]], '-x', 
                  color='black',
                  label=None, zorder=20)
        #pass
                
    plt.legend(loc='upper right').set_zorder(10)
    plt.yscale('log')
    plt.xlabel('E$_i$ - E$_f$ [meV]')
    plt.xlim(he3_min, he3_max)
    plt.ylim(1, 1.5*He3_hist_max)
    plt.ylabel('Normalized Counts')
    title = calibration + '_meV' 
    if back_yes is not True:
        title += '\n(Background subtracted)'
    if isPureAluminium:
        title += '\nESS: Pure Aluminium'
    if isFiveByFive:
        title += '\n(Van__5x5 sample for Multi-Grid)'
    if isCLB:
        title += '\nESS: Coated Long Blades'
    plt.title(title)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)


    # Export histograms to text-files
    if back_yes == True:
        dir_name = os.path.dirname(__file__)
        MG_path = os.path.join(dir_name, '../Results/Histograms/MG/dE/MG_%s_meV.txt' % calibration)
        He3_path = os.path.join(dir_name, '../Results/Histograms/He3/dE/He3_%s_meV.txt' % calibration)
        MG_dict = {'dE [meV]': MG_bin_centers,
                   'Signal [Normalized Counts]': MG_dE_hist,
                   'Background estimation [Normalized counts]': hist_back
                    }
        He3_dict = {'dE [meV]': He3_bin_centers,
                   'Signal [Normalized Counts]': He3_dE_hist
                    }
        MG_df = pd.DataFrame(MG_dict)
        He3_df = pd.DataFrame(He3_dict)
        MG_df.to_csv(MG_path, index=None, sep=' ', mode='w', encoding='ascii')
        He3_df.to_csv(He3_path, index=None, sep=' ', mode='w', encoding='ascii')
    return fig, MG_over_He3, MG_over_He3_max, He3_FWHM, MG_FWHM, p0

        
# =============================================================================
# 23. Iterate through all energies and export energies
# =============================================================================    

def plot_all_energies(isPureAluminium, isRaw, is5by5, isCLB, isCorrected,
					  useGaussianFit, window):
    window.iter_progress.show()
    # Declare all the relevant file names
    Van_3x3_HF_clusters = ["['mvmelst_1577_15meV_HF.zip'].h5",
                           "['mvmelst_1578_20meV_HF.zip'].h5",
                           "['mvmelst_1579_25meV_HF.zip', '...'].h5",
                           "['mvmelst_1581_31p7meV_HF.zip'].h5",
                           "['mvmelst_1582_34meV_HF.zip', '...'].h5",
                           "['mvmelst_1586_40p8meV_HF.zip'].h5",
                           "['mvmelst_1587_48meV_HF.zip', '...'].h5",
                           "['mvmelst_1591_60meV_HF.zip', '...'].h5",
                           "['mvmelst_1595_70meV_HF.zip', '...'].h5",
                           "['mvmelst_1597_80meV_HF.zip'].h5",
                           "['mvmelst_1598_90meV_HF.zip', '...'].h5",
                           "['mvmelst_1600_100meV_HF.zip', '...'].h5",
                           "['mvmelst_1602_120meV_HF.zip'].h5",
                           "['mvmelst_1603_140meV_HF.zip', '...'].h5",
                           "['mvmelst_1605_160meV_HF.zip'].h5",
                           "['mvmelst_1606_180meV_HF.zip'].h5",
                           "['mvmelst_1607_200meV_HF.zip'].h5",
                           "['mvmelst_1608_225meV_HF.zip'].h5",
                           "['mvmelst_1609_250meV_HF.zip'].h5",
                           "['mvmelst_1610_275meV_HF.zip'].h5",
                           "['mvmelst_1611_300meV_HF.zip', '...'].h5",
                           "['mvmelst_1613_350meV_HF.zip'].h5",
                           "['mvmelst_1614_400meV_HF.zip', '...'].h5",
                           "['mvmelst_153.mvmelst', '...'].h5",
                           "['mvmelst_156.mvmelst'].h5",
                           "['mvmelst_157.mvmelst'].h5",
                           "['mvmelst_158.mvmelst'].h5",
                           "['mvmelst_160.mvmelst'].h5",
                           "['mvmelst_161.mvmelst'].h5",
                           "['mvmelst_162.mvmelst'].h5",
                           "['mvmelst_163.mvmelst'].h5",
                           "['mvmelst_164.mvmelst'].h5",
                           "['mvmelst_165.mvmelst'].h5",
                           "['mvmelst_166.mvmelst'].h5",
                           "['mvmelst_167.mvmelst'].h5",
                           "['mvmelst_168.mvmelst'].h5",
                           "['mvmelst_169.mvmelst'].h5"]
    
    Van_3x3_HR_clusters = ["['mvmelst_125.mvmelst', '...'].h5",
                           "['mvmelst_127.mvmelst'].h5",
                           "['mvmelst_129.mvmelst'].h5",
                           "['mvmelst_130.mvmelst'].h5",
                           "['mvmelst_131.mvmelst'].h5",
                           "['mvmelst_132.mvmelst', '...'].h5",
                           "['mvmelst_134.mvmelst', '...'].h5",
                           "['mvmelst_137.mvmelst'].h5",
                           "['mvmelst_138.mvmelst'].h5",
                           "['mvmelst_139.mvmelst'].h5",
                           "['mvmelst_140.mvmelst'].h5",
                           "['mvmelst_141.mvmelst'].h5",
                           "['mvmelst_142.mvmelst', '...'].h5",
                           "['mvmelst_145.mvmelst', '...'].h5",
                           "['mvmelst_147.mvmelst'].h5",
                           "['mvmelst_148.mvmelst'].h5",
                           "['mvmelst_149.mvmelst'].h5",
                           "['mvmelst_150.mvmelst'].h5",
                           "['mvmelst_151.mvmelst'].h5",
                           "['mvmelst_1556_60meV_HR.zip'].h5",
                           "['mvmelst_1557_70meV_HR.zip', '...'].h5",
                           "['mvmelst_1561_80meV_HR.zip'].h5",
                           "['mvmelst_1562_90meV_HR.zip'].h5",
                           "['mvmelst_1563_100meV_HR.zip'].h5",
                           "['mvmelst_1564_120meV_HR.zip', '...'].h5",
                           "['mvmelst_1566_140meV_HR.zip'].h5",
                           "['mvmelst_1567_160meV_HR.zip', '...'].h5",
                           "['mvmelst_1569_180meV_HR.zip'].h5",
                           "['mvmelst_1570_200meV_HR.zip', '...'].h5",
                           "['mvmelst_1572_225meV_HR.zip'].h5",
                           "['mvmelst_1573_250meV_HR.zip', '...'].h5",
                           "['mvmelst_1575_275meV_HR.zip'].h5",
                           "['mvmelst_1576_300meV_HR.zip'].h5"]

    Van_3x3_cluster_for_efficiency_correction = ["['mvmelst_129.mvmelst'].h5",
    											 "['mvmelst_131.mvmelst'].h5",
    											 "['mvmelst_134.mvmelst', '...'].h5",
    											 "['mvmelst_138.mvmelst'].h5",
    											 "['mvmelst_140.mvmelst'].h5",
    											 "['mvmelst_145.mvmelst', '...'].h5",
    											 "['mvmelst_149.mvmelst'].h5",
    											 "['mvmelst_1556_60meV_HR.zip'].h5",
    											 "['mvmelst_1562_90meV_HR.zip'].h5",
    											 "['mvmelst_1566_140meV_HR.zip'].h5",
    											 "['mvmelst_1570_200meV_HR.zip', '...'].h5",
    											 "['mvmelst_1575_275meV_HR.zip'].h5",
    											 "['mvmelst_1600_100meV_HF.zip', '...'].h5",
    											 "['mvmelst_1605_160meV_HF.zip'].h5",
    											 "['mvmelst_1608_225meV_HF.zip'].h5",
    											 "['mvmelst_1611_300meV_HF.zip', '...'].h5",
    											 "['mvmelst_153.mvmelst', '...'].h5",
    											 "['mvmelst_158.mvmelst'].h5",
    											 "['mvmelst_162.mvmelst'].h5",
    											 "['mvmelst_165.mvmelst'].h5",
    											 "['mvmelst_168.mvmelst'].h5"
    											 ]
    # Convert to numpy arrays
    Van_3x3_HF_clusters = np.array(Van_3x3_HF_clusters)
    Van_3x3_HR_clusters = np.array(Van_3x3_HR_clusters)
    # Declare parameters
    calcFWHM = True
    vis_help = False
    back_yes_vec = [True, False]
    # Declare input-folder
    dir_name = os.path.dirname(__file__)
    clusters_folder = os.path.join(dir_name, '../Clusters/MG/')
    data_set_names = ['V_3x3_HF', 'V_3x3_HR']

    HF_folder = os.path.join(dir_name, '../Clusters/MG_new/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder) if file[-3:] == '.h5'])
    HF_files_sorted = sorted(HF_files, key=lambda element: float(element[element.find('Calibration_')+len('Calibration_'):element.find('_meV')]))
    Van_3x3_HF_clusters = np.core.defchararray.add(np.array(len(HF_files)*[HF_folder]), HF_files_sorted)
    HR_folder = os.path.join(dir_name, '../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder) if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: float(element[element.find('Calibration_')+len('Calibration_'):element.find('_meV')]))
    Van_3x3_HR_clusters = np.core.defchararray.add(np.array(len(HR_files)*[HR_folder]), HR_files_sorted)

    m_data_sets = [Van_3x3_HF_clusters, Van_3x3_HR_clusters]
    #if True:
    #	HR_choices =  [2, 4, 6, 8, 10, 13, 16, 19, 22, 25, 28, 31]
    #	HF_choices 	= [11, 14, 17, 20, 23, 26, 29, 32, 35]
    #	m_data_sets = [Van_3x3_HF_clusters[HF_choices],
    #				   ]
    choices = [1, 2]

    for choice in choices:
        p0 = [1.20901528e+04, 5.50978749e-02, 1.59896619e+00] #, -2.35758418, 9.43166002e+01]
        m_type = data_set_names[choice - 1]
        clusters = m_data_sets[choice - 1]
        print(clusters)
        
        # Declare output-folder
        with_background_result_folder = os.path.join(dir_name,
                                                     '../Results/' + m_type + '/With_Background/')
        without_background_result_folder = os.path.join(dir_name,
                                                    '../Results/' + m_type + '/Without_Background/')
        folder_vec = [with_background_result_folder, 
                      without_background_result_folder]
        # Declare overview folder
        overview_folder = os.path.join(dir_name, '../Results/' + m_type + '_overview/')
        E_i_vec = []
        max_frac_vec = []
        area_frac_vec = []
        He3_FWHM_vec = []
        MG_FWHM_vec = []
        for i, cluster_name in enumerate(clusters):
            # Import clusters
            #clusters_path = clusters_folder + cluster_name
            clusters_path = cluster_name
            df = pd.read_hdf(clusters_path, 'coincident_events')
            df = filter_ce_clusters(window, df)
            E_i = pd.read_hdf(clusters_path, 'E_i')['E_i'].iloc[0]
            measurement_time = pd.read_hdf(clusters_path, 'measurement_time')['measurement_time'].iloc[0]
            calibration = pd.read_hdf(clusters_path, 'calibration')['calibration'].iloc[0]
            data_sets = pd.read_hdf(clusters_path, 'data_set')['data_set'].iloc[0]
            # Plot clusters
            for back_yes, folder in zip(back_yes_vec, folder_vec):
                fig = plt.figure()
                fig, a_frac, max_frac, He3_FWHM, MG_FWHM, p0 = plot_He3_data(df, data_sets, calibration,
                    	                                                     measurement_time, E_i, calcFWHM,
                        	                                                 vis_help, 
                            	                                             back_yes, window,
                                	                                         isPureAluminium,
                                    	                                     isRaw,
                                        	                                 is5by5,
                                            	                             useGaussianFit,
                                                	                         p0,
                                                    	                     isCLB,
                                                    	                     isCorrected
                                                    	                     )

                print('A frac: ' + str(a_frac))
                print('MG_FWHM: ' + str(MG_FWHM))
                path = folder + calibration
                if back_yes is not True:
                    path += '_Background_subtracted'
                    E_i_vec.append(E_i)
                    max_frac_vec.append(max_frac)
                    area_frac_vec.append(a_frac)
                    He3_FWHM_vec.append(float(He3_FWHM))
                    MG_FWHM_vec.append(float(MG_FWHM))
                path += '.pdf'
                fig.savefig(path, bbox_inches='tight')
                plt.close('all')
                    
            percentage_finished = (choice-1)*50 + int(round((i/len(clusters))*100, 1))//2
            window.iter_progress.setValue(percentage_finished)
            window.update()
            window.app.processEvents()

        np.savetxt(overview_folder + 'E_i_vec.txt', E_i_vec, delimiter=",")
        np.savetxt(overview_folder + 'max_frac_vec.txt', max_frac_vec, delimiter=",")
        np.savetxt(overview_folder + 'area_frac_vec.txt', area_frac_vec, delimiter=",")
        np.savetxt(overview_folder + 'He3_FWHM_vec.txt', He3_FWHM_vec, delimiter=",")
        np.savetxt(overview_folder + 'MG_FWHM_vec.txt', MG_FWHM_vec, delimiter=",")
    
    window.iter_progress.close()
                        
# =============================================================================
# Plot overview of energies
# =============================================================================
    
def plot_FWHM_overview():
    fig = plt.figure()
    dir_name = os.path.dirname(__file__)
    data_set_names = ['V_3x3_HR', 'V_3x3_HF']
    color_vec = ['blue', 'red']
    for i, data_set_name in enumerate(data_set_names):
        overview_folder = os.path.join(dir_name, '../Results/' + data_set_name + '_overview/')
        E_i = np.loadtxt(overview_folder + 'E_i_vec.txt', delimiter=",")
        max_frac = np.loadtxt(overview_folder + 'max_frac_vec.txt', delimiter=",")
        area_frac = np.loadtxt(overview_folder + 'area_frac_vec.txt', delimiter=",")
        He3_FWHM = np.loadtxt(overview_folder + 'He3_FWHM_vec.txt', delimiter=",")
        MG_FWHM = np.loadtxt(overview_folder + 'MG_FWHM_vec.txt', delimiter=",")
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0) 
        plt.title('FWHM vs Energy')
        plt.plot(E_i, MG_FWHM, '-x', label=data_set_name + ', MG', zorder=5)
        plt.plot(E_i, He3_FWHM, '-x', label=data_set_name + ', He3', zorder=5)
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('FWHM [meV]')
        plt.legend()
    
    plt.tight_layout()
    fig.show()


def plot_Efficency_overview():
    def energy_correction(energy):
        A=0.99827
        b = 1.8199
        return A/(1-np.exp(-b*meV_to_A(energy)))
    
    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

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

    eff_theo = import_efficiency_theoretical()
    eff_corr = import_efficiency_correction()
    shift = 1/1.43
    #HF_corr = np.array([1.24, 1.37, 1.49, 1.62, 1.84, 2.13, 2.42, 3, 3.72])
    #HF_energies = np.array([99.796, 158.263, 221.080, 293.690, 438.116, 677.286, 959.894, 1655.367, 2786.407])
    #HR_corr = np.array([1, 1, 1, 1, 1.01, 1.02, 1.07, 1.14, 1.21, 1.33, 1.45, 1.59])
    #HR_energies = np.array([4.018, 5.633, 8.048, 10.05, 19.902, 20.118, 35.23, 60.413, 90.444, 139.795, 198.52, 271.986])
    E_i_vec = [HR_energies, HF_energies]
    #corr_vec = [HR_corr, HF_corr]
    fig = plt.figure()
    dir_name = os.path.dirname(__file__)
    data_set_names = ['V_3x3_HR', 'V_3x3_HF']
    color_vec = ['blue', 'red']
    plt.plot(eff_theo[0], eff_theo[1], color='black', label='Theoretical', zorder=3)
    for i, data_set_name in enumerate(data_set_names):
        overview_folder = os.path.join(dir_name, '../Results/' + data_set_name + '_overview/')
        E_i = np.loadtxt(overview_folder + 'E_i_vec.txt', delimiter=",")
        max_frac = np.loadtxt(overview_folder + 'max_frac_vec.txt', delimiter=",")
        area_frac = np.loadtxt(overview_folder + 'area_frac_vec.txt', delimiter=",") / energy_correction(E_i_vec[i]) * shift
        He3_FWHM = np.loadtxt(overview_folder + 'He3_FWHM_vec.txt', delimiter=",")
        MG_FWHM = np.loadtxt(overview_folder + 'MG_FWHM_vec.txt', delimiter=",")
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0) 
        plt.title('Efficiency vs Energy\n(Peak area comparrison, MG/He3, including efficiency correction for He3)')
        plt.plot(E_i_vec[i], area_frac, '-x', color=color_vec[i], label=data_set_name + ' (Scaled by %.2f)' % shift, zorder=5)
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        plt.ylabel('Efficiency')
        plt.legend()


    # Export efficiency correction data to text-files
    #path1 = os.path.join(dir_name, '../Results/correction_vs_energy1.txt')
    #path2 = os.path.join(dir_name, '../Results/correction_vs_energy2.txt')
    #df1_dict = {'meV': np.concatenate((HR_energies, HF_energies), axis=None), 
    #			'A': meV_to_A(np.concatenate((HR_energies, HF_energies), axis=None)),
    #			'Correction': np.concatenate((HR_corr, HF_corr), axis=None)
    #			}
    #df2_dict = {'meV': eff_corr[0], 'A': eff_corr[1], 'Correction': eff_corr[2]}
    #df1 = pd.DataFrame(df1_dict)
    #df2 = pd.DataFrame(df2_dict)
    #df1.to_csv(path1, index=None, sep=' ', mode='w', encoding='ascii')
    #df2.to_csv(path2, index=None, sep=' ', mode='w', encoding='ascii')



    plt.tight_layout()
    fig.show()

# =============================================================================
# 25. Plot raw He3 data
# =============================================================================
    
def plot_Raw_He3(fig, E_i, calibration):
    # Declare parameters
    m_id = str(find_He3_measurement_id(calibration))
    # Import raw He3 data
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name,'../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
    He3_file = h5py.File(path, 'r')
    ToF_tot = []
    pixels_tot = []
    for bank_value in range(40, 151):
        bank = 'bank' + str(bank_value) + '_events'
        ToF = He3_file['entry'][bank]['event_time_offset'].value
        pixels = He3_file['entry'][bank]['event_id'].value
        if ToF != []:
            ToF_tot.extend(ToF)
            pixels_tot.extend(pixels)
    pixels_tot = np.array(pixels_tot)
    distance = np.zeros([len(pixels_tot)], dtype=float)
    __, __, __, d = import_He3_coordinates_NEW()
    for i, pixel in enumerate(pixels_tot):
        distance[i] = d[pixel-39936]
    
    T_0 = get_T0(calibration, E_i) * np.ones(len(ToF_tot))
    t_off = get_t_off_He3(calibration) * np.ones(len(ToF_tot))
    E_i = E_i * np.ones(len(ToF_tot))
    dE, t_f = get_dE_He3(E_i, np.array(ToF_tot), distance, T_0, t_off)
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    plt.hist(dE, bins=390, range=[-E_i[0], E_i[0]], histtype='step',
             color='black', zorder=5)
    plt.xlabel('$\Delta$E')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Raw Helium data\n' + calibration)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    
    path = os.path.join(dir_name,'../RAW_HELIUM_TEST.pdf')
    
    return fig, path

# =============================================================================
# 26. Plot 3D plotly histogram
# ============================================================================= 
    
def Coincidences_3D_for_animation(df_tot, df, path, data_set, min_tof=0, 
                                  max_tof=9100, min_val=1, max_val=50):
    # Declare max and min count
    min_count = 2
    max_count = np.inf
    # Calculate vertical line position
    v_pos_x = (min_tof + max_tof)/2
    # Perform initial filters
    df = df[df.d != -1]
    df = df[(df.wADC > 500) & (df.gADC > 400)]
    df = df[(df.wM == 1) & (df.gM <= 5)]
    # Declare offsets
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
    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2])/(ILL_C[6][0]-ILL_C[5][0]))
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2])/(ESS_1_C[4][0]-ESS_1_C[3][0]))
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2])/(ESS_2_C[2][0]-ESS_2_C[1][0]))
    # Initiate detector mappings
    detector_1 = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    detector_2 = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    detector_3 = create_ess_channel_to_coordinate_map(theta_3, offset_3)
    detector_vec = [detector_1, detector_2, detector_3]
    # Initiate border lines   
    pairs = [[[80, 0], [80, 60]],
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
    
    b_traces = []
    for bus in range(3, 9):
        detector = detector_vec[bus//3]
        for pair in pairs:
            x_vec = []
            y_vec = []
            z_vec = []
            for loc in pair:
                gCh = loc[0]
                wCh = loc[1]
                coord = detector[bus%3, gCh, wCh]
                x_vec.append(coord['x'])
                y_vec.append(coord['y'])
                z_vec.append(coord['z'])
                
      
            b_trace = go.Scatter3d(x=z_vec,
                                   y=x_vec,
                                   z=y_vec,
                                   mode='lines',
                                   line = dict(
                                                color='rgba(0, 0, 0, 0.5)',
                                                width=5)
                                    )
            b_traces.append(b_trace)
       
    detector = detector_vec[0]
    pairs_2 = [[[80, 0, 0], [80, 60, 2]],
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
    for pair in pairs_2:
        x_vec = []
        y_vec = []
        z_vec = []
        for loc in pair:
            gCh = loc[0]
            wCh = loc[1]
            bus = loc[2]
            coord = detector[bus%3, gCh, wCh]
            x_vec.append(coord['x'])
            y_vec.append(coord['y'])
            z_vec.append(coord['z'])
                
      
        b_trace = go.Scatter3d(x=z_vec,
                               y=x_vec,
                               z=y_vec,
                               mode='lines',
                               line = dict(
                                       color='rgba(0, 0, 0, 0.5)',
                                       width=5)
                                           )
        b_traces.append(b_trace)
        
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wCh','gCh', 'Bus']].values, 
                              bins=(80, 40, 9), 
                              range=((0, 80), (80, 120), (0,9))
                             )
    # Insert results into an array
    flip_bus = {0: 2, 1: 1, 2: 0}
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
            
    hist = [[], [], [], []]
    loc = 0
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 9):
                detector = detector_vec[bus//3]
                if H[wCh, gCh-80, bus] > min_count and H[wCh, gCh-80, bus] <= max_count:
                    coord = detector[flip_bus[bus%3], gCh, flip_wire(wCh)]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    loc = loc + 1
#    for coord in [[-3.5, -1.6, 5.1, 1], [-3.5, -1.6, 5.1, 100]]:
#        for index in range(4):
#            hist[index].append(coord[index])
                    
    
    max_val_log = np.log10(max_val)
    min_val_log = np.log10(min_val)
    
    
    for lim_value in [min_val_log, max_val_log]:
        print(lim_value)
        hist[2].append(5.35)
        hist[0].append(-0.9)
        hist[1].append(-3.07)
        hist[3].append(lim_value)
    
    for i in range(4):
        hist[i] = np.array(hist[i])
    
    
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[2],
                               y=hist[0],
                               z=hist[1],
                               mode='markers',
                               marker=dict(
                                       size=5,
                                       color = np.log10(hist[3]),
                                       colorscale = 'Jet',
                                       opacity=1,
                                       colorbar=dict(thickness=20,
                                                     title = 'log<sub>10</sub>(counts)',
                                                     tickmode = 'array',
                                                     tickvals = [min_val_log,
                                                                 (max_val_log - min_val_log)/2,
                                                                 max_val_log],
                                                     ticktext = [round(min_val_log, 2),
                                                                 round((max_val_log - min_val_log)/2, 2),
                                                                 round(max_val_log, 2)],
                                                     ticks = 'outside'
                                                     ),
                              cmin=min_val_log,
                              cmax=max_val_log,
                              ),

                               name='Multi-Grid',
                               scene='scene1'
                               )
                        
    color_lim_trace = go.Scatter3d(x=[5.35],
                                   y=[-0.9],
                                   z=[-3.07],
                                   mode='markers',
                                   marker=dict(
                                           size=5,
                                           color = 'rgb(255,255,255)',
                                           opacity=1,
                                           line = dict(
                                                   color = 'rgb(255,255,255)',
                                                   width = 1
                                                   )
                                        ),
                                    )
                                       
    
                                     
    # Produce histogram
    ToF_hist, ToF_bins = np.histogram(df_tot.ToF * 62.5e-9 * 1e6, bins=1000,
                                      range=[0, 16667])

    max_val_ToF_hist = max(ToF_hist)
    ToF_bin_centers = 0.5 * (ToF_bins[1:] + ToF_bins[:-1])
    ToF_trace = go.Scatter(
                           x = ToF_bin_centers,
                           y = ToF_hist,
                           #fill = 'tozerox',
                           marker = dict(
                                         color = 'rgb(0, 0, 0)'
                                        ),
                           fillcolor = 'rgba(0, 0, 255, .5)',
                           #opacity = 0.5,
                           line = dict(
                                       width = 2
                                       )
                           )
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=2, 
                             specs=[[{'is_3d': False}, 
                                     {'is_3d': True}]]
                                 )
    
    fig.append_trace(ToF_trace, 1, 1)
    fig.append_trace(MG_3D_trace, 1, 2)
    fig.append_trace(color_lim_trace, 1, 2)
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 2)
  
    a = 0.92
    camera = dict(
                 up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0),
                 eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                 )
    fig['layout']['scene1']['xaxis'].update(title='z [m]') # range=[5.28, 5.66]
    fig['layout']['scene1']['yaxis'].update(title='x [m]') # range=[-1.64, -0.6]
    fig['layout']['scene1']['zaxis'].update(title='y [m]') # range=[-3.13, -2.2]
    fig['layout']['scene1']['camera'].update(camera)
    fig['layout']['xaxis1'].update(title='ToF [µs]', showgrid=True,
                                   range=[0, 16000])
    fig['layout']['yaxis1'].update(title='Counts', range=[0.1, np.log10(max_val_ToF_hist)],
                                   showgrid=True, type='log')
    fig['layout'].update(title=data_set, height=600, width=1300)
    fig.layout.showlegend = False
    shapes = [
            {'type': 'line', 'x0': v_pos_x, 'y0': -1000, 
             'x1': v_pos_x, 'y1': 20000000,
             'line':  {'color': 'rgb(500, 0, 0)', 'width': 5}, 'opacity': 0.7}
            ]
    fig['layout'].update(shapes=shapes)
    if path == 'hej':
        py.offline.plot(fig, filename='../Ce3Dhisto.html', auto_open=True)
    else:
        pio.write_image(fig, path)

                                
# =============================================================================
# ToF-sweep
# ============================================================================= 


def ToF_sweep_animation(coincident_events, data_sets, start, stop, step, window):
    window.tof_sweep_progress.show()
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../temp_ToF_sweep_folder/')
    results_folder = os.path.join(dir_name, '../Results/')
    tof_vec = range(start, stop, step)
    ce = coincident_events
    mkdir_p(temp_folder)
    for i in range(0, (stop-start)//step-1):
        progress = round((i/(((stop-start)//step)-2))*100)
        window.tof_sweep_progress.setValue(progress)
        window.update()
        window.app.processEvents()
        min_tof = tof_vec[i]
        max_tof = tof_vec[i+1]
        ce_temp = ce[  (ce['ToF'] * 62.5e-9 * 1e6 >= min_tof)
                     & (ce['ToF'] * 62.5e-9 * 1e6 <= max_tof)]

        path = temp_folder + str(i) + '.png'
        Coincidences_3D_for_animation(ce, ce_temp, path, data_sets,
                                      min_tof, max_tof,
                                      float(window.cmin.text()),
                                      float(window.cmax.text()))
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = results_folder + '/Animations/' + str(data_sets) + '_ToF_sweep.gif'
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.tof_sweep_progress.close()
    webbrowser.open(output_path)


# =============================================================================
# Wire-sweep
# ============================================================================= 

def dE_Ce3D_SNR(df, data_sets, min_val, max_val, window, max_val_dE_hist, path,
                ribbons):
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Perform initial filters
    df = df[df.d != -1]
    # Declare offsets
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
    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2])/(ILL_C[6][0]-ILL_C[5][0]))
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2])/(ESS_1_C[4][0]-ESS_1_C[3][0]))
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2])/(ESS_2_C[2][0]-ESS_2_C[1][0]))
    # Initiate detector mappings
    detector_1 = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    detector_2 = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    detector_3 = create_ess_channel_to_coordinate_map(theta_3, offset_3)
    detector_vec = [detector_1, detector_2, detector_3]
    # Initiate border lines   
    pairs = [[[80, 0], [80, 60]],
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
    
    b_traces = []
    for bus in range(3, 9):
        detector = detector_vec[bus//3]
        for pair in pairs:
            x_vec = []
            y_vec = []
            z_vec = []
            for loc in pair:
                gCh = loc[0]
                wCh = loc[1]
                coord = detector[bus%3, gCh, wCh]
                x_vec.append(coord['x'])
                y_vec.append(coord['y'])
                z_vec.append(coord['z'])
                
      
            b_trace = go.Scatter3d(x=z_vec,
                                   y=x_vec,
                                   z=y_vec,
                                   mode='lines',
                                   line = dict(
                                                color='rgba(0, 0, 0, 0.5)',
                                                width=5)
                                    )
            b_traces.append(b_trace)
       
    detector = detector_vec[0]
    pairs_2 = [[[80, 0, 0], [80, 60, 2]],
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
    for pair in pairs_2:
        x_vec = []
        y_vec = []
        z_vec = []
        for loc in pair:
            gCh = loc[0]
            wCh = loc[1]
            bus = loc[2]
            coord = detector[bus%3, gCh, wCh]
            x_vec.append(coord['x'])
            y_vec.append(coord['y'])
            z_vec.append(coord['z'])
                
      
        b_trace = go.Scatter3d(x=z_vec,
                               y=x_vec,
                               z=y_vec,
                               mode='lines',
                               line = dict(
                                       color='rgba(0, 0, 0, 0.5)',
                                       width=5)
                                           )
        b_traces.append(b_trace)
        
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wCh', 'gCh', 'Bus']].values,
                              bins=(80, 40, 9),
                              range=((0, 80), (80, 120), (0, 9))
                              )
    # Insert results into an array
    flip_bus = {0: 2, 1: 1, 2: 0}
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
            
    hist = [[], [], [], []]
    loc = 0
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 9):
                detector = detector_vec[bus//3]
                if H[wCh, gCh-80, bus] > min_count and H[wCh, gCh-80, bus] <= max_count:
                    coord = detector[flip_bus[bus%3], gCh, flip_wire(wCh)]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    loc = loc + 1
#    for coord in [[-3.5, -1.6, 5.1, 1], [-3.5, -1.6, 5.1, 100]]:
#        for index in range(4):
#            hist[index].append(coord[index])
                        
    max_val_log = np.log10(max_val)
    min_val_log = np.log10(min_val)
    
    for lim_value in [min_val_log, max_val_log]:
        print(lim_value)
        hist[2].append(5.35)
        hist[0].append(-0.9)
        hist[1].append(-3.07)
        hist[3].append(lim_value)
    
    for i in range(4):
        hist[i] = np.array(hist[i])
    
    
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[2],
                               y=hist[0],
                               z=hist[1],
                               mode='markers',
                               marker=dict(
                                       size=5,
                                       color = np.log10(hist[3]),
                                       colorscale = 'Jet',
                                       opacity=1,
                                       colorbar=dict(thickness=20,
                                                     title = 'Intensity [a.u.]',
                                                     tickmode = 'array',
                                                     tickvals = [min_val_log, 
                                                                 (max_val_log - min_val_log)/2, 
                                                                 max_val_log],
                                                     ticktext = ['Low','Medium','High'],
                                                     ticks = 'outside'
                                                     ),
                                       cmin = min_val_log,
                                       cmax = max_val_log,
                                       ),
                              name='Multi-Grid',
                              scene='scene1'
                              )
                        
    color_lim_trace = go.Scatter3d(x=[5.35],
                                   y=[-0.9],
                                   z=[-3.07],
                                   mode='markers',
                                   marker=dict(
                                           size=5,
                                           color = 'rgb(255,255,255)',
                                           opacity=1,
                                           line = dict(
                                                   color = 'rgb(255,255,255)',
                                                   width = 1
                                                   )
                                        ),
                                    )
                                       
    
                                     
    # Produce dE histogram
    __, dE_back, dE_MG, dE_bins = dE_plot(df, window.data_sets, window.E_i,
                                          window.get_calibration(),
                                          window.measurement_time,
                                          window.back_yes.isChecked(), window)

    # Normalize the histogram
    dE_MG = dE_MG / sum(dE_MG)
    colors = np.ones(len(dE_MG)) * (len(ribbons) % 20 + 1)

#    fill_color = 'rgb(' + str(50 * (len(ribbons) % 20)) + ', 0,' + str(abs(50 - 50 * (len(ribbons) % 20))) + ')'
    dE_MG_trace = go.Scatter(
                             x = dE_bins,
                             y = dE_MG,
                             #fill = 'tozerox',
                             marker = dict(
                                          # colorscale = 'Jet',
                                           color = colors,
                                           cmin = 0,
                                           cmax = 20
                                          ),
                             line = dict(
                                         width = 2
                                        )
                             )
    ribbons.append(dE_MG_trace)


    

    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=2,
                                 specs=[[{'is_3d': False}, {'is_3d': True}]]
                                 )
    # Ribbon plot
#    for i in range(0, len(ribbons[0])):
#        fig.append_trace(dict(
#                              z=np.log10(ribbons[1][i]),
#                              x=ribbons[2][i],
#                              y=ribbons[0][i],
                             # color=np.log10(ribbons[1][i]),
#                              colorscale='Jet',
#                              showscale=False,
#                              type='surface'
#                          ), 1, 2)
    for dE_trace in ribbons[((len(ribbons)-1)//20)*20:]:
        fig.append_trace(dE_trace, 1, 1)

    fig.append_trace(MG_3D_trace, 1, 2)
    fig.append_trace(color_lim_trace, 1, 2)
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 2)
  
    a = 0.92
    b = 1.0
    camera = dict(
                 up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0),
                 eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                 )
#    camera2 = dict(
##                   up=dict(x=0, y=0, z=1),
 #                  center=dict(x=0, y=0, z=0),
 #                  eye=dict(x=2*b, y=0.5*b, z=1.3*b)
 #                  )
    fig['layout']['scene1']['xaxis'].update(title='z [m]') # range=[5.28, 5.66]
    fig['layout']['scene1']['yaxis'].update(title='x [m]') # range=[-1.64, -0.6]
    fig['layout']['scene1']['zaxis'].update(title='y [m]') # range=[-3.13, -2.2]
    fig['layout']['scene1']['camera'].update(camera)
#    fig['layout']['scene1']['xaxis'].update(title='Wire row', range=[0, 20])
#    fig['layout']['scene1']['yaxis'].update(title='dE [meV]', range=[-window.E_i, window.E_i])
#    fig['layout']['scene1']['zaxis'].update(title='Intensity', range=[0, np.log10(max(ribbons[1][0][0]))])
#    fig['layout']['scene1']['camera'].update(camera2)

    fig['layout']['xaxis1'].update(title='dE [meV]', showgrid=True,
                                   range=[-window.E_i, window.E_i])
    fig['layout']['yaxis1'].update(title='Intensity [Normalized Counts]', range=[-5, np.log10(max_val_dE_hist)],
                                   showgrid=True, type='log')
    fig['layout'].update(title=str(data_sets),
                         height=600, width=1200)
    fig.layout.showlegend = False
#    shapes = [
#            {'type': 'line', 'x0': v_pos_x, 'y0': -1000, 
#             'x1': v_pos_x, 'y1': 20000000,
#             'line':  {'color': 'rgb(500, 0, 0)', 'width': 5}, 'opacity': 0.7}
#            ]
#    fig['layout'].update(shapes=shapes)

    if path == 'hej':
        py.offline.plot(fig, filename='../Ce3Dhisto.html', auto_open=True)
    else:
        pio.write_image(fig, path)


def wires_sweep_animation(coincident_events, data_sets, window):
    window.wires_sweep_progress.show()
    window.update()
    window.app.processEvents()
    dir_name = os.path.dirname(__file__)
    wire_sweep_peak_folder = os.path.join(dir_name, '../Results/wires_sweep_data/')
    temp_folder = os.path.join(dir_name, '../temp_voxel_sweep_folder/')
    results_folder = os.path.join(dir_name, '../Results/')
    ce = coincident_events
    __, __, dE_MG, __ = dE_plot(window.filter_ce_clusters(), window.data_sets, window.E_i,
                                window.get_calibration(), window.measurement_time,
                                window.back_yes.isChecked(), window)
    max_val_dE_hist = max(dE_MG/sum(dE_MG))
    mkdir_p(temp_folder)
    count = 0
    module_ranges = [[0, 2], [3, 5], [6, 8]]
    for i in range(0, 3):
        ribbons = []
        for wire_row in range(1, 21):
            progress = round((count/(3*20)) * 100, 1)
            count += 1
            window.wires_sweep_progress.setValue(progress)
            window.update()
            window.app.processEvents()
            path = temp_folder + str(count) + '.png'
            df_temp = ce[(((ce.wCh >= wire_row - 1) &
                      (ce.wCh <= wire_row - 1)) 
                        |
                      ((ce.wCh >= wire_row + 20 - 1) &
                      (ce.wCh <= wire_row + 20 - 1))
                        |
                      ((ce.wCh >= wire_row + 40 - 1) &
                      (ce.wCh <= wire_row + 40 - 1)) 
                        |
                      ((ce.wCh >= wire_row + 60 - 1) &
                      (ce.wCh <= wire_row + 60 - 1))
                      )
                     ]
            df_temp = df_temp[(df_temp.Bus >= module_ranges[i][0]) & 
                              (df_temp.Bus <= module_ranges[i][1])
                              ]
            dE_Ce3D_SNR(df_temp, window.data_sets, float(window.cmin.text()), 
                        float(window.cmax.text()), window, max_val_dE_hist, path, ribbons)
    
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = results_folder + '/Animations/' + str(data_sets) + '_wire_sweep.gif'
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.wires_sweep_progress.close()
    window.update()
    window.app.processEvents()
    webbrowser.open(output_path)


def grids_sweep_animation(coincident_events, data_sets, window):
    window.grids_sweep_progress.show()
    window.update()
    window.app.processEvents()
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../temp_voxel_sweep_folder/')
    results_folder = os.path.join(dir_name, '../Results/')
    ce = coincident_events
    __, __, dE_MG, __ = dE_plot(window.filter_ce_clusters(), window.data_sets, window.E_i,
                                window.get_calibration(), window.measurement_time,
                                window.back_yes.isChecked(), window)
    max_val_dE_hist = max(dE_MG)
    mkdir_p(temp_folder)
    count = 0
    module_ranges = [[0, 2], [3, 5], [6, 8]]
    for i in range(0, 3):
        ribbons = [[], [], []]
        for grid in range(1, 41):
#            ribbons[2].append([np.ones(390) * (wire_row-0.5),
#                               np.ones(390) * (wire_row+0.5)])
            progress = round((count/(3*20)) * 100, 1)
            count += 1
            window.wires_sweep_progress.setValue(progress)
            window.update()
            window.app.processEvents()
            path = temp_folder + str(count) + '.png'
            df_temp = ce[(ce.gCh >= grid + 80 - 1) &
                         (ce.gCh <= grid + 80 - 1)]
            df_temp = df_temp[(df_temp.Bus >= module_ranges[i][0]) & 
                              (df_temp.Bus <= module_ranges[i][1])
                              ]
            if df_temp.shape[0] > 0:
                dE_Ce3D_SNR(df_temp, window.data_sets, float(window.cmin.text()), 
                            float(window.cmax.text()), window, max_val_dE_hist, path, ribbons)
    
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = results_folder + '/Animations/' + str(data_sets) + '_grids_sweep.gif'
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.grids_sweep_progress.close()
    window.update()
    window.app.processEvents()
    webbrowser.open(output_path)


# =============================================================================
# Depth profiles
# =============================================================================

def different_depths(window):
    module_ranges = [[0, 2], [3, 5], [6, 8]]
    labels = ['ILL', 'ESS.CLB', 'ESS.PA']
    colors = ['blue', 'red', 'green']
    ce = window.Coincident_events
    bins_and_hist = [[[], []], [[], []], [[], []]]
    for i in range(0, 3):
        for j, wire_row in enumerate([1, 20]):
            df_temp = ce[(((ce.wCh >= wire_row - 1) &
                        (ce.wCh <= wire_row - 1)) 
                        |
                        ((ce.wCh >= wire_row + 20 - 1) &
                          (ce.wCh <= wire_row + 20 - 1))
                        |
                        ((ce.wCh >= wire_row + 40 - 1) &
                        (ce.wCh <= wire_row + 40 - 1)) 
                        |
                        ((ce.wCh >= wire_row + 60 - 1) &
                        (ce.wCh <= wire_row + 60 - 1))
                        )
                         ]
            df_temp = df_temp[(df_temp.Bus >= module_ranges[i][0]) & 
                              (df_temp.Bus <= module_ranges[i][1])
                              ]

            __, __, hist_MG, bin_centers = dE_plot(df_temp, window.data_sets, window.E_i,
                                                   window.get_calibration(), window.measurement_time,
                                                   window.back_yes.isChecked(), window)

            bins_and_hist[i][0].append(bin_centers)
            bins_and_hist[i][1].append(hist_MG/sum(hist_MG))
    
    fig = plt.figure()
    fig.suptitle('Data set: ' + str(window.data_sets), x=0.5, y=1.07)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for i in range(0, 3):
        for j, wire_row in enumerate([1, 20]):
            plt.subplot(1, 3, i+1)
            plt.grid(True, which='major', linestyle='--', zorder=0)
            plt.grid(True, which='minor', linestyle='--', zorder=0)
            plt.plot(bins_and_hist[i][0][j], bins_and_hist[i][1][j],
                     color=colors[j], label='Wire row: ' + str(wire_row))
            plt.xlabel('E$_i$-E$_f$ [meV]')
            plt.ylabel('Normalized counts')
            plt.yscale('log')
            plt.title('Energy transfer\n(Detector type: ' + labels[i] + ')')
        plt.legend()

    plt.tight_layout()
    fig.show()


# =============================================================================
# Angular Dependence
# =============================================================================

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan(x/z)
    phi = np.arccos(y/r)
    return r, theta, phi


def get_detectors():
     # Declare offsets
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
    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2])/(ILL_C[6][0]-ILL_C[5][0]))
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2])/(ESS_1_C[4][0]-ESS_1_C[3][0]))
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2])/(ESS_2_C[2][0]-ESS_2_C[1][0]))
    # Initiate detector mappings
    detector_1 = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    detector_2 = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    detector_3 = create_ess_channel_to_coordinate_map(theta_3, offset_3)
    detector_vec = [detector_1, detector_2, detector_3]
    return detector_vec


def get_r_theta_phi(df):
    # Use flip-functions to get correct coordinates
    flip_bus = {0: 2, 1: 1, 2: 0}

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

    # Get all r, theta and phi
    size = df.shape[0]
    r_tot = np.zeros([size], dtype=float)
    theta_tot = np.zeros([size], dtype=float)
    phi_tot = np.zeros([size], dtype=float)
    detectors = get_detectors()
    wChs = df.wCh.values
    gChs = df.gCh.values
    buses = df.Bus.values
    for i in range(df.shape[0]):
        wCh = wChs[i]
        gCh = gChs[i]
        bus = buses[i]
        coord = detectors[bus//3][flip_bus[bus%3], gCh, flip_wire(wCh)]
        r, theta, phi = cartesian_to_spherical(coord['x'],
                                               coord['y'],
                                               coord['z']
                                               )
        r_tot[i] = r
        theta_tot[i] = theta
        phi_tot[i] = phi
    return r_tot, theta_tot*180/np.pi, phi_tot*180/np.pi

def angular_dependence_plot(df_MG, data_sets, calibration):
    df_He3 = load_He3_h5(calibration)
    r, theta, phi = get_r_theta_phi(df_MG)
    fig = plt.figure()
    plt.hist2d(df_He3.az, df_He3.pol, range=[[-180, 180], [0, 63]], bins=[300, 300],
               cmap='jet', norm=LogNorm(), vmin=1, vmax=10000)
    plt.xlabel('Azimuthal angle [degrees]')
    plt.ylabel('Polar angle [degrees]')
    plt.hist2d(theta, phi,  range=[[-180, 180], [0, 63]], bins=[300, 300],
               cmap='jet', norm=LogNorm(), vmin=1, vmax=10000)
    plt.xlabel('Azimuthal angle [degrees]')
    plt.ylabel('Polar angle [degrees]')
    plt.title('Neutron scattering angle dependence\n' + data_sets)
    plt.colorbar()
    return fig


def get_all_calibrations():
#    dirname = os.path.dirname(__file__)
#    path = os.path.join(dirname, '../Tables/He3_offset.xlsx')
#    matrix = pd.read_excel(path).values
#    calibrations = []
#    for i, row in enumerate(matrix[:74]):
#        calibrations.append(row[0])
    
    calibrations = ["Van__3x3_High_Resolution_Calibration_2.0",
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
                    "Van__3x3_High_Resolution_Calibration_300.0",
                    "Van__3x3_High_Flux_Calibration_15.0",
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
                    "Van__3x3_High_Flux_Calibration_400.0",
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
                    "Van__3x3_High_Flux_Calibration_3500.0"]
    return calibrations


def load_He3_h5(calibration):
    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../Clusters/He3/')
    path = folder + calibration + '.h5'
    return pd.read_hdf(path, calibration)


def get_all_energies(calibrations):
    energies = []
    for calibration in calibrations:
        if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
            start = len('Van__3x3_High_Flux_Calibration_')
            energy = float(calibration[start:])
        else:
            start = len('Van__3x3_High_Resolution_Calibration_')
            energy = float(calibration[start:])
        energies.append(energy)
    return energies


def cluster_all_raw_He3():
    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../Clusters/He3/')
    calibrations = get_all_calibrations()
    energies = get_all_energies(calibrations)
    x, y, z, d, az, pol = import_He3_coordinates_raw()
    count = 1
    tot = len(energies)
    for calibration, energy in zip(calibrations, energies):
        df_temp = cluster_raw_He3(calibration, x, y, z, d, az, pol)
        print(df_temp)
        path = folder + calibration + '.h5'
        df_temp.to_hdf(path, calibration, complevel=9)
        print(str(count) + '/' + str(tot))
        count += 1


def cluster_raw_He3(calibration, x, y, z, d, az, pol, path=None):
    if path is None:
        # Declare parameters
        m_id = str(find_He3_measurement_id(calibration))
        # Import raw He3 data
        dir_name = os.path.dirname(__file__)
        path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
    # Declare parameters
    banks_to_skip = np.array([74, 75, 76, 97, 98, 99, 100, 101,
                              102, 103, 114, 115, 119])
    He3_file = h5py.File(path, 'r')
    ToF_tot = []
    pixels_tot = []
    for bank_value in range(40, 150):  # FULL ARRAY IS 40->150, MG big sized is 65->70, MG small 66->69, MG polar is 70->74
        if bank_value in banks_to_skip:
            pass
        else:
            bank = 'bank' + str(bank_value) + '_events'
            ToF = He3_file['entry'][bank]['event_time_offset'].value
            pixels = He3_file['entry'][bank]['event_id'].value
            if ToF != []:
                ToF_tot.extend(ToF)
                pixels_tot.extend(pixels)
    # Declare all vectors for data storage
    pixels_tot = np.array(pixels_tot)
    distance = np.zeros([len(pixels_tot)], dtype=float)
    x_He3 = np.zeros([len(pixels_tot)], dtype=float)
    y_He3 = np.zeros([len(pixels_tot)], dtype=float)
    z_He3 = np.zeros([len(pixels_tot)], dtype=float)
    pol_He3 = np.zeros([len(pixels_tot)], dtype=float)
    az_He3 = np.zeros([len(pixels_tot)], dtype=float)
    for i, pixel in enumerate(pixels_tot):
        distance[i] = d[pixel-37888]
        x_He3[i] = x[pixel-37888]
        y_He3[i] = y[pixel-37888]
        z_He3[i] = z[pixel-37888]
        az_He3[i] = az[pixel-37888]
        pol_He3[i] = pol[pixel-37888]

    He3_clusters = {'x': x_He3, 'y': y_He3, 'z': z_He3, 'az': az_He3,
                    'pol': pol_He3, 'distance': distance, 'ToF': ToF_tot,
                    'pixel': pixels_tot}

    return pd.DataFrame(He3_clusters)


# =============================================================================
# Helium-3 Histogram
# =============================================================================

def He3_histogram_3D_plot(m_id='145280', save_path=None, data_set='',
                          path=None, window=None):

    # 652 000 000 elements, killed at 374 000 000. Perhaps take first 100 000 000 elements?
    offset = 39936 - 2 * 1024
    x, y, z, d, __, __ = import_He3_coordinates_raw()
    # Import raw He3 data
    dir_name = os.path.dirname(__file__)
    if path is None:
        path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
    He3_file = h5py.File(path, 'r')
    ToF_tot = []
    pixels_tot = []
    for bank_value in range(40, 150):
        if bank_value in [75, 76, 97, 98, 99, 100, 101, 102, 103, 114, 115, 119]:
            bank = 'bank' + str(bank_value) + '_events'
            ToF = He3_file['entry'][bank]['event_time_offset'].value
            pixels = He3_file['entry'][bank]['event_id'].value
            if pixels != []:
                ToF_tot.extend(ToF)
                pixels_tot.extend(pixels)

    #surplus3 = np.arange(77824//2, 79870//2, 1)
    counts = np.zeros([len(x)], dtype=int)
    for i, pixel in enumerate(pixels_tot):
        counts[pixel-offset] += 1
        if i % 100000 == 0:
            print(str(i) + '/' + str(len(pixels_tot)))

    indices = np.where(counts >= 0)
    labels = []
    for i, value in enumerate(counts[indices]):
        labels.append('Counts: %d, ID: %d' % (value, i))

    min_val_log = 1
    max_val_log = 3

    He3_3D_hist = go.Scatter3d(x=z[indices],
                               y=x[indices],
                               z=y[indices],
                               mode='markers',
                               marker=dict(
                                       size=5,
                                       color=np.log10(counts[indices]),
                                       colorscale='Jet',
                                       opacity=1,
                                       colorbar=dict(thickness=20,
                                                     title = 'Log10(counts)',
                                                     #tickmode = 'array',
                                                     #tickvals = [min_val_log,
                                                     #           (max_val_log-min_val_log)/2,
                                                     #            max_val_log],
                                                     #ticktext = [min_val_log,
                                                     #            (max_val_log-min_val_log)/2,
                                                     #            max_val_log],
                                                     #ticks = 'outside'
                                                     ),
                                       #cmin = min_val_log,
                                       #cmax = max_val_log
                                       ),
                               text=labels
                              )

    fig = py.tools.make_subplots(rows=1, cols=1,
                                 specs=[ 
                                    [{'is_3d': True}]
                                    ]
                                 )

    # Plot Multi-GRid
    #clusters_path = os.path.join(dir_name,
    #                             '../Clusters/MG_old/US_120_meV.h5')
    #df_MG = pd.read_hdf(clusters_path, 'coincident_events')
    #__, x_MG, y_MG, z_MG, log_counts_MG = Coincidences_3D_plot(df_MG,
    #                                                           '',
    #                                                           window)
    #trace_MG = go.Scatter3d(x=z_MG,
    #                        y=x_MG,
    #                        z=y_MG,
    #                        mode='markers',
    #                        marker=dict(
    #                               size=2,
    #                               color=log_counts_MG,
    #                               colorscale='Jet',
    #                               opacity=1,
    #                               colorbar=dict(thickness=20,
    #                                             title = 'Log10(counts)',
    #                                             #tickmode = 'array',
    #                                             #tickvals = [min_val_log,
    #                                             #           (max_val_log-min_val_log)/2,
    #                                             #            max_val_log],
    #                                             #ticktext = [min_val_log,
    #                                             #            (max_val_log-min_val_log)/2,
    #                                             #            max_val_log],
    #                                             #ticks = 'outside'
    #                                             ),
    #                               cmin = min_val_log,
    #                               cmax = max_val_log
    #                               ),
    #                        scene='scene1'
    #                        )
    #fig.append_trace(trace_MG, 1, 1)
    fig.append_trace(He3_3D_hist, 1, 1)

    a = 1
    camera = dict(
                 up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0),
                 eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                 )

    fig['layout']['scene1']['xaxis'].update(title='z [m]')
    fig['layout']['scene1']['yaxis'].update(title='x [m]') 
    fig['layout']['scene1']['zaxis'].update(title='y [m]') 
    fig['layout'].update(title='<sup>3</sup>He-tubes 3D histogram of hit location<br>Data set: ' + data_set)
    fig.layout.showlegend = False 
    fig['layout']['scene1']['camera'].update(camera)
    if save_path is not None:
        pio.write_image(fig, save_path, width=1500, height=1200)
        dir_name = os.path.dirname(__file__)
        counts_path = os.path.join(dir_name, '../Results/Animations/Counts/' + data_set + '.txt')
        np.savetxt(counts_path, counts, delimiter=",")
    else:
        py.offline.plot(fig, filename='../Results/HTML_files/He3_3D_histogram.html', auto_open=True)
        pio.write_image(fig, '../Results/HTML_files/He3Ce3Dhistogram.pdf')

    # REmove after use

    fig = plt.figure()
    hist, bins, __ = plt.hist(ToF_tot, bins=100, range=[9000, 9100])
    print('Rate')
    print(sum(hist)/((100/16667)*132))
    fig.show()


# =============================================================================
# He3 3D ToF-sweep
# =============================================================================

def He3_histogram_3D_ToF_sweep(window, path, title):
    offset = 39936 - 2 * 1024
    x, y, z, d, __, __ = import_He3_coordinates_raw()
    # Import raw He3 data
    #path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
    He3_file = h5py.File(path, 'r')
    ToF_tot = []
    pixels_tot = []
    for bank_value in range(40, 151):
        bank = 'bank' + str(bank_value) + '_events'
        ToF = He3_file['entry'][bank]['event_time_offset'].value
        pixels = He3_file['entry'][bank]['event_id'].value
        if pixels != []:
            ToF_tot.extend(ToF[0:len(ToF)])
            pixels_tot.extend(pixels[0:len(pixels)])

    temp_dict = {'ToF': ToF_tot, 'Pixels': pixels_tot}
    df = pd.DataFrame(temp_dict)
    start = int(window.tof_start.text())
    stop  = int(window.tof_stop.text())
    step  = int(window.tof_step.text())
    tof_vec = np.arange(start, stop, step)
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../Results/temp_folder_He3_ToF_sweep/')

    mkdir_p(temp_folder)


    # Import MG data for Si-sweep
    MG_file_name = 'US_120_meV.h5'
    MG_file_path = os.path.join(dir_name, '../Clusters/MG_old/' + MG_file_name)
    df_MG = pd.read_hdf(MG_file_path, 'coincident_events')

    for i in range(0, len(tof_vec)-1):
        tof_min = tof_vec[i]
        tof_max = tof_vec[i+1]
        temp_df = df[(df.ToF > tof_min) & (df.ToF < tof_max)]
        temp_df_MG = df_MG[   (df_MG.ToF*1e6*62.5e-9 > tof_min)  
                            & (df_MG.ToF*1e6*62.5e-9  < tof_max)
                           ]
        counts = np.zeros([len(x)], dtype=int)
        for j, pixel in enumerate(temp_df.Pixels):
            counts[pixel-offset] += 1
            if j % 100000 == 0:
                print(str(j) + '/' + str(len(pixels_tot)))
        save_path = temp_folder + str(i) + '.png'
        b_traces, x_MG, y_MG, z_MG, counts_MG, __ = Coincidences_3D_plot(temp_df_MG, '')

        He3_3D_hist_for_ToF(x, y, z, counts, tof_min, tof_max, save_path,
                            b_traces, x_MG, y_MG, z_MG, counts_MG, title)

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name, '../Results/Animations/He3_ToF_sweep.gif')
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)




def He3_3D_hist_for_ToF(x, y, z, counts, tof_min, tof_max, save_path,
                        b_traces, x_MG, y_MG, z_MG, counts_MG, title):
    x_MG = np.array(x_MG)
    y_MG = np.array(y_MG)
    z_MG = np.array(z_MG)
    counts_MG = np.array(counts_MG)
    x_full = np.concatenate((x, x_MG), axis=None)
    y_full = np.concatenate((y, y_MG), axis=None)
    z_full = np.concatenate((z, z_MG), axis=None)
    counts_full = np.concatenate((counts, counts_MG), axis=None)
    
    He3_3D_hist = go.Scatter3d(x=z_full,
                               y=x_full,
                               z=y_full,
                               mode='markers',
                               marker=dict(
                                       size=2,
                                       color=np.log10(counts_full),
                                       colorscale='Jet',
                                       opacity=1,
                                       colorbar=dict(thickness=20,
                                                     title = 'Log10(counts)',
                                                    ),
                                       cmin=0,
                                       cmax=1
                                       ),
                              )

    fig = py.tools.make_subplots(rows=1, cols=1,
                                 specs=[ 
                                    [{'is_3d': True}]
                                    ]
                                 )
  
    fig.append_trace(He3_3D_hist, 1, 1)
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 1)

    a = 1
    camera = dict(
                 up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0),
                 eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                 )

    fig['layout']['scene1']['xaxis'].update(title='z [m]')
    fig['layout']['scene1']['yaxis'].update(title='x [m]') 
    fig['layout']['scene1']['zaxis'].update(title='y [m]') 
    fig['layout'].update(title= title + '<br>ToF: [' + str(tof_min) + ', ' + str(tof_max) + ']')
    fig.layout.showlegend = False 
    fig['layout']['scene1']['camera'].update(camera)
    pio.write_image(fig, save_path, width=1500, height=1200)

















# =============================================================================
# Iterate through all energies and produce He3 3D histograms
# =============================================================================

def He3_histo_all_energies_animation():
    calibrations = ["Van__3x3_High_Resolution_Calibration_2.0",
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
                    "Van__3x3_High_Resolution_Calibration_300.0",
                    "Van__3x3_High_Flux_Calibration_15.0",
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
                    "Van__3x3_High_Flux_Calibration_400.0",
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
                    "Van__3x3_High_Flux_Calibration_3500.0"]

    measurements = np.arange(145160, 145233, 1)


    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../Results/temp_folder_He3_3D_hist/')
    mkdir_p(temp_folder)

    count = 0
    for m_id, cal in zip(measurements, calibrations):
        print(cal)
        path = temp_folder + str(count) + '.png'
        He3_histogram_3D_plot(str(m_id), path, cal)
        count += 1

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name, '../Results/Animations/He3_3D_sweep.gif')
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)



















# =============================================================================
# Figure-of-Merit animation (wire row sweep)
# =============================================================================

def figure_of_merit_plot(window):
    def find_FOM(bin_centers, hist_MG, start_length, end_length):
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        zero_idx = find_nearest(bin_centers, max(hist_MG))
        FOM = (  sum(hist_MG[zero_idx+start_length:zero_idx+end_length]) 
               / sum(hist_MG[zero_idx-end_length:zero_idx-start_length])
               )

        print('Zero index: ' + str(zero_idx))
        print('Bin: ' + str(bin_centers[zero_idx]))
        print('Left: ' + str(bin_centers[zero_idx-end_length:zero_idx-start_length]))
        print('Left number of elements: ' + str(len(bin_centers[zero_idx-end_length:zero_idx-start_length])))
        print('Right: ' + str(bin_centers[zero_idx+start_length:zero_idx+end_length]))
        print('Right number of elements: ' + str(len(bin_centers[zero_idx+start_length:zero_idx+end_length])))
        return FOM, zero_idx

    def find_gaussian_FoM(bin_centers, hist_MG, k, m, background, norm_MG):
        def Gaussian(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2)) + k*x + m
        zero_idx = len(bin_centers)//2
        indices = np.arange(zero_idx-zero_idx//2, zero_idx+zero_idx//2+1, 1)
        print(hist_MG[indices])
        popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[indices], hist_MG[indices],
                                            p0=[0.17608019, -0.0178427, -0.16605704])
        print(popt)
        sigma = abs(popt[2])
        x0 = popt[1]

        # Calculate FoM
        indices_shoulder = np.where((bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma))
        indices_peak = np.where((bin_centers <= x0 + sigma) & (bin_centers >= x0 - sigma))
        shoulder_area = sum(hist_MG[indices_shoulder]) - sum(Gaussian(bin_centers[indices_shoulder], popt[0], popt[1], popt[2]))
        peak_area = sum(hist_MG[indices_peak]) - sum(background[indices_peak])
        FOM = shoulder_area/peak_area

        shoulder = sum(hist_MG[indices_shoulder])
        peak = sum(hist_MG[indices_peak])
        FOM_stat_err = 0 # FOM * np.sqrt(shoulder/(shoulder**2) + peak/(peak**2))
        return (bin_centers, Gaussian(bin_centers, popt[0], popt[1], popt[2]),
                FOM, indices, popt[1], abs(popt[2]), FOM_stat_err)

    def find_linear_background(bin_centers, hist_MG):
        def Linear(x, k, m):
            return k*x + m
        zero_idx    = len(bin_centers)//2
        indices_1   = np.arange(zero_idx-zero_idx//2, zero_idx-zero_idx//3, 1)
        indices_2   = np.arange(zero_idx+zero_idx//3, zero_idx+zero_idx//2, 1)
        indices     = np.concatenate((indices_1, indices_2), axis=None)
        popt, __ = scipy.optimize.curve_fit(Linear, bin_centers[indices], hist_MG[indices])
        return popt[0], popt[1], indices, Linear(bin_centers, popt[0], popt[1])

    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../Results/temp_folder_FOM/')
    mkdir_p(temp_folder)
    module_ranges = [[0, 2], [3, 5], [6, 8]]
    colors = ['blue', 'red', 'green']
    labels = ['ILL', 'ESS.CLB', 'ESS.PA']
    FOM_and_row = [[[], [], []], [[], [], []], [[], [], []]]
    count = 0
    start_length = 5
    end_length   = 20
    ce = window.Coincident_events
    ce = filter_ce_clusters(window, ce)
    for i in range(0, 3):
        for wire_row in range(1, 21):
            progress = round((count/(3*20)) * 100, 1)
            count += 1
            window.update()
            window.app.processEvents()
            path = temp_folder + str(count) + '.png'
            df_temp = ce[(((ce.wCh >= wire_row - 1) &
                      (ce.wCh <= wire_row - 1)) 
                        |
                      ((ce.wCh >= wire_row + 20 - 1) &
                      (ce.wCh <= wire_row + 20 - 1))
                        |
                      ((ce.wCh >= wire_row + 40 - 1) &
                      (ce.wCh <= wire_row + 40 - 1)) 
                        |
                      ((ce.wCh >= wire_row + 60 - 1) &
                      (ce.wCh <= wire_row + 60 - 1))
                      )
                     ]
            df_temp = df_temp[(df_temp.Bus >= module_ranges[i][0]) & 
                              (df_temp.Bus <= module_ranges[i][1])
                              ]

            __, __, hist_MG, bin_centers = dE_plot(df_temp, window.data_sets, window.E_i,
                                                   window.get_calibration(), window.measurement_time,
                                                   window.back_yes.isChecked(), window,
                                                   numberBins=1000)
            # Normalize spectrums
            hist_MG = hist_MG/sum(hist_MG)
            norm_MG = sum(hist_MG)
            k, m, b_indices, background = find_linear_background(bin_centers, hist_MG)
            Gx, Gy, FOM, G_indices, x0, sigma, FOM_stat_err = find_gaussian_FoM(bin_centers, hist_MG,
                                                                                k, m, background, norm_MG)
            print('FOM: %f' % FOM)
            print('FOM err: %f' % FOM_stat_err)
            FOM_and_row[i][0].append(wire_row)
            FOM_and_row[i][1].append(FOM)
            FOM_and_row[i][2].append(FOM_stat_err)
            fig = plt.figure()
            fig.suptitle('Data set: ' + str(window.data_sets)) #, x=0.5, y=1.08)
            fig.set_figheight(6)
            fig.set_figwidth(12)
            plt.subplot(1, 2, 1)
            plt.grid(True, which='major', linestyle='--', zorder=0)
            plt.grid(True, which='minor', linestyle='--', zorder=0)
            plt.plot(bin_centers, hist_MG, label='Multi-Grid', color='black', zorder=3)
            plt.plot(Gx, Gy, label='Gaussian fit', color='red', zorder=3)
            plt.fill_between(bin_centers[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                             hist_MG[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                             Gy[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                             facecolor='orange',
                             label='[3$\sigma$, 5$\sigma$]',
                             alpha=0.9, zorder=2) 
            plt.fill_between(bin_centers[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                             hist_MG[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                             background[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                             facecolor='purple',
                             label='[-$\sigma$, $\sigma$]',
                             alpha=0.9, zorder=2)  
            plt.yscale('log')
            plt.xlabel('E$_i$-E$_f$ [meV]')
            plt.xlim(-20, 20)
            plt.ylabel('Normalized counts')
            plt.title('Energy transfer\n(Row: ' + str(wire_row) + ')')
            plt.legend(loc=1)
            plt.subplot(1, 2, 2)
            plt.grid(True, which='major', linestyle='--', zorder=0)
            plt.grid(True, which='minor', linestyle='--', zorder=0)
            plt.xlim(0, 20)
            plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
            #plt.ylim(0.4, 0.6)
            plt.xlabel('Wire row')
            plt.ylabel('Figure-of-Merit')
            plt.title('Figure-of-Merit vs Wire row')
            for j in range(1, i+1):
                #plt.plot(FOM_and_row[j][0], FOM_and_row[j][1], color=colors[j], label=labels[j], 
                #         linestyle='-', marker='o')
                plt.errorbar(FOM_and_row[j][0], FOM_and_row[j][1], yerr=FOM_and_row[j][2],
                             ecolor=colors[j], color=colors[j], marker='o', label=labels[j], capsize=5)
            plt.legend(loc=1)
            plt.savefig(temp_folder + str(count) + '.png')
            plt.close()

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name, '../Results/Animations/' + str(window.data_sets) + '_FOM_sweep.gif')
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.update()
    window.app.processEvents()

# =============================================================================
# Figure-of-Merit animation (energy sweep)
# =============================================================================

def figure_of_merit_energy_sweep(window):
    def find_FOM(bin_centers, hist_MG, start_length, end_length, back_hist):
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx


        zero_idx = 40 # 195 #find_nearest(bin_centers, max(hist_MG))
        edge_idxs = np.arange(zero_idx+start_length, zero_idx+end_length+1, 1)
        back_idxs = np.arange(zero_idx+end_length+1, 
                               zero_idx+end_length+1+6,
                               1)
        #back_idxs2 = np.arange(zero_idx-(end_length+6), zero_idx-end_length, 1)
        #back_idxs = np.concatenate((back_idxs1, back_idxs2), axis=None)


        background = np.ones(100) * (sum(hist_MG[back_idxs]) / len(back_idxs))

        maximum = max(hist_MG[zero_idx-10:zero_idx+10])
        max_idx = np.where(hist_MG == maximum)[0][0]
        print('Max_idx: ' + str(max_idx))

        peak_idxs = np.where(hist_MG[zero_idx-10:zero_idx+10] > (maximum - background[max_idx])/2)
        peak_idxs = (np.array(peak_idxs) + zero_idx - 10)[0]
        
        diff_peak = hist_MG[peak_idxs] - background[peak_idxs]
        sum_peak = sum(diff_peak[np.where(diff_peak > 0)])
        diff_edge = hist_MG[edge_idxs] - background[edge_idxs]
        sum_edge = sum(diff_edge[np.where(diff_edge > 0)])






     #   back_idxs = np.arange(zero_idx+end_length+1, zero_idx+end_length+11)

     #   sum_back = sum(hist_MG[back_idxs])

    #    FOM = (  sum(hist_MG[zero_idx+start_length:zero_idx+end_length]) 
    #           / sum(hist_MG[zero_idx-end_length:zero_idx-start_length])
    #           )

        FOM = sum_edge/sum_peak
    
        print('Zero index: ' + str(zero_idx))
        print('Bin: ' + str(bin_centers[zero_idx]))
        print('Left: ' + str(bin_centers[zero_idx-end_length:zero_idx-start_length]))
        print('Left number of elements: ' + str(len(bin_centers[zero_idx-end_length:zero_idx-start_length])))
        print('Right: ' + str(bin_centers[zero_idx+start_length:zero_idx+end_length]))
        print('Right number of elements: ' + str(len(bin_centers[zero_idx+start_length:zero_idx+end_length])))

        return FOM, zero_idx, peak_idxs, edge_idxs, max_idx, background, back_idxs, maximum


    def find_gaussian_FoM(bin_centers, hist_MG, k, m, background):
        def Gaussian(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2)) + k*x + m
        zero_idx = 50
        indices = np.arange(zero_idx-30, zero_idx+30+1, 1)
        print(hist_MG[indices])
        popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[indices], hist_MG[indices], 
                                            p0=[0.17608019, -0.0178427, -0.16605704])
        print(popt)
        sigma = abs(popt[2])
        x0 = popt[1]

        # Calculate FoM
        indices_shoulder = np.where((bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma))
        indices_peak = np.where((bin_centers <= x0 + sigma) & (bin_centers >= x0 - sigma))
        shoulder_area = sum(hist_MG[indices_shoulder]) - sum(Gaussian(bin_centers[indices_shoulder], popt[0], popt[1], popt[2]))
        peak_area = sum(hist_MG[indices_peak]) - sum(background[indices_peak])
        FOM = shoulder_area/peak_area
        return (bin_centers, Gaussian(bin_centers, popt[0], popt[1], popt[2]),
                FOM, indices, popt[1], abs(popt[2]))

    def find_linear_background(bin_centers, hist_MG):
        def Linear(x, k, m):
            return k*x + m

        zero_idx    = 50
        indices_1   = np.arange(zero_idx-30, zero_idx-20, 1)
        indices_2   = np.arange(zero_idx+20, zero_idx+30, 1)
        indices     = np.concatenate((indices_1, indices_2), axis=None)
        popt, __ = scipy.optimize.curve_fit(Linear, bin_centers[indices], hist_MG[indices])
        return popt[0], popt[1], indices, Linear(bin_centers, popt[0], popt[1])


    dir_name = os.path.dirname(__file__)
    overview_folder = os.path.join(dir_name, '../Results/V_3x3_HR_overview/')
    E_i = np.loadtxt(overview_folder + 'E_i_vec.txt', delimiter=",")
    FWHM = np.loadtxt(overview_folder + 'MG_FWHM_vec.txt', delimiter=",")

    Van_3x3_HR_clusters = ["['mvmelst_125.mvmelst', '...'].h5",
                           "['mvmelst_127.mvmelst'].h5",
                           "['mvmelst_129.mvmelst'].h5",
                           "['mvmelst_130.mvmelst'].h5",
                           "['mvmelst_131.mvmelst'].h5",
                           "['mvmelst_132.mvmelst', '...'].h5",
                           "['mvmelst_134.mvmelst', '...'].h5",
                           "['mvmelst_137.mvmelst'].h5",
                           "['mvmelst_138.mvmelst'].h5",
                           "['mvmelst_139.mvmelst'].h5",
                           "['mvmelst_140.mvmelst'].h5",
                           "['mvmelst_141.mvmelst'].h5",
                           "['mvmelst_142.mvmelst', '...'].h5",
                           "['mvmelst_145.mvmelst', '...'].h5",
                           "['mvmelst_147.mvmelst'].h5",
                           "['mvmelst_148.mvmelst'].h5",
                           "['mvmelst_149.mvmelst'].h5",
                           "['mvmelst_150.mvmelst'].h5",
                           "['mvmelst_151.mvmelst'].h5",
                           "['mvmelst_1556_60meV_HR.zip'].h5",
                           "['mvmelst_1557_70meV_HR.zip', '...'].h5",
                           "['mvmelst_1561_80meV_HR.zip'].h5",
                           "['mvmelst_1562_90meV_HR.zip'].h5",
                           "['mvmelst_1563_100meV_HR.zip'].h5",
                           "['mvmelst_1564_120meV_HR.zip', '...'].h5",
                           "['mvmelst_1566_140meV_HR.zip'].h5",
                           "['mvmelst_1567_160meV_HR.zip', '...'].h5",
                           "['mvmelst_1569_180meV_HR.zip'].h5",
                           "['mvmelst_1570_200meV_HR.zip', '...'].h5",
                           "['mvmelst_1572_225meV_HR.zip'].h5",
                           "['mvmelst_1573_250meV_HR.zip', '...'].h5",
                           "['mvmelst_1575_275meV_HR.zip'].h5",
                           "['mvmelst_1576_300meV_HR.zip'].h5"
                           ]

    Van_3x3_HF_clusters = ["['mvmelst_1577_15meV_HF.zip'].h5",
                           "['mvmelst_1578_20meV_HF.zip'].h5",
                           "['mvmelst_1579_25meV_HF.zip', '...'].h5",
                           "['mvmelst_1581_31p7meV_HF.zip'].h5",
                           "['mvmelst_1582_34meV_HF.zip', '...'].h5",
                           "['mvmelst_1586_40p8meV_HF.zip'].h5",
                           "['mvmelst_1587_48meV_HF.zip', '...'].h5",
                           "['mvmelst_1591_60meV_HF.zip', '...'].h5",
                           "['mvmelst_1595_70meV_HF.zip', '...'].h5",
                           "['mvmelst_1597_80meV_HF.zip'].h5",
                           "['mvmelst_1598_90meV_HF.zip', '...'].h5",
                           "['mvmelst_1600_100meV_HF.zip', '...'].h5",
                           "['mvmelst_1602_120meV_HF.zip'].h5",
                           "['mvmelst_1603_140meV_HF.zip', '...'].h5",
                           "['mvmelst_1605_160meV_HF.zip'].h5",
                           "['mvmelst_1606_180meV_HF.zip'].h5",
                           "['mvmelst_1607_200meV_HF.zip'].h5",
                           "['mvmelst_1608_225meV_HF.zip'].h5",
                           "['mvmelst_1609_250meV_HF.zip'].h5",
                           "['mvmelst_1610_275meV_HF.zip'].h5",
                           "['mvmelst_1611_300meV_HF.zip', '...'].h5",
                           "['mvmelst_1613_350meV_HF.zip'].h5",
                           "['mvmelst_1614_400meV_HF.zip', '...'].h5",
                           "['mvmelst_153.mvmelst', '...'].h5",
                           "['mvmelst_156.mvmelst'].h5",
                           "['mvmelst_157.mvmelst'].h5",
                           "['mvmelst_158.mvmelst'].h5",
                           "['mvmelst_160.mvmelst'].h5",
                           "['mvmelst_161.mvmelst'].h5",
                           "['mvmelst_162.mvmelst'].h5",
                           "['mvmelst_163.mvmelst'].h5",
                           "['mvmelst_164.mvmelst'].h5",
                           "['mvmelst_165.mvmelst'].h5",
                           "['mvmelst_166.mvmelst'].h5",
                           "['mvmelst_167.mvmelst'].h5",
                           "['mvmelst_168.mvmelst'].h5",
                           "['mvmelst_169.mvmelst'].h5"]


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
                       "Van__3x3_High_Flux_Calibration_400.0",
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
                       "Van__3x3_High_Flux_Calibration_3500.0"]

    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../Clusters/MG/')
    overview_folder = os.path.join(dir_name, '../Results/V_3x3_HR_overview/')
    FWHM = np.loadtxt(overview_folder + 'MG_FWHM_vec.txt', delimiter=",")

    back_path = os.path.join(dir_name, '../Clusters/MG/Background.h5')
    df_back = pd.read_hdf(back_path, 'coincident_events') 

    temp_folder = os.path.join(dir_name, '../Results/temp_folder_FOM_energy/')
    #pdf_temp = os.path.join(dir_name, '../Results/pdf_temp/')
    #mkdir_p(pdf_temp)
    mkdir_p(temp_folder)
    colors = ['blue', 'red', 'green']
    labels = ['ILL', 'ESS.CLB', 'ESS.PA']
    count = 0
    start_length = 4
    end_length   = 9
    module_ranges = [[0, 2], [3, 5], [6, 8]]
    FOM_and_E = [[[], [], []], [[], [], []], [[], [], []]]
    positions = np.arange(0, len(HR_calibrations[1:-5]), 1)
    for run in range(2):
        for i in range(3):
            for file, cal, scale_factor, pos in zip(Van_3x3_HR_clusters[1:-5], HR_calibrations[1:-5], FWHM[1:-5], positions):
                path = folder + file
                df_temp = pd.read_hdf(path, 'coincident_events')
                data_sets = pd.read_hdf(path, 'data_set')['data_set'].iloc[0]
                E_i = float(cal[len('Van__3x3_High_Resolution_Calibration_'):])
                measurement_time = pd.read_hdf(path, 'measurement_time')['measurement_time'].iloc[0]
                calibration = pd.read_hdf(path, 'calibration')['calibration'].iloc[0]
                df_temp = filter_ce_clusters(window, df_temp)
                df_temp = df_temp[(df_temp.Bus >= module_ranges[i][0]) & 
                                  (df_temp.Bus <= module_ranges[i][1])
                                  ]
                df_back_temp = df_back[(df_back.Bus >= module_ranges[i][0]) &
                                       (df_back.Bus <= module_ranges[i][1])]

          #  __, __, hist_MG, bin_centers = dE_plot(df_temp, data_sets, E_i,
          #                                         calibration, measurement_time,
          #                                  scale_factor =                    window.back_yes.isChecked(), window)
            
                hist_MG, bin_centers, back_hist = dE_plot_peak(df_temp, data_sets, E_i, calibration,
                                                                   measurement_time, scale_factor, df_back_temp)

                # Normalize spectrums
                norm = sum(hist_MG)
                hist_MG = hist_MG/norm
                back_hist = back_hist/norm
                #FOM, zero_idx, peak_idxs, edge_idxs, max_idx, background, back_idxs, maximum = find_FOM(bin_centers, hist_MG, start_length, end_length, back_hist)
                k, m, b_indices, background = find_linear_background(bin_centers, hist_MG)
                Gx, Gy, FOM, G_indices, x0, sigma = find_gaussian_FoM(bin_centers, hist_MG, k, m, background)
                if run == 0:
                    FOM_and_E[i][0].append(float(cal[len('Van__3x3_High_Resolution_Calibration_'):]))
                    FOM_and_E[i][1].append(FOM)

                #fig.suptitle('Data set: ' + str(window.data_sets)) #, x=0.5, y=1.08)
                if run == 1:
                    fig = plt.figure()
                    fig.set_figheight(6)
                    fig.set_figwidth(14)
                    # First subplot
                    plt.subplot(1, 2, 1)
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.plot(bin_centers, hist_MG, label='Multi-Grid', color='black', zorder=3)
                    plt.plot(Gx, Gy, label='Gaussian fit', color='red', zorder=3)
                    #plt.plot(bin_centers[b_indices], hist_MG[b_indices], 'rx', label='Bins used for fit', zorder=3)


                    plt.fill_between(bin_centers[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                                     hist_MG[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                                     Gy[(bin_centers >= x0 + 3*sigma) & (bin_centers <= x0 + 5*sigma)],
                                     facecolor='orange',
                                     label='[3$\sigma$, 5$\sigma$]',
                                     alpha=0.9, zorder=2) 
                    plt.fill_between(bin_centers[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                                     hist_MG[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                                     background[(bin_centers >= x0 - sigma) & (bin_centers <= x0 + sigma)],
                                     facecolor='purple',
                                     label='[-$\sigma$, $\sigma$]',
                                     alpha=0.9, zorder=2) 


                    plt.yscale('log')
                    plt.xlabel('E$_i$-E$_f$ [meV]')
                    plt.ylabel('Normalized counts')
                    plt.ylim(0.0001, 0.2)
                    plt.title('Energy transfer\n(Calibration: ' + cal + ')\n(' + 'Detector: ' + labels[i] +')')
                    plt.legend(loc=1)
                    # Second subplot
                    plt.subplot(1, 2, 2)
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.xlim(2, 300)
                    #plt.ylim(0, 0.055)
                    plt.xscale('log')
                    #plt.yscale('log')
                    plt.xlabel('Energy [meV]')
                    plt.ylabel('Figure-of-Merit (Tail area / Peak area)')
                    plt.title('Figure-of-Merit vs Energy')
                    for j in range(3):
                        average = round(sum(FOM_and_E[j][1])/len(FOM_and_E[j][1]), 4)
                        plt.plot(FOM_and_E[j][0], FOM_and_E[j][1], color=colors[j],
                                 label=labels[j] + ', average = ' + str(average),
                                 linestyle='-', marker='.')
                        plt.plot(FOM_and_E[i][0][pos], FOM_and_E[i][1][pos], 'o', color=colors[i])
                        plt.axvline(x=FOM_and_E[i][0][pos], color=colors[i], zorder=5)
                    
                    plt.legend(loc=2)
                    plt.tight_layout()
                    plt.savefig(temp_folder + str(count) + '.png')
                    #plt.savefig(pdf_temp + str(count) + '.pdf')
                    plt.close()
                    count += 1

    print('ILL average: ' + str(sum(FOM_and_E[0][1])/len(FOM_and_E[0][1])))
    print('ESS-CLB average: ' + str(sum(FOM_and_E[1][1])/len(FOM_and_E[1][1])))
    print('ESS-PA average: ' + str(sum(FOM_and_E[i][1])/len(FOM_and_E[i][1])))

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name, '../Results/Animations/FOM_sweep_energies_HR.gif')
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.update()
    window.app.processEvents()









# =============================================================================
# Angular Dependence Animation
# =============================================================================

def angular_animation_plot(window):
    Van_3x3_HR_clusters = ["['mvmelst_125.mvmelst', '...'].h5",
                           "['mvmelst_127.mvmelst'].h5",
                           "['mvmelst_129.mvmelst'].h5",
                           "['mvmelst_130.mvmelst'].h5",
                           "['mvmelst_131.mvmelst'].h5",
                           "['mvmelst_132.mvmelst', '...'].h5",
                           "['mvmelst_134.mvmelst', '...'].h5",
                           "['mvmelst_137.mvmelst'].h5",
                           "['mvmelst_138.mvmelst'].h5",
                           "['mvmelst_139.mvmelst'].h5",
                           "['mvmelst_140.mvmelst'].h5",
                           "['mvmelst_141.mvmelst'].h5",
                           "['mvmelst_142.mvmelst', '...'].h5",
                           "['mvmelst_145.mvmelst', '...'].h5",
                           "['mvmelst_147.mvmelst'].h5",
                           "['mvmelst_148.mvmelst'].h5",
                           "['mvmelst_149.mvmelst'].h5",
                           "['mvmelst_150.mvmelst'].h5",
                           "['mvmelst_151.mvmelst'].h5",
                           "['mvmelst_1556_60meV_HR.zip'].h5",
                           "['mvmelst_1557_70meV_HR.zip', '...'].h5",
                           "['mvmelst_1561_80meV_HR.zip'].h5",
                           "['mvmelst_1562_90meV_HR.zip'].h5",
                           "['mvmelst_1563_100meV_HR.zip'].h5",
                           "['mvmelst_1564_120meV_HR.zip', '...'].h5",
                           "['mvmelst_1566_140meV_HR.zip'].h5",
                           "['mvmelst_1567_160meV_HR.zip', '...'].h5",
                           "['mvmelst_1569_180meV_HR.zip'].h5",
                           "['mvmelst_1570_200meV_HR.zip', '...'].h5",
                           "['mvmelst_1572_225meV_HR.zip'].h5",
                           "['mvmelst_1573_250meV_HR.zip', '...'].h5",
                           "['mvmelst_1575_275meV_HR.zip'].h5",
                           "['mvmelst_1576_300meV_HR.zip'].h5"]

    Van_3x3_HF_clusters = ["['mvmelst_1577_15meV_HF.zip'].h5",
                           "['mvmelst_1578_20meV_HF.zip'].h5",
                           "['mvmelst_1579_25meV_HF.zip', '...'].h5",
                           "['mvmelst_1581_31p7meV_HF.zip'].h5",
                           "['mvmelst_1582_34meV_HF.zip', '...'].h5",
                           "['mvmelst_1586_40p8meV_HF.zip'].h5",
                           "['mvmelst_1587_48meV_HF.zip', '...'].h5",
                           "['mvmelst_1591_60meV_HF.zip', '...'].h5",
                           "['mvmelst_1595_70meV_HF.zip', '...'].h5",
                           "['mvmelst_1597_80meV_HF.zip'].h5",
                           "['mvmelst_1598_90meV_HF.zip', '...'].h5",
                           "['mvmelst_1600_100meV_HF.zip', '...'].h5",
                           "['mvmelst_1602_120meV_HF.zip'].h5",
                           "['mvmelst_1603_140meV_HF.zip', '...'].h5",
                           "['mvmelst_1605_160meV_HF.zip'].h5",
                           "['mvmelst_1606_180meV_HF.zip'].h5",
                           "['mvmelst_1607_200meV_HF.zip'].h5",
                           "['mvmelst_1608_225meV_HF.zip'].h5",
                           "['mvmelst_1609_250meV_HF.zip'].h5",
                           "['mvmelst_1610_275meV_HF.zip'].h5",
                           "['mvmelst_1611_300meV_HF.zip', '...'].h5",
                           "['mvmelst_1613_350meV_HF.zip'].h5",
                           "['mvmelst_1614_400meV_HF.zip', '...'].h5",
                           "['mvmelst_153.mvmelst', '...'].h5",
                           "['mvmelst_156.mvmelst'].h5",
                           "['mvmelst_157.mvmelst'].h5",
                           "['mvmelst_158.mvmelst'].h5",
                           "['mvmelst_160.mvmelst'].h5",
                           "['mvmelst_161.mvmelst'].h5",
                           "['mvmelst_162.mvmelst'].h5",
                           "['mvmelst_163.mvmelst'].h5",
                           "['mvmelst_164.mvmelst'].h5",
                           "['mvmelst_165.mvmelst'].h5",
                           "['mvmelst_166.mvmelst'].h5",
                           "['mvmelst_167.mvmelst'].h5",
                           "['mvmelst_168.mvmelst'].h5",
                           "['mvmelst_169.mvmelst'].h5"]

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
                       "Van__3x3_High_Flux_Calibration_400.0",
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
                       "Van__3x3_High_Flux_Calibration_3500.0"]


    window.ang_dist_progress.show()
    window.update()
    window.app.processEvents()

    dir_name = os.path.dirname(__file__)
    MG_folder = os.path.join(dir_name, '../Clusters/MG/')
    temp_folder = os.path.join(dir_name, '../temp_angle_dE_sweep_folder/')
    results_folder = os.path.join(dir_name, '../Results/Animations/')
    mkdir_p(temp_folder)
    count = 0
    for MG_file, calibration in zip(Van_3x3_HF_clusters, HF_calibrations):
        MG_path = MG_folder + MG_file
        df_temp_MG = pd.read_hdf(MG_path, 'coincident_events')
        df_temp_MG = filter_ce_clusters(window, df_temp_MG)
        path = temp_folder + str(count) + '.png'
        progress = round((count/(len(HR_calibrations))) * 100, 1)
        window.ang_dist_progress.setValue(progress)
        window.update()
        window.app.processEvents()
        fig = angular_dependence_plot(df_temp_MG, calibration,
                                      calibration)
        fig.savefig(path)
        plt.close()
        count += 1
        print(count)

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = results_folder + 'angle_dependence_dE_sweep.gif'
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images) #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.ang_dist_progress.close()
    window.update()
    window.app.processEvents()


# =============================================================================
# C4H2I2S - Compare all energies
# =============================================================================

def C4H2I2S_compare_all_energies(window, isPureAluminium=True):
    dir_name = os.path.dirname(__file__)
    MG_folder = os.path.join(dir_name, '../Clusters/MG/')
    He3_folder = os.path.join(dir_name, '../Archive/2019_01_10_SEQ_Diiodo/')

    MG_files = ["C4H2I2S_21meV.h5",
                "['mvmelst_1361.mvmelst', '...'].h5",
                "['mvmelst_1260.mvmelst', '...'].h5",
                "['mvmelst_1266.mvmelst', '...'].h5",
                "['mvmelst_1301.mvmelst', '...'].h5",
                "['mvmelst_1319.mvmelst', '...'].h5",
                "['mvmelst_1334.mvmelst', '...'].h5",
                "['mvmelst_1348.mvmelst', '...'].h5"
                ]
    
    He3_files_full = [['169112', '169113', '169114', '169115', '169116', '169117'],
                      ['169118', '169119', '169120', '169121', '169122', '169123'],
                      ['169124', '169125', '169126', '169127', '169128', '169129'],
                      ['169130', '169131', '169132', '169133', '169134', '169135'],
                      ['169136', '169137', '169138', '169139', '169140', '169141'],
                      ['169142', '169143', '169144', '169145', '169146', '169147'],
                      ['169148', '169149', '169150', '169151', '169152', '169153'],
                      ['169154', '169155', '169156', '169157', '169158', '169159']
                      ]

    Energies = [21, 35, 50.12, 70.13, 99.9, 197.3, 296.1, 492]



    Calibrations = ['C4H2I2S_21.0',
                    'C4H2I2S_35.0',
                    'C4H2I2S_50.0',
                    'C4H2I2S_70.0',
                    'C4H2I2S_100.0',
                    'C4H2I2S_200.0',
                    'C4H2I2S_300.0',
                    'C4H2I2S_500.0',
                    ]

    number_of_bins = 290

    He3_solid_angle = 0.7449028590952331
    MG_solid_angle = 0.01660177142644554 - 0.0013837948633069277

    MG_number_files = [38, 37, 6, 35, 18, 15, 14, 13]
    no_glitch_times = [12164, 6318, 4709, 6479, 6666, 7394, 5872, 6532]
    MG_charge = 3
    He3_charge = 6 * 0.2778

    for MG_file, He3_files, Energy, calibration, file_number, no_glitch_time in zip(MG_files, He3_files_full,
                                                                                    Energies, Calibrations,
                                                                                    MG_number_files,
                                                                                    no_glitch_times):
        # Import He3 data
        He3_dE_hist_full = np.zeros(290)
        for file_number, measurement_id in enumerate(He3_files):
            nxs_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxs'
            nxs_path = He3_folder + nxs_file
            nxs = h5py.File(nxs_path, 'r')
            he3_bins = nxs['mantid_workspace_1']['event_workspace']['axis1'].value
            print('Number of bins: ' + str(len(he3_bins)))
            he3_min = he3_bins[0]
            he3_max = he3_bins[-1]
            He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
            dE = nxs['mantid_workspace_1']['event_workspace']['tof'].value
            He3_dE_hist, __ = np.histogram(dE, bins=number_of_bins,
                                           range=[he3_min, he3_max])
            He3_dE_hist_full += He3_dE_hist
        # Import MG data
        df = pd.read_hdf(MG_folder+MG_file, 'coincident_events')
        df = df[df.d != -1]
        df = filter_ce_clusters(window, df)
        t_off = np.ones(df.shape[0]) * get_t_off(calibration)
        T_0 = get_T0(calibration, Energy) * np.ones(df.shape[0])
        frame_shift = get_frame_shift(Energy) * np.ones(df.shape[0])
        E_i = Energy * np.ones(df.shape[0])
        ToF = df.ToF.values
        d = df.d.values
        dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
        df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
        dE = df_temp[df_temp['t_f'] > 0].dE
        # Get MG dE histogram
        MG_dE_hist, __ = np.histogram(dE, bins=number_of_bins,
                                      range=[he3_min, he3_max]
                                      )

        
        measurement_time = pd.read_hdf(MG_folder+MG_file, 'measurement_time')['measurement_time'].iloc[0]

        fig = plt.figure()
        change_time = 20
        time_frac = no_glitch_time/((file_number-1)*change_time + measurement_time)
        if Energy == 21:
            time_frac = no_glitch_time/((file_number-1)*change_time + 14972)
        charge_norm = (MG_charge * time_frac)/He3_charge
        tot_norm = ((MG_solid_angle/He3_solid_angle)*charge_norm)
        print('tot_norm: ' + str(tot_norm))

        hist_back = plot_dE_background(Energy, calibration, measurement_time,
                                       tot_norm, he3_min, he3_max, back_yes,
                                       tot_norm, window, isPureAluminium, None,
                                       number_of_bins)

        title = 'C$_4$H$_2$I$_2$S\nE$_i$ = ' + str(Energy)
        if back_yes:
            MG_dE_hist = MG_dE_hist/tot_norm
        else:
            MG_dE_hist = MG_dE_hist/tot_norm - hist_back
            title += ', Background reduced'
        plt.plot(He3_bin_centers, MG_dE_hist,
                 color='red',
                 label='Multi-Grid')
        plt.plot(He3_bin_centers, He3_dE_hist_full,
                 '--',
                 color='blue',
                 label='$^3$He-tubes')
        if back_yes:
            plt.plot(He3_bin_centers, hist_back, color='green', label='Background') 
        plt.legend(loc=1)
        fig = stylize(fig, 'Energy transfer [meV]', 'Normalized counts', title,
                      'linear', 'log', [he3_min, he3_max], None, True)
        file_name = 'C4H2I2S_' + str(Energy) + '_meV.pdf'        
        plt.savefig(os.path.join(dir_name, '../Results/C4H2I2S/' + file_name))
        plt.close()


# =============================================================================
# Count rate
# =============================================================================

def get_count_rate(ce, duration, data_sets, window):
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    name = 'ToF\nData set(s): ' + str(data_sets)
    fig = plt.figure()
    rnge = [window.ToF_min.value(), window.ToF_max.value()]
    number_bins = 2000
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    hist, bins, patches = plt.hist(ce.ToF * 62.5e-9 * 1e6, bins=number_bins,
                                   range=rnge,
                                   log=True, color='black', zorder=4,
                                   histtype='step')
    plt.xlabel('ToF [$\mu$s]')
    plt.ylabel('Counts')
    plt.title(name)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    maximum = max(hist)
    peak_idxs = np.where(hist == maximum)
    peak_idx = peak_idxs[len(peak_idxs)//2][0]
    start = find_nearest(hist[:peak_idx], hist[peak_idx]/2)
    stop = find_nearest(hist[peak_idx:], hist[peak_idx]/2)
    print('Bincenters')
    print(bin_centers[start])
    print(bin_centers[peak_idx+stop])
    plt.plot([bin_centers[start], bin_centers[peak_idx+stop]], [hist[start],
             hist[peak_idx+stop]], '-x', zorder=7)
    tot_events = ce.shape[0]
    events_per_s = tot_events/duration
    print('events_per_s' + str(events_per_s))
    print('signal_events: ' + str(sum(hist[start:peak_idx+stop])))
    s_time = (bin_centers[peak_idx+stop] - bin_centers[start]) * 1e-6
    period_time = 0.016666666666666666
    print('S_time: ' + str(s_time))
    s_events_per_s = sum(hist[start:peak_idx+stop])/(duration*(s_time/period_time))
    print('S events per s: %f' % s_events_per_s)
    print(ce)
    ce_peak = ce[  (ce.ToF*(62.5e-9)*(1e6) >= bin_centers[start]) 
                 & (ce.ToF*(62.5e-9)*(1e6) <= bin_centers[peak_idx+stop])
                 ]
    print(ce_peak)

    __ , __, __, __, __, maximum_voxel = Coincidences_3D_plot(ce_peak, '')
    print('Maximum voxel: %f, %f, %f' % (maximum_voxel[0], maximum_voxel[1], maximum_voxel[2]))
    ce_voxel = ce_peak[(ce_peak.Bus == maximum_voxel[0]) & (ce_peak.wCh == maximum_voxel[1]) & (ce_peak.gCh == maximum_voxel[2])]
 
    text_string = 'Average rate: ' + str(round(events_per_s, 1)) + ' [n/s]'
    text_string += '\nPeak Rate: ' + str(round(s_events_per_s, 1)) + ' [n/s]'
   
    plt.text(16000, maximum*0.7, text_string, ha='right', va='top',
             bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10}, fontsize=8,
             zorder=5)
    fig.show()


# =============================================================================
# Beam monitor
# =============================================================================

def beam_monitor_histogram():
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

    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../Archive/V_3x3_HR_2meV_to_50meV_MG_nexusFiles/')
    files = np.array([file for file in os.listdir(folder) if file[-3:] == '.h5'])
    number_of_bins = 200
    chopper_1 = {'MG': {'phase': [], 'rotation_speed': []},
                 'He3': {'phase': [], 'rotation_speed': []}
                 }
    chopper_2 = {'MG': {'phase': [], 'rotation_speed': []},
                 'He3': {'phase': [], 'rotation_speed': []}
                 }
    chopper_3 = {'MG': {'phase': [], 'rotation_speed': []},
                 'He3': {'phase': [], 'rotation_speed': []}
                 }
    energies = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]
    ranges = [[12e3, 14e3],
              [6e3, 8e3],
              [3e3, 5e3],
              [1e3, 3e3],
              [0e3, 3e3],
              [15e3, 17e3],
              [14e3, 16e3],
              [13e3, 15e3],
              [12e3, 14e3],
              [12e3, 14e3],
              [11e3, 13e3],
              [10e3, 12e3],
              [10e3, 12e3],
              [9e3, 11e3],
              [11e3, 13e3],
              [8e3, 10e3],
              [7e3, 9e3],
              [6e3, 8e3],
              [6e3, 8e3],
              [5.5e3, 6.5e3]
              ]
    print(files)
    for i, file in enumerate(files):
        # Find calibration
        calibration = HR_calibrations[i]
        # Get beam monitor data from Multi-Grid measurements
        path_MG = folder + file
        data_MG = h5py.File(path_MG, 'r')
        print('Duration: %f' % data_MG['entry']['duration'].value)
        ToF_MG = data_MG['entry']['monitor1']['event_time_offset'].value
        # Get beam monitor data from He3 measurements
        m_id = str(find_He3_measurement_id(calibration))
        path_He3 = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
        data_He3 = h5py.File(path_He3, 'r')
        ToF_He3 = data_He3['entry']['monitor1']['event_time_offset'].value
        # Get all chopper data
        for j, chopper in enumerate([chopper_1, chopper_2, chopper_3]):
            chopper_id = 'chopper%d' % (j+1)
            #print('chopper_id: %s' % chopper_id)
            for data, name in zip([data_He3, data_MG], ['He3', 'MG']):
                chopper_temp = data['entry']['instrument'][chopper_id]
                chopper[name]['phase'].append(chopper_temp['phase']['average_value'].value)
                chopper[name]['rotation_speed'].append(chopper_temp['rotation_speed']['average_value'].value)
                #print('phase: %f' % chopper_temp['phase']['average_value'].value)
                #print('rotation speed: %f' % chopper_temp['rotation_speed']['average_value'].value)
        # Get elastic peak location
        hist, bins = np.histogram(ToF_He3, bins=500, range=[200, 17500])
        max_idx = np.where(hist == max(hist))
        elastic_peak = bins[max_idx]
        # Plot data
        fig = plt.figure()
        plt.yscale('log')
        plt.title(calibration)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.xlabel('ToF [µs]')
        plt.ylabel('Counts')
        plt.hist(ToF_MG, bins=number_of_bins, histtype='step',
                 range=[elastic_peak-250, elastic_peak+250],
                 color='red', label='Multi-Grid measurement', zorder=3)
        plt.hist(ToF_He3, bins=number_of_bins, histtype='step',
                 range=[elastic_peak-250, elastic_peak+250],
                 color='blue', label='He3 measurement', zorder=3)
        plt.legend()
        save_path = os.path.join(dir_name, '../Results/Beam_monitor/%s.pdf' % calibration)
        fig.savefig(save_path)
        plt.close()

    fig = plt.figure()
    colors = ['blue', 'red']
    ylabels = ['Phase', 'Rotation speed']
    # Plot all chopper data
    for i, chopper in enumerate([chopper_1, chopper_2, chopper_3]):
        for j, detector in enumerate(chopper.keys()):
            for k, value in enumerate(chopper[detector].keys()):
                plt.subplot(2, 3, i+1+(k*3))
                plt.grid(True, which='major', linestyle='--', zorder=0)
                plt.grid(True, which='minor', linestyle='--', zorder=0)
                plt.plot(energies, chopper[detector][value], label=detector,
                         zorder=3, color=colors[j], marker='.', linestyle=None)
                if k == 0:
                    plt.title('Chopper %d' % (i+1))
                plt.xlabel('Energy [meV]')
                plt.ylabel(ylabels[k])
                plt.legend()
    plt.tight_layout()
    return fig


def plot_He3_variation():
    dir_name = os.path.dirname(__file__)
    number_bins = 100
    calibrations = get_all_calibrations()
    for calibration in calibrations:
        m_id = str(find_He3_measurement_id(calibration))
        input_path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
        file = h5py.File(input_path, 'r')
        bank_values = []
        bin_center_values = []
        hist_values = []
        tot_counts = []
        for bank_value in range(40, 151):
            bank = 'bank' + str(bank_value) + '_events'
            ToF = file['entry'][bank]['event_time_offset'].value
            hist, bins = np.histogram(ToF, bins=number_bins)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            bank_values.extend(bank_value*np.ones(number_bins))
            bin_center_values.extend(bin_centers)
            hist_values.extend(hist)
            tot_counts.append(0 + len(ToF))

        fig = plt.figure()
        height = 5
        width = 10
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.suptitle(calibration, y=1.01)
        plt.subplot(1, 2, 1)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.plot(np.arange(40, 151, 1), tot_counts, 'o', color='black', zorder=3)
        plt.xlabel('Bank')
        plt.ylabel('Total counts')

        plt.subplot(1, 2, 2)
        plt.hist2d(bank_values, bin_center_values, bins=[112, number_bins],
                   weights=hist_values, norm=LogNorm(), cmap='jet')
        plt.xlabel('Bank')
        plt.ylabel('ToF [µs]')
        plt.colorbar()

        plt.tight_layout()
        output_folder = os.path.join(dir_name, '../Results/He3_variations/')
        output_path = output_folder + calibration + '.pdf'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()
            


def plot_He3_variation_dE():
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def Gaussian_fit(bin_centers, dE_hist, p0):
        def Gaussian(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
        
        center_idx = len(bin_centers)//2
        zero_idx = find_nearest(bin_centers[center_idx-20:center_idx+20], 0) + center_idx - 20
        fit_bins = np.arange(zero_idx-20, zero_idx+20+1, 1)

        popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[fit_bins],
                                            dE_hist[fit_bins], p0=p0)
        x0 = popt[1]
        sigma = abs(popt[2])
        #print('Sigma: %f' % sigma)
        #print('x0: %f' % x0)

        left_edge = x0 - sigma
        right_edge = x0 + sigma
        left_idx = find_nearest(bin_centers, left_edge)
        right_idx = find_nearest(bin_centers, right_edge)
        area_bins = sum(dE_hist[left_idx:right_idx])

        return left_edge, right_edge, popt, area_bins

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

    dir_name = os.path.dirname(__file__)
    calibrations = get_all_calibrations()
    #energies = np.concatenate((HR_energies, HF_energies), axis=None)
    energies = get_all_energies(calibrations)
    __, __, __, d, __, __ = import_He3_coordinates_raw()
    number_bins = 300
    start = 0
    p0_cal = None
    for calibration, energy in zip(calibrations[start:], energies[start:]):
        print(calibration)
        m_id = str(find_He3_measurement_id(calibration))
        input_path = os.path.join(dir_name, '../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
        file = h5py.File(input_path, 'r')
        bank_values = []
        bin_center_values = []
        hist_values = []
        peak_counts = []
        peak_counts_bins = []
        tot_counts = []
        fig = plt.figure()
        #plt.subplot(1, 2, 2)
        for bank_value in range(40, 151):
            bank = 'bank' + str(bank_value) + '_events'
            ToF = file['entry'][bank]['event_time_offset'].value
            pixels = file['entry'][bank]['event_id'].value
            distance = np.zeros([len(pixels)], dtype=float)
            for i, pixel in enumerate(pixels):
                distance[i] = d[pixel-37888]
            T_0 = get_T0(calibration, energy) * np.ones(len(ToF))
            t_off = get_t_off_He3(calibration) * np.ones(len(ToF))
            E_i = energy * np.ones(len(ToF))
            dE, t_f = get_dE_He3(E_i, ToF, distance, T_0, t_off)
            df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
            dE_filtered = df_temp[df_temp['t_f'] > 0].dE
            dE_hist, bins = np.histogram(dE_filtered, bins=number_bins,
                                         range=[-energy, energy])
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            bank_values.extend(bank_value*np.ones(number_bins))
            bin_center_values.extend(bin_centers)
            hist_values.extend(dE_hist)
            left_edge, right_edge, p0, area_bins = Gaussian_fit(bin_centers, dE_hist, p0_cal)
            if bank_value == 40:
                p0_cal = p0
            xmin = [40, 40]
            xmax = [150, 150]
           # plt.hlines([left_edge, right_edge], xmin, xmax)
            area_peak = dE_filtered[(dE_filtered >= left_edge) & (dE_filtered <= right_edge)].shape[0]
            peak_counts.append(area_peak)
            tot_counts.append(sum(dE_hist))
            peak_counts_bins.append(area_bins)

        
        height = 5
        width = 10
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.suptitle(calibration, y=1.01)
        plt.subplot(1, 2, 1)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.plot(np.arange(40, 151, 1), peak_counts, '.', color='black', zorder=3, label='Elastic peak area (±σ)')
       # plt.plot(np.arange(40, 151, 1), peak_counts_bins, '.', color='blue', zorder=3, label='Elastic peak from bins')
        plt.plot(np.arange(40, 151, 1), tot_counts, '.', color='red', zorder=3, label='Total area')
        plt.legend(loc=1)
        plt.xlabel('Bank')
        plt.ylabel('Counts')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.hist2d(bank_values, bin_center_values, bins=[112, number_bins],
                   weights=hist_values, norm=LogNorm(), cmap='jet')
        plt.xlabel('Bank')
        plt.ylabel('E$_i$ - E$_f$ [meV]')
        plt.colorbar()

        plt.tight_layout()
        output_folder = os.path.join(dir_name, '../Results/He3_variations_dE/')
        output_path = output_folder + calibration + '.pdf'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()



# =============================================================================
# Helper Functions
# ============================================================================= 

def stylize(fig, x_label, y_label, title, xscale='linear', yscale='linear',
            xlim=None, ylim=None, grid=False):
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

    
def get_plot_path(data_set):
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Plot/' + data_set + '/')
    return folder

def get_output_path(data_set):
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Output/' + data_set + '/')
    return folder

def import_helium_tubes():
    pass

def filter_clusters(df):
    df = df[df.d != -1]
    df = df[df.tf > 0]
    df = df[(df.wADC > 500) & (df.gADC > 400)]
    df = df[(df.wM == 1) & (df.gM <= 5)]
    return df


def calculate_peak_norm(bin_centers, hist, left_edge, right_edge):
    x_l = bin_centers[left_edge]
    y_l = hist[left_edge]
    x_r = bin_centers[right_edge-1]
    y_r = hist[right_edge-1]
    area = sum(hist[left_edge:right_edge])
    bins_under_peak = abs(right_edge - 1 - left_edge)
    area_noise = ((abs(y_r - y_l) * bins_under_peak) / 2
                  + bins_under_peak * min([y_l, y_r]))
    peak_area = area - area_noise
    return peak_area


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
    


def get_dE(E_i, ToF, d, T_0, t_off, frame_shift):
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
    

def import_T0_table():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/' + 'T0_vs_Energy.xlsx')
    matrix = pd.read_excel(path).values
    t0_table = {}
    for row in matrix:
        t0_table.update({str(row[0]): row[1]})
    return t0_table


def import_Ei_table():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/' + 'experiment_log.xlsx')
    matrix = pd.read_excel(path).values
    Ei_table = {}
    for row in matrix:
        Ei_table.update({str(row[0]): row[59]})
    return Ei_table


def get_Ei(measurement_id):
    Ei_table = import_Ei_table()
    return Ei_table[str(measurement_id)]


def get_T0(calibration, energy):
    T0_table = import_T0_table()
    return T0_table[calibration]


def get_t_off_He3(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/He3_offset.xlsx')
    matrix = pd.read_excel(path).values
    He3_offset_table = {}
    for row in matrix:
        He3_offset_table.update({row[0]: row[2]})
    offset = float(He3_offset_table[calibration])
    return offset


def get_t_off_table():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/' + 'time_offset.xlsx')
    matrix = pd.read_excel(path).values
    t_off_table = {}
    for row in matrix:
        t_off_table.update({row[0]: row[1]})
    return t_off_table


def get_t_off(calibration):
    t_off_table = get_t_off_table()
    return t_off_table[calibration]


def get_frame_shift(E_i):
    frame_shift = 0
    if E_i == 2:
        frame_shift += 2 * (16666.66666e-6) - 0.0004475
    if E_i == 3:
        frame_shift += 2 * (16666.66666e-6) - 0.00800875
    if E_i == 4:
        frame_shift += 2 * (16666.66666e-6) - 0.0125178125
    if E_i == 5:
        frame_shift += 2 * (16666.66666e-6) - 0.015595
    if E_i == 6:
        frame_shift += (16666.66666e-6) - 0.001190399
    if E_i == 7:
        frame_shift += (16666.66666e-6) - 0.002965625
    if E_i == 8:
        frame_shift += (16666.66666e-6) - 0.0043893
    if E_i == 9:
        frame_shift += (16666.66666e-6) - 0.0055678125
    if E_i == 10:
        frame_shift += (16666.66666e-6) - 0.0065653125
    if E_i == 12:
        frame_shift += (16666.66666e-6) - 0.00817125
    if E_i == 14:
        frame_shift += (16666.66666e-6) - 0.00942
    if E_i == 15:
        frame_shift += (16666.66666e-6) - 0.009948437499999999
    if E_i == 16:
        frame_shift += (16666.66666e-6) - 0.01042562499
    if E_i == 18:
        frame_shift += (16666.66666e-6) - 0.011259375
    if E_i == 21:
        frame_shift += (16666.66666e-6) - 0.01227875
    if E_i == 20:
        frame_shift += (16666.66666e-6) - 0.011965
    if E_i == 25:
        frame_shift += (16666.66666e-6) - 0.013340625
    if E_i == 30:
        frame_shift += (16666.66666e-6) - 0.01435625
    if E_i == 32:
        frame_shift += (16666.66666e-6) - 0.014646875
    if E_i == 34:
        frame_shift += (16666.66666e-6) - 0.015009375
    if E_i == 35:
        frame_shift += (16666.66666e-6) - 0.01514625
    if E_i == 40:
        frame_shift += (16666.66666e-6) - 0.0157828125
    if E_i == 40.8:
        frame_shift += (16666.66666e-6) - 0.015878125
    if E_i == 48:
        frame_shift += (16666.66666e-6) - 0.0165909375
    return frame_shift


def find_He3_measurement_id(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/experiment_log.xlsx')
    matrix = pd.read_excel(path).values
    measurement_table = {}
    for row in matrix:
        measurement_table.update({row[1]: row[0]})
    return measurement_table[calibration]

def find_He3_measurement_calibration(id):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/experiment_log_June20.xlsx')
    matrix = pd.read_excel(path).values
    measurement_table = {}
    for row in matrix:
        measurement_table.update({row[0]: row[1]})
    return measurement_table[id]



def get_He3_offset(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/He3_offset.xlsx')
    matrix = pd.read_excel(path).values
    He3_offset_table = {}
    for row in matrix:
        He3_offset_table.update({row[0]: row[1]})
    
    offset = float(He3_offset_table[calibration])
    
    return offset


def get_FWHM(bin_centers, hist, left_edge, right_edge, vis_help,
             b_label='Background'):
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    # Calculate background level
    x_l = bin_centers[left_edge]
    y_l = hist[left_edge]
    x_r = bin_centers[right_edge]
    y_r = hist[right_edge]
    par_back = np.polyfit([x_l, x_r], [y_l, y_r], deg=1)
    f_back = np.poly1d(par_back)
    xx_back = np.linspace(x_l, x_r, 100)
    yy_back = f_back(xx_back)
     
    #plt.plot(xx_back, yy_back, 'orange', label=b_label)
     
    bins_under_peak = abs(right_edge - 1 - left_edge)
    
    area_noise = ((abs(y_r - y_l) * bins_under_peak) / 2 
                      + bins_under_peak * y_l)
        
    area = sum(hist[left_edge:right_edge])
    peak_area = area - area_noise
        
    # Calculate HM
  #  peak = peakutils.peak.indexes(hist[left_edge:right_edge])
    maximum = max(hist[left_edge:right_edge])
    peak=np.where(hist[left_edge:right_edge] == maximum)
    peak = peak[len(peak)//2][len(peak)//2]
    if vis_help:
        plt.plot(bin_centers[left_edge:right_edge][peak],
                 hist[left_edge:right_edge][peak], 'bx', label='Maximum', 
                 zorder=5)
    M = hist[left_edge:right_edge][peak]
    xM = bin_centers[left_edge:right_edge][peak]
    noise_level = yy_back[find_nearest(xx_back, xM)]
    HM = (M-noise_level)/2 + noise_level
    SNR = M/noise_level
    SNR = SNR

    # Calculate FWHM
    left_idx = find_nearest(hist[left_edge:left_edge+peak], HM)
    right_idx = find_nearest(hist[left_edge+peak:right_edge], HM)
        
    sl = []
    sr = []
        
    if hist[left_edge+left_idx] > HM:
        sl = [-1, 0]
    else:
        sl = [0, 1]
        
    if hist[left_edge+peak+right_idx] < HM:
        rl = [-1, 0]
    else:
        rl = [0, 1]
        
    left_x = [bin_centers[left_edge+left_idx+sl[0]], 
              bin_centers[left_edge+left_idx+sl[1]]]
    left_y = [hist[left_edge+left_idx+sl[0]], hist[left_edge+left_idx+sl[1]]]
    right_x = [bin_centers[left_edge+peak+right_idx+rl[0]], 
               bin_centers[left_edge+peak+right_idx+rl[1]]]
    right_y = [hist[left_edge+peak+right_idx+rl[0]], 
               hist[left_edge+peak+right_idx+rl[1]]]

    par_left = np.polyfit(left_x, left_y, deg=1)
    f_left = np.poly1d(par_left)
    par_right = np.polyfit(right_x, right_y, deg=1)
    f_right = np.poly1d(par_right)
        
    xx_left = np.linspace(left_x[0], left_x[1], 100)
    xx_right = np.linspace(right_x[0], right_x[1], 100)
    yy_left = f_left(xx_left)
    yy_right = f_right(xx_right)
    if vis_help:
        plt.plot(xx_left, yy_left, 'blue', label=None)
        plt.plot(xx_right, yy_right, 'blue', label=None)
        
    left_idx = find_nearest(yy_left, HM)
    right_idx = find_nearest(yy_right, HM)
        
        
    if vis_help:
        plt.plot([xx_left[left_idx], xx_right[right_idx]], 
                 [HM, HM], 'g', label='FWHM')
        
    L = xx_left[left_idx]
    R = xx_right[right_idx]
    FWHM = R - L

    return FWHM, SNR, maximum


def get_peak_edges(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 
                        '../Tables/Van__3x3_He3_and_MG_peak_edges.xlsx')
    matrix = pd.read_excel(path).values
    He3_and_MG_peak_edges_table = {}
    for row in matrix:
        He3_and_MG_peak_edges_table.update({row[0]: [row[1], row[2], row[3], 
                                            row[4]]})
    
    He3_and_MG_peak_edges = He3_and_MG_peak_edges_table[calibration]
    
    MG_left = int(He3_and_MG_peak_edges[0])
    MG_right = int(He3_and_MG_peak_edges[1])
    He3_left = int(He3_and_MG_peak_edges[2])
    He3_right = int(He3_and_MG_peak_edges[3])
    
    return MG_left, MG_right, He3_left, He3_right

def get_peak_edges_raw(calibration):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 
                        '../Tables/Van__3x3_He3_and_MG_peak_edges_raw_He3_data.xlsx')
    matrix = pd.read_excel(path).values
    He3_and_MG_peak_edges_table = {}
    for row in matrix:
        He3_and_MG_peak_edges_table.update({row[0]: [row[1], row[2], row[3], 
                                            row[4]]})
    
    He3_and_MG_peak_edges = He3_and_MG_peak_edges_table[calibration]
    MG_left = int(He3_and_MG_peak_edges[0])
    MG_right = int(He3_and_MG_peak_edges[1])
    He3_left = int(He3_and_MG_peak_edges[2])
    He3_right = int(He3_and_MG_peak_edges[3])
    return MG_left, MG_right, He3_left, He3_right



def import_He3_coordinates():
    dirname = os.path.dirname(__file__)
    he_folder = os.path.join(dirname, '../Tables/Helium3_coordinates/')
    az_path = he_folder + '145160_azimuthal.txt'
    dis_path = he_folder + '145160_distance.txt'
    pol_path = he_folder + '145160_polar.txt'

    az = np.loadtxt(az_path)
    dis = np.loadtxt(dis_path)
    pol = np.loadtxt(pol_path)

    x = dis*np.sin(pol * np.pi/180)*np.cos(az * np.pi/180)
    y = dis*np.sin(az * np.pi/180)*np.sin(pol * np.pi/180)
    z = dis*np.cos(pol * np.pi/180)
    return x, y, z


def plot_dE_background(E_i, calibration, measurement_time,
                       MG_norm, he_min, he_max, back_yes, tot_norm,
                       window, isPureAluminium=False, ToF_vec=None,
                       numberBins=390, isCLB=False):
    # Import background data
    dirname = os.path.dirname(__file__)
    clu_path = os.path.join(dirname, '../Clusters/MG/Background.h5')
    df = pd.read_hdf(clu_path, 'coincident_events')
    df = filter_ce_clusters(window, df)
    df = df[df.Time < 1.5e12]
    df = df[df.d != -1]
    if ToF_vec is not None:
        df = df[(df.ToF * 62.5e-9 * 1e6 >= ToF_vec[0]) &
                (df.ToF * 62.5e-9 * 1e6 <= ToF_vec[1])]


    modules_to_exclude = []
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        if E_i < 450:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])

    else:
        if E_i > 50:
            modules_to_exclude.append(4)
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])
        else:
            if isCLB:
                modules_to_exclude.extend([0, 1, 2, 6, 7, 8])
            if isPureAluminium:
                modules_to_exclude.extend([0, 1, 2, 3, 4, 5])

                
    print(modules_to_exclude)
    for bus in modules_to_exclude:
        df = df[df.Bus != bus]
    # Calculate background duration
    start_time = df.head(1)['Time'].values[0]
    end_time = df.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9
    # Calculate background
    t_off = get_t_off(calibration) * np.ones(df.shape[0])
    T_0 = get_T0(calibration, E_i) * np.ones(df.shape[0])
    frame_shift = get_frame_shift(E_i) * np.ones(df.shape[0])
    E_i = E_i * np.ones(df.shape[0])
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift)
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    # Calculate weights
    number_of_events = len(dE)
    events_per_s = number_of_events / duration
    events_s_norm = events_per_s / number_of_events
    weights = ((1/tot_norm) 
                * events_s_norm * measurement_time * np.ones(len(dE)))
    # Histogram background
    dE_bins = numberBins
    dE_range = [he_min, he_max]
    MG_dE_hist, MG_bins = np.histogram(dE, bins=dE_bins, range=dE_range,
                                       weights=weights)
    MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
  #  if back_yes:
  #      plt.plot(MG_bin_centers, MG_dE_hist, color='green', label='MG background', 
  #              zorder=5)
    return MG_dE_hist


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def get_charge_norm(calibration):
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, '../Tables/Charge_normalisation.xlsx')
    matrix = pd.read_excel(path).values
    charge_norm_table = {}
    for i, row in enumerate(matrix):
        charge_norm_table.update({row[0]: [row[2], row[5], row[7], row[8]]})
    
    no_glitch_time = charge_norm_table[calibration][0]
    total_time = charge_norm_table[calibration][1]
    MG_charge = charge_norm_table[calibration][2]
    He3_charge = charge_norm_table[calibration][3]
    charge_norm = ((MG_charge * (no_glitch_time/total_time)) / He3_charge)    
    return charge_norm

def create_ess_channel_to_coordinate_map(theta, offset):
    dirname = os.path.dirname(__file__)
    file_path = os.path.join(dirname, 
                             '../Tables/Coordinates_MG_SEQ_ESS.xlsx')
    matrix = pd.read_excel(file_path).values
    coordinates = matrix[1:801]
    ess_ch_to_coord = np.empty((3,124,80),dtype='object')
    coordinate = {'x': -1, 'y': -1, 'z': -1}
    axises =  ['x','y','z']
    
    c_offset = [[-1, 1, -1], [-1, -1, 1], [-1, 1, 1]]
    c_count = 0
    
    for i, row in enumerate(coordinates):
        grid_ch = i // 20 + 80
        for j, col in enumerate(row):
            module = j // 12
            layer = (j // 3) % 4
            wire_ch = (19 - (i % 20)) + (layer * 20)
            coordinate_count = j % 3
            coordinate[axises[coordinate_count]] = col
            if coordinate_count == 2:
                x = coordinate['x']
                y = coordinate['y']
                z = coordinate['z']
                # Convert from [mm] to [m]
                x = x/1000
                y = y/1000
                z = z/1000
                # Insert corners of vessels
                if wire_ch == 0 and grid_ch == 80 and module == 0:
                    ess_ch_to_coord[0][120][0] = {'x': offset['x'], 'y': offset['y'], 'z': offset['z']}
                if (  (wire_ch == 0 and grid_ch == 119 and module == 0)
                    | (wire_ch == 60 and grid_ch == 80 and module == 2)
                    | (wire_ch == 60 and grid_ch == 119 and module == 2)
                    ):
                    x_temp = x + 46.514/1000 * c_offset[c_count][0] + np.finfo(float).eps
                    y_temp = y + 37.912/1000 * c_offset[c_count][1] + np.finfo(float).eps
                    z_temp = z + 37.95/1000 * c_offset[c_count][2] + np.finfo(float).eps
                    z_temp, x_temp, y_temp = x_temp, y_temp, z_temp
                    x_temp, z_temp = get_new_x(x_temp, z_temp, theta), get_new_y(x_temp, z_temp, theta)
                    # Apply translation
                    x_temp += offset['x']
                    y_temp += offset['y']
                    z_temp += offset['z']
                    ess_ch_to_coord[0][121+c_count][0] = {'x': x_temp, 
                                                          'y': y_temp,
                                                          'z': z_temp}
                    c_count += 1
                
                # Shift to match internal and external coordinate system
                z, x, y = x, y, z
                # Apply rotation
                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                # Apply translation
                x += offset['x']
                y += offset['y']
                z += offset['z']

                ess_ch_to_coord[module, grid_ch, wire_ch] = {'x': x, 'y': y,
                                                             'z': z}
                coordinate = {'x': -1, 'y': -1, 'z': -1}

    return ess_ch_to_coord
    
def create_ill_channel_to_coordinate_map(theta, offset):
        
    WireSpacing  = 10     #  [mm]
    LayerSpacing = 23.5   #  [mm]
    GridSpacing  = 23.5   #  [mm]
    
    x_offset = 46.514     #  [mm]
    y_offset = 37.912     #  [mm]
    z_offset = 37.95      #  [mm]
    
    corners =   [[0, 80], [0, 119], [60, 80], [60, 119]]
    corner_offset = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1]]
    
    # Make for longer to include the for corners of the vessel
    ill_ch_to_coord = np.empty((3, 124, 80),dtype='object')
    for Bus in range(0,3):
        for GridChannel in range(80, 120):
            for WireChannel in range(0, 80):
                    x = (WireChannel % 20)*WireSpacing + x_offset
                    y = ((WireChannel // 20)*LayerSpacing 
                         + (Bus*4*LayerSpacing) + y_offset)
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
                                        
                    ill_ch_to_coord[Bus, GridChannel, WireChannel] = {'x': x,
                                                                      'y': y,
                                                                      'z': z}
        if Bus == 0:
            for i, corner in enumerate(corners[1:2]):
                WireChannel = corner[0]
                GridChannel = corner[1]
                x = (WireChannel % 20)*WireSpacing + x_offset
                y = ((WireChannel // 20)*LayerSpacing + (Bus*4*LayerSpacing) + y_offset)
                z = ((GridChannel-80)*GridSpacing) + z_offset 
                x += corner_offset[i+1][0] * x_offset
                y += corner_offset[i+1][1] * y_offset
                z += corner_offset[i+1][2] * z_offset
                x = x/1000 + np.finfo(float).eps
                y = y/1000 + np.finfo(float).eps
                z = z/1000 + np.finfo(float).eps
                z, x, y = x, y, z

                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                x += offset['x']
                y += offset['y']
                z += offset['z']
                ill_ch_to_coord[0, 121+i, 0] = {'x': x, 'y': y, 'z': z}
        
            ill_ch_to_coord[Bus, 120, 0] = {'x': offset['x'], 'y': offset['y'], 'z': offset['z']}

            
        if Bus == 2:
            for i, corner in enumerate(corners[2:]):
                WireChannel = corner[0]
                GridChannel = corner[1]
                x = (WireChannel % 20)*WireSpacing + x_offset
                y = ((WireChannel // 20)*LayerSpacing + (Bus*4*LayerSpacing) + y_offset)
                z = ((GridChannel-80)*GridSpacing) + z_offset 
                x += corner_offset[i+2][0] * x_offset
                y += corner_offset[i+2][1] * y_offset
                z += corner_offset[i+2][2] * z_offset
                x = x/1000
                y = y/1000
                z = z/1000
                z, x, y = x, y, z
                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                x += offset['x']
                y += offset['y']
                z += offset['z']
                ill_ch_to_coord[0, 122+i, 0] = {'x': x, 'y': y, 'z': z}
            
    return ill_ch_to_coord


def get_new_x(x, y, theta):
    return np.cos(np.arctan(y/x)+theta)*np.sqrt(x ** 2 + y ** 2)
    

def get_new_y(x, y, theta):
    return np.sin(np.arctan(y/x)+theta)*np.sqrt(x ** 2 + y ** 2)


def import_He3_coordinates_NEW():
    dirname = os.path.dirname(__file__)
    he_folder = os.path.join(dirname, '../Tables/Helium3_coordinates/')
    az_path = he_folder + '145160_azimuthal.txt'
    dis_path = he_folder + '145160_distance.txt'
    pol_path = he_folder + '145160_polar.txt'

    az = np.loadtxt(az_path)
    dis = np.loadtxt(dis_path)
    pol = np.loadtxt(pol_path)
    distance = np.ones(len(dis) * 2)
    count = 0
    for d in dis:
        for i in range(2):
            distance[count] = d
            count += 1

    x = dis*np.sin(pol * np.pi/180)*np.cos(az * np.pi/180)
    y = dis*np.sin(az * np.pi/180)*np.sin(pol * np.pi/180)
    z = dis*np.cos(pol * np.pi/180)
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    return x, y, z, distance


def import_efficiency_correction():
    def angstrom_to_meV(a):
        return (9.045 ** 2) / (a ** 2)
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../Tables/pressure_wavelengthdat.txt')
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
    path = os.path.join(dirname, '../Tables/MG_SEQ_theoretical_eff.txt')
    eff_a_meV_table = np.loadtxt(path, delimiter=',')
    size = len(eff_a_meV_table)
    meV_vs_eff = np.array([np.zeros(size), np.zeros(size)]) 
    for i, row in enumerate(eff_a_meV_table):
        meV_vs_eff[0][i] = row[0]
        meV_vs_eff[1][i] = row[2]
    return meV_vs_eff


def import_He3_coordinates_raw():
    # Declare parameters
    dirname = os.path.dirname(__file__)
    he_folder = os.path.join(dirname, '../Tables/Helium3_coordinates/')
    az_path = he_folder + '145160_azimuthal.txt'
    dis_path = he_folder + '145160_distance.txt'
    pol_path = he_folder + '145160_polar.txt'
    # Import data
    az = np.loadtxt(az_path)
    dis = np.loadtxt(dis_path)
    pol = np.loadtxt(pol_path)
    distance = np.ones(len(dis) * 2)
    azimuthal = np.ones(len(dis) * 2)
    polar = np.ones(len(dis) * 2)
    count = 0
    # Insert into vector twice as long
    for i in range(len(dis)):
        for j in range(2):
            distance[count] = dis[i]
            azimuthal[count] = az[i]
            polar[count] = pol[i]
            count += 1
    # Calculate carthesian coordinates
    x = distance*np.sin(polar * np.pi/180)*np.cos(azimuthal * np.pi/180)
    y = distance*np.sin(azimuthal * np.pi/180)*np.sin(polar * np.pi/180)
    z = distance*np.cos(polar * np.pi/180)
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x, y, z, d, azimuthal, polar


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
                     (ce.gCh <= window.lowerStartGrid.value() + 80 - 1)) |
                     ((ce.gCh <= window.grid_max.value() + 80 - 1) &
                     (ce.gCh >= window.upperStartGrid.value() + 80 - 1))) &
                     (((ce.wCh >= window.wire_min.value() - 1) &
                      (ce.wCh <= window.wire_max.value() - 1)) 
                        |
                      ((ce.wCh >= window.wire_min.value() + 20 - 1) &
                      (ce.wCh <= window.wire_max.value() + 20 - 1))
                        |
                      ((ce.wCh >= window.wire_min.value() + 40 - 1) &
                      (ce.wCh <= window.wire_max.value() + 40 - 1)) 
                        |
                      ((ce.wCh >= window.wire_min.value() + 60 - 1) &
                      (ce.wCh <= window.wire_max.value() + 60 - 1))
                      )
                     ]
    return ce_filtered


def get_charge_solid_norm(calibration, E_i, isCLB, isPureAluminium):
    # Declare solid angle for Helium-3 and Multi-Grid
    He3_solid_angle = 0.7498023343722737
    MG_solid_angle = 0
    MG_solid_angle_tot = 0.01660177142644554
    MG_missing_solid_angle_1 = 0.0013837948633069277
    MG_missing_solid_angle_2 = 0.0018453301781999457
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        print(calibration)
        if E_i < 450:
            print('Under 450')
            MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1
                              - MG_missing_solid_angle_2)
            if isCLB:
                MG_solid_angle = (0.005518217498193907 - 0.00046026495055297194
                                  - 0.0018453301781999457)

        else:
            print('Over 450')
            MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1)
            if isCLB:
                MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194

    else:
        print(calibration)
        if E_i > 50:
            print('Over 50')
            MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1
                              - MG_missing_solid_angle_2)
            if isCLB:
                MG_solid_angle = (0.005518217498193907 - 0.00046026495055297194
                                  - 0.0018453301781999457)
        else:
            print('Under 50')
            MG_solid_angle = (MG_solid_angle_tot - MG_missing_solid_angle_1)
            if isCLB:
                MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194
    
    if isPureAluminium:
        print('Calibration')
        print('I am in pure aluminum')
        MG_solid_angle = 0.005518217498193907 - 0.00046026495055297194
    


    # Get charge normalization
    charge_norm = get_charge_norm(calibration)
    # Calculate total normalization
    charge_solid_norm = ((MG_solid_angle/He3_solid_angle)*charge_norm)
    return charge_solid_norm


def get_area_time_norm(calibration, E_i, isCLB, isPureAluminium,
                       measurement_time, He3_measurement_time):
    # He3 area 
    He3_area = 23.457226948780608
    # Multi-Grid area
    voxel_area = 0.0005434375
    MG_area_wire_row = voxel_area * 35
    MG_area_module = MG_area_wire_row * 4
    MG_area_detector = MG_area_module * 3
    if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
        if E_i < 450:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row) - MG_area_module) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row - MG_area_module) * (34/40)

        else:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row)) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row) * (34/40)

    else:
        if E_i > 50:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row) - MG_area_module) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row - MG_area_module) * (34/40)
        else:
            MG_area = ((MG_area_detector * 3) - (3 * MG_area_wire_row)) * (34/40)
            if isCLB:
                MG_area = (MG_area_detector - MG_area_wire_row) * (34/40)
    
    if isPureAluminium:
        MG_area = MG_area_detector - MG_area_wire_row - voxel_area * 35 - (4 * 4 * voxel_area)  # Last part is because we remove the middle three grids and top grid

    area_frac = MG_area/He3_area
    time_frac = measurement_time/He3_measurement_time
    area_time_norm = area_frac * time_frac
    return area_time_norm







    