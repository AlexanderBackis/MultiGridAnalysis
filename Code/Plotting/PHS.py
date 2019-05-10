# =======  LIBRARIES  ======= #
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import LogNorm
from Plotting.HelperFunctions import (stylize, set_figure_properties,
                                      filter_ce_clusters)

# ============================================================================
# PHS (1D)
# ============================================================================


def PHS_1D_plot(events, data_sets, window):
    # Declare parameters
    number_bins = int(window.phsBins.text())
    Channel = window.Channel.value()
    Bus = window.Module.value()
    # Plot
    fig = plt.figure()
    title = ('PHS (1D) - Channel: %d, Bus: %d\nData set(s): %s'
             % (Channel, Bus, data_sets))
    xlabel = 'Counts'
    ylabel = 'Collected charge [ADC channels]'
    fig = stylize(fig, xlabel, ylabel, title, yscale='log', grid=True)
    plt.hist(events[(events.Channel == Channel) & (events.Bus == Bus)].ADC,
             bins=number_bins, range=[0, 4400], histtype='step',
             color='black', zorder=5)
    return fig


# =============================================================================
# PHS (2D)
# =============================================================================


def PHS_2D_plot(events, data_sets, module_order):
    def PHS_2D_plot_bus(fig, events, sub_title, vmin, vmax):
        xlabel = 'Channel'
        ylabel = 'Charge [ADC channels]'
        bins = [120, 120]
        fig = stylize(fig, xlabel, ylabel, title=sub_title, colorbar=True)
        plt.hist2d(events.Channel, events.ADC, bins=bins, norm=LogNorm(),
                   range=[[-0.5, 119.5], [0, 4400]], vmin=vmin, vmax=vmax,
                   cmap='jet')
        plt.colorbar()
        return fig

    fig = plt.figure()
    title = 'PHS (2D)\nData set(s): %s' % data_sets
    height = 12
    width = 14
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    for i, bus in enumerate(module_order):
        events_bus = events[events.Bus == bus]
        wire_events = events_bus[events_bus.Channel < 80].shape[0]
        grid_events = events_bus[events_bus.Channel >= 80].shape[0]
        plt.subplot(3, 3, i+1)
        sub_title = 'Bus: %d, events: %d' % (bus, events_bus.shape[0])
        sub_title += ('\nWire events: %d, Grid events: %d'
                      % (wire_events, grid_events)
                      )
        fig = PHS_2D_plot_bus(fig, events_bus, sub_title, vmin, vmax)
    fig = set_figure_properties(fig, title, height, width)
    return fig

# =============================================================================
# PHS (Wires vs Grids)
# =============================================================================


def PHS_wires_vs_grids_plot(ce, data_sets, module_order, window):
    def charge_scatter(fig, ce, sub_title, bus, vmin, vmax):
        xlabel = 'Collected charge wires [ADC channels]'
        ylabel = 'Collected charge grids [ADC channels]'
        bins = [200, 200]
        ADC_range = [[0, 5000], [0, 5000]]
        fig = stylize(fig, xlabel, ylabel, title=sub_title, colorbar=True)
        plt.hist2d(ce.wADC, ce.gADC, bins=bins,
                   norm=LogNorm(), range=ADC_range,
                   vmin=vmin, vmax=vmax, cmap='jet')
        plt.colorbar()
        return fig
    # Intial filter
    ce = filter_ce_clusters(window, ce)
    # Plot data
    fig = plt.figure()
    title = 'PHS (Wires vs Grids)\nData set(s): %s' % data_sets
    height = 12
    width = 14
    if ce.shape[0] != 0:
        vmin = 1
        vmax = ce.shape[0] // 4500 + 1000
    else:
        vmin = 1
        vmax = 1
    for i, bus in enumerate(module_order):
        events_bus = ce[ce.Bus == bus]
        sub_title = 'Bus %d\n(%d events)' % (bus, events_bus.shape[0])
        plt.subplot(3, 3, i+1)
        fig = charge_scatter(fig, events_bus, sub_title, bus, vmin, vmax)
    fig = set_figure_properties(fig, title, height, width)
    return fig


# ============================================================================
# PHS (Wires and Grids)
# ============================================================================

def PHS_wires_and_grids_plot(ce, data_sets, window):
    # Declare parameters
    number_bins = int(window.phsBins.text())
    # Initial filter
    ce = filter_ce_clusters(window, ce)
    # Plot data
    fig = plt.figure()
    title = '%s' % (data_sets)
    plt.subplot(1, 2, 1)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.hist(ce.wADC, bins=number_bins, histtype='step', color='black')
    plt.title('Wires (wM: %d -> %d)' % (window.wM_min.value(),
                                        window.wM_max.value()))
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts')
    plt.subplot(1, 2, 2)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.hist(ce.gADC, bins=number_bins, histtype='step', color='black')
    plt.title('Grids (gM: %d -> %d)' % (window.gM_min.value(),
                                        window.gM_max.value()))
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts')
    fig.suptitle(title, x=0.5, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# PHS (Iterate through all energies)
# =============================================================================

def plot_all_PHS():
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
    # Declare parameters
    module_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    number_of_detectors = 3
    window = None
    # Iterate through all energies and save data
    for input_path in input_paths:
        # Import data
        e = pd.read_hdf(input_path, 'events')
        ce = pd.read_hdf(input_path, 'coincident_events')
        calibration = pd.read_hdf(input_path,
                                  'calibration')['calibration'].iloc[0]
        # Produce histograms
        fig1 = PHS_1D_plot(e, calibration, window)
        fig2 = PHS_2D_plot(e, calibration, module_order, number_of_detectors)
        fig3 = PHS_wires_vs_grids_plot(ce, calibration, module_order,
                                       number_of_detectors)
        # Define output paths
        output_file_1 = '../../Results/PHS/PHS_1D/%s.pdf' % calibration
        output_file_2 = '../../Results/PHS/PHS_2D/%s.pdf' % calibration
        output_file_3 = '../../Results/PHS/PHS_wires_vs_grids/%s.pdf' % calibration
        output_path_1 = os.path.join(dir_name, output_file_1)
        output_path_2 = os.path.join(dir_name, output_file_2)
        output_path_3 = os.path.join(dir_name, output_file_3)
        # Save histograms
        fig1.savefig(output_path_1, bbox_inches='tight')
        fig2.savefig(output_path_2, bbox_inches='tight')
        fig3.savefig(output_path_3, bbox_inches='tight')










