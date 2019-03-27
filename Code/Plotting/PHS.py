# =======  LIBRARIES  ======= #
import matplotlib.pyplot as plt
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
    vmax = events.shape[0] // 1000
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
    vmin = 1
    vmax = ce.shape[0] // 1000
    for i, bus in enumerate(module_order):
        events_bus = ce[ce.Bus == bus]
        sub_title = 'Bus %d\n(%d events)' % (bus, events_bus.shape[0])
        plt.subplot(3, 3, i+1)
        fig = charge_scatter(fig, events_bus, sub_title, bus, vmin, vmax)
    fig = set_figure_properties(fig, title, height, width)
    return fig
