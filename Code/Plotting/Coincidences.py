# =======  LIBRARIES  ======= #
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from HelperFunctions import get_duration, set_figure_properties, stylize


# =============================================================================
# Coincidence Histogram (2D)
# =============================================================================


def Coincidences_2D_plot(ce, data_sets, module_order, number_of_detectors):
    def plot_2D_bus(fig, sub_title, ce, vmin, vmax, duration):
        plt.hist2d(ce.wCh, ce.gCh, bins=[80, 40],
                   range=[[-0.5, 79.5], [79.5, 119.5]],
                   vmin=vmin, vmax=vmax, norm=LogNorm(), cmap='jet')
        xlabel = 'Wire [Channel number]'
        ylabel = 'Grid [Channel number]'
        fig = stylize(fig, xlabel, ylabel, title=sub_title, colorbar=True)

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
    fig.show()


# =============================================================================
# Coincidence Histogram (3D)
# =============================================================================