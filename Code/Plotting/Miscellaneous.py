import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import os
import h5py
import imageio
import shutil
import scipy
import plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from Plotting.HelperFunctions import (stylize, filter_ce_clusters,
                                      get_raw_He3_dE, get_calibration,
                                      Gaussian_fit, find_nearest,
                                      detector_filter, mkdir_p, load_He3_h5,
                                      get_frame_shift, find_He3_measurement_id,
                                      export_ToF_histograms_to_text,
                                      get_He3_duration,
                                      get_He3_tubes_area_and_solid_angle,
                                      get_multi_grid_area_and_solid_angle,
                                      get_ToF_intervals,
                                      import_MG_coincident_events,
                                      import_MG_Ei,
                                      import_MG_calibration,
                                      import_MG_measurement_time,
                                      get_chopper_setting,
                                      get_MG_dE,
                                      get_all_calibrations,
                                      get_all_energies,
                                      get_detector_mappings,
                                      flip_bus,
                                      flip_wire
                                      )

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
# Multiplicity
# =============================================================================

def Multiplicity_plot(df, data_sets, module_order, number_of_detectors,
                      window):
    # Initial filter
    df = filter_ce_clusters(window, df)
    # Declare parameters
    name = 'Multiplicity\nData set(s): ' + str(data_sets)
    fig = plt.figure()
    figwidth = 14
    figheight = 12
    vmin = 1
    vmax = df.shape[0] // (len(module_order) * 1)
    # Prepare figure
    fig.suptitle(name, x=0.5, y=1.08)
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    # Iterate through all buses
    for loc, bus in enumerate(module_order):
        plt.subplot(3, 3, loc+1)
        df_bus = df[df.Bus == bus]
        plot_multiplicity_bus(df_bus, bus, fig, vmin, vmax)

    plt.tight_layout()
    fig.show()


def plot_multiplicity_bus(df, bus, fig, vmin, vmax):
    # Declare parameters
    m_range = [0, 5, 0, 5]
    # Plot data
    hist, xbins, ybins, im = plt.hist2d(df.wM, df.gM,
                                        bins=[m_range[1]-m_range[0]+1,
                                              m_range[3]-m_range[2]+1],
                                        range=[[m_range[0], m_range[1]+1],
                                               [m_range[2], m_range[3]+1]],
                                        norm=LogNorm(),
                                        vmin=vmin,
                                        vmax=vmax,
                                        cmap='jet')
    # Iterate through all squares and write percentages
    tot = df.shape[0]
    font_size = 12
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.1f%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    # Set ticks on axis
    ticks_x = np.arange(m_range[0], m_range[1]+1, 1)
    locs_x = np.arange(m_range[0] + 0.5, m_range[1]+1.5, 1)
    ticks_y = np.arange(m_range[2], m_range[3]+1, 1)
    locs_y = np.arange(m_range[2] + 0.5, m_range[3]+1.5, 1)
    plt.xticks(locs_x, ticks_x)
    plt.yticks(locs_y, ticks_y)
    plt.xlabel("Wire Multiplicity")
    plt.ylabel("Grid Multiplicity")
    plt.colorbar()
    plt.tight_layout()
    name = 'Bus ' + str(bus) + '\n(' + str(df.shape[0]) + ' events)'
    plt.title(name)


# =============================================================================
# Depth variation of counts
# =============================================================================

def calculate_depth_variation(df, window):
    def exponential(A, B, C, x):
        return A*np.exp(B*x)+C

    # Initial filter
    df = filter_ce_clusters(window, df)
    # Declare parameters
    data = {'rows': [], 'counts': []}
    min_wire = window.wire_min.value()
    max_wire = window.wire_max.value()
    rows = range(min_wire, max_wire+1)
    # Iterate through data
    for row in rows:
        df_temp = df[(((df.wCh >= row - 1) & (df.wCh <= row - 1)) |
                      ((df.wCh >= row + 20 - 1) & (df.wCh <= row + 20 - 1)) |
                      ((df.wCh >= row + 40 - 1) & (df.wCh <= row + 40 - 1)) |
                      ((df.wCh >= row + 60 - 1) & (df.wCh <= row + 60 - 1)))
                     ]
        counts = df_temp.shape[0]
        # Insert values in 'data'-vector
        data['rows'].append(row)
        data['counts'].append(counts)
    # Fit data to exponential
    popt, pcov = scipy.optimize.curve_fit(lambda t, A, B, C:
                                          A*np.exp(B*t)+C,
                                          data['rows'], data['counts'])
    perr = np.sqrt(np.diag(pcov))
    # Plot data
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(data['rows'], data['counts'], '.-', color='black', zorder=2,
             label='Data')
    plt.plot(np.linspace(min_wire, max_wire, 100),
             popt[0] * (np.exp(popt[1]*np.linspace(1, 20, 100))-20) + popt[2],
             color='red',
             label='Fitted function: $f(x)$=A$e^{Bx}$+C', zorder=2)
    plt.xlabel('Wire row')
    plt.ylabel('Counts')
    plt.title('Depth variation of counts\n' +
              '(effect of back rows in Pure Aluminum)')
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
    plt.legend()
    # Calculate chi^2
    observations = np.array(data['counts'])
    deviations = np.sqrt(np.array(data['counts']))
    expectations = exponential(popt[0], popt[1], popt[2],
                               np.array(data['rows']))
    chi2 = sum(((observations-expectations)/deviations) ** 2)
    red_chi2 = chi2 / (max_wire - min_wire + 1 - 3)
    text_string = (('Fit parameters:\nA = %.1f ± %.1f \n'
                   + 'B = %.1f ± %.1f\nC = %.1f ± %.1f\n'
                   + '$\chi^2$ = %.1f')
                   % (popt[0], perr[0], popt[1], perr[1], popt[2], perr[2],
                      red_chi2))
    plt.text(1, data['counts'][4]*7, text_string, ha='left', va='center',
             bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 10}, fontsize=10,
             zorder=50)
    return fig

# =============================================================================
# Calculate signal variation as a function of ADC-threshold
# =============================================================================


def signal_dependence_on_ADC_threshold(df, Ei, calibration, window):
    # Declare parameters
    bins_full = 1000
    bins_part = 500
    step = 20
    df = filter_ce_clusters(window, df)
    thresholds = np.arange(0, 1000 + step, step)
    number_thresholds = len(thresholds)
    ## Plot limits
    maximum_full = 1
    minimum_full = 0
    maximum_part = 1
    minimum_part = 0
    ## Bins with data
    gamma_bins = [320, 328]
    thermal_bins = [400, 550]
    fast_bins = [328, 343]
    back_bins = [800, 900]
    gamma_bins_part = [150, 225]
    fast_bins_part = [225, 350]
    # Declare data folder
    fast_neutrons = np.zeros(number_thresholds)
    thermal_neutrons = np.zeros(number_thresholds)
    gammas = np.zeros(number_thresholds)
    background = np.zeros(number_thresholds)
    # Create output folder
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../../Results/temp_folder_ADC/')
    mkdir_p(temp_folder)
    # Iterate through data
    for run in [1, 2]:
        for i, threshold in enumerate(thresholds):
            df_temp = df[(df.wADC >= threshold) & (df.gADC >= threshold)]
            # Define figure and axises
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 5))
            left, bottom, width, height = [0.335, 0.6, 0.13, 0.25]
            ax2 = fig.add_axes([left, bottom, width, height])
            # Plot data
            ## First plot, first axis
            hist_full, bins_full = np.histogram(df_temp.ToF*62.5e-9*1e6,
                                                bins=bins_full,
                                                range=[0, 16667])
            bin_centers_full = 0.5 * (bins_full[1:] + bins_full[:-1])
            ### Plot full data
            ax1.plot(bin_centers_full, hist_full, color='black',
                     label=None)
            ax1.plot(bin_centers_full[thermal_bins[0]:thermal_bins[1]],
                     hist_full[thermal_bins[0]:thermal_bins[1]],
                     color='orange', label='Thermal')
            ax1.plot(bin_centers_full[fast_bins[0]:fast_bins[1]],
                     hist_full[fast_bins[0]:fast_bins[1]],
                     color='blue', label='Fast')
            ax1.plot(bin_centers_full[gamma_bins[0]:gamma_bins[1]],
                     hist_full[gamma_bins[0]:gamma_bins[1]],
                     color='red', label='Gamma')
            ax1.plot(bin_centers_full[back_bins[0]:back_bins[1]],
                     hist_full[back_bins[0]:back_bins[1]],
                     color='green', label='Background')
            ax1.set_ylim(minimum_full*0.8, maximum_full*1.5)
            ax1.set_yscale('log')
            ax1.set_xlabel('ToF [µs]')
            ax1.set_ylabel('Counts')
            ax1.legend(loc=2)
            ax1.set_title('ToF histogram: US - 70 meV')
            ## First plot, second axis
            hist_part, bins_part = np.histogram(df_temp.ToF*62.5e-9*1e6,
                                                bins=bins_part,
                                                range=[5250, 5750])
            bin_centers_part = 0.5 * (bins_part[1:] + bins_part[:-1])
            ax2.plot(bin_centers_part, hist_part, color='black')
            ax2.plot(bin_centers_part[gamma_bins_part[0]:gamma_bins_part[1]],
                     hist_part[gamma_bins_part[0]:gamma_bins_part[1]],
                     color='red')
            ax2.plot(bin_centers_part[fast_bins_part[0]:fast_bins_part[1]],
                     hist_part[fast_bins_part[0]:fast_bins_part[1]],
                     color='blue')
            ax2.set_yscale('log')
            ax2.set_ylim(minimum_part*0.8, maximum_part*1.5)
            ax2.set_xlabel('ToF [µs]')
            ax2.set_ylabel('Counts')
            if run == 1:
                if i == 1:
                    maximum_full = max(hist_full)
                    maximum_part = max(hist_part)
                if i == len(thresholds) - 1:
                    minimum_full = min(hist_full)
                    minimum_part = min(hist_part)
                # Save data
                fast_data = hist_part[fast_bins_part[0]:fast_bins_part[1]]
                fast_duration = (bin_centers_part[fast_bins_part[1]]
                                 - bin_centers_part[fast_bins_part[0]])
                thermal_data = hist_full[thermal_bins[0]:thermal_bins[1]]
                thermal_duration = (bin_centers_full[thermal_bins[1]]
                                    - bin_centers_full[thermal_bins[0]])
                gamma_data = hist_part[gamma_bins_part[0]:gamma_bins_part[1]]
                gamma_duration = (bin_centers_part[gamma_bins_part[1]]
                                  - bin_centers_part[gamma_bins_part[0]])
                background_data = hist_full[back_bins[0]:back_bins[1]]
                background_duration = (bin_centers_full[back_bins[1]]
                                       - bin_centers_full[back_bins[0]])
                fast_neutrons[i] = sum(fast_data)/fast_duration
                thermal_neutrons[i] = sum(thermal_data)/thermal_duration
                gammas[i] = sum(gamma_data)/gamma_duration
                background[i] = sum(background_data)/background_duration
                plt.close()
            else:
                ## Second plot
                ax3.plot(thresholds, thermal_neutrons, '.-', color='orange',
                         label='Thermal')
                ax3.plot(thresholds, fast_neutrons, '.-', color='blue',
                         label='Fast')
                ax3.plot(thresholds, gammas, '.-', color='red', label='Gamma')
                ax3.plot(thresholds, background, '.-', color='green',
                         label='Background')
                ax3.axvline(x=threshold, color='black')
                ax3.axvline(x=500, color='grey', linestyle='--')
                ax3.set_xlabel('Threshold [ADC channels]')
                ax3.set_ylabel('Counts/µs')
                ax3.grid(True, which='major', linestyle='--', zorder=0)
                ax3.grid(True, which='minor', linestyle='--', zorder=0)
                ax3.legend(loc=2)
                ax3.set_yscale('log')
                # Save data
                output_path = temp_folder + str(threshold) + '.png'
                fig.savefig(output_path, bbox_inches='tight')
    # Save animation
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files
             if file[-9:] != '.DS_Store' and file != '.gitignore']
    animation_path = os.path.join(dir_name,
                                  '../../Results/Animations/ADC_sweep.gif')
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(animation_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)


# =============================================================================
# Uncertainty calculation - single calibration
# =============================================================================

def calculate_uncertainty(calibration):
    def get_mean_and_standard_deviation(values):
        N = len(values)
        average = sum(values)/N
        STD = np.sqrt((1/N)*sum((values - average) ** 2))
        return average, STD
    # Declare parameters
    dir_name = os.path.dirname(__file__)
    m_id = str(find_He3_measurement_id(calibration))
    pixels_per_three_banks = 64*8*3
    total_number_of_pixels = 0
    total_events = 0
    three_banks_events = np.zeros(29)  # 29 is the total number of three banks
    three_bank_ids = np.zeros(29)  # 29 is the total number of three banks
    banks_to_skip = np.array([75, 76, 97, 98, 99, 100,
                              101, 102, 103, 114, 115, 119])
    # Import He3-data
    path = os.path.join(dir_name,
                        '../../Archive/SEQ_raw/SEQ_' + m_id + '.nxs.h5')
    data = h5py.File(path, 'r')
    # Iterate through data, comparing groups of 3 banks with full data
    number_exceptions = 0
    position = 0
    for i in range(40, 150, 3):
        banks_to_sum = np.arange(i, i+3, 1)
        in_exceptions = any(bank_id in banks_to_skip
                            for bank_id in banks_to_sum)
        if not in_exceptions:
            three_bank_ids[position] = i
            for bank_id in banks_to_sum:
                bank = 'bank%d_events' % bank_id
                events = data['entry'][bank]['event_time_offset'].value
                number_of_events = len(events)
                # Insert values in data-vector
                total_events += number_of_events
                three_banks_events[position] += number_of_events
                total_number_of_pixels += 64*8
            position += 1
        else:
            number_exceptions += 1
    # Convert lists to numpy arrays
    three_banks_events = np.array(three_banks_events)
    # Normalize based on total number of pixels
    total_events_normalized = total_events/total_number_of_pixels
    three_banks_events_normalized = three_banks_events/pixels_per_three_banks
    ratios = three_banks_events_normalized/total_events_normalized
    # Get mean and standard deviation from mean
    mean, STD = get_mean_and_standard_deviation(ratios)
    # Visualize difference between different pieces of area in the He3-array
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.plot(three_bank_ids, ratios, 'o', color='black', zorder=5)
    plt.xlabel('Equivalent Multi-Grid area fragment')
    plt.ylabel('Ratio')
    plt.title('Uncertainty in isotropic distribution of Vanadium scattering\n'
              + 'Mean: %.2f, STD: %.2f' % (mean, STD))
    return fig, mean, STD

# =============================================================================
# Calculate background variation as a function of ADC-threshold
# =============================================================================


def background_dependence_on_ADC_threshold(df, data_sets, window):
    # Initial filter of clusters
    df = filter_ce_clusters(window, df)
    df = df[(df.wCh != -1) & (df.gCh != -1)]
    if data_sets == "['mvmelst_039.mvmelst']":
        df = df[df.Time < 1.5e12]
    # Declare parameters
    step = 20
    thresholds = np.arange(0, 1500 + step, step)
    number_thresholds = len(thresholds)
    # Declare data folder
    data = {'top': [thresholds, np.zeros(number_thresholds)],
            'bottom': [thresholds, np.zeros(number_thresholds)],
            'middle': [thresholds, np.zeros(number_thresholds)],
            'back': [thresholds, np.zeros(number_thresholds)],
            'reference': [thresholds, np.zeros(number_thresholds)]
            }
    # Create output folder
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../../Results/temp_folder_ADC/')
    mkdir_p(temp_folder)
    # Iterate through data
    for run in [1, 2]:
        for i, threshold in enumerate(thresholds):
            df_temp = df[(df.wADC >= threshold) & (df.gADC >= threshold)]
            if run == 1:
                counts_top = df_temp[(df_temp.Bus >= 6) &
                                     (df_temp.Bus <= 8) &
                                     (df_temp.gCh == 119)
                                     ].shape[0]/(20*11)
                counts_bottom = df_temp[(df_temp.Bus >= 6) &
                                        (df_temp.Bus <= 8) &
                                        (df_temp.gCh == 80)
                                        ].shape[0]/(20*11)
                counts_back = df_temp[(df_temp.Bus >= 6) &
                                      (df_temp.Bus <= 8) &
                                      ((df_temp.wCh == 19) |
                                       (df_temp.wCh == 39) |
                                       (df_temp.wCh == 59) |
                                       (df_temp.wCh == 79)
                                       )
                                      ].shape[0]/(11*40)
                counts_middle = df_temp[((df_temp.gCh >= 95)
                                         & (df_temp.gCh <= 105)
                                         ) &
                                        (((df_temp.wCh >= 0) &
                                          (df_temp.wCh <= 9)) |
                                         ((df_temp.wCh >= 20) &
                                          (df_temp.wCh <= 29)) |
                                         ((df_temp.wCh >= 40) &
                                          (df_temp.wCh <= 49)) |
                                         ((df_temp.wCh >= 60) &
                                          (df_temp.wCh <= 69))
                                         ) &
                                        (df_temp.Bus >= 7)
                                        ].shape[0]/(10*10*7)
                counts_reference = df_temp[((df_temp.gCh >= 112)
                                            & (df_temp.gCh <= 117)
                                            ) &
                                           (((df_temp.wCh >= 0) &
                                             (df_temp.wCh <= 9)) |
                                            ((df_temp.wCh >= 20) &
                                             (df_temp.wCh <= 29)) |
                                            ((df_temp.wCh >= 40) &
                                             (df_temp.wCh <= 49)) |
                                            ((df_temp.wCh >= 60) &
                                             (df_temp.wCh <= 69))
                                            ) &
                                           (df_temp.Bus == 6)
                                           ].shape[0]/(6*10*4)
                data['top'][1][i] = counts_top
                data['bottom'][1][i] = counts_bottom
                data['middle'][1][i] = counts_middle
                data['back'][1][i] = counts_back
                data['reference'][1][i] = counts_reference
            else:
                path = temp_folder + str(threshold) + '.png'
                hist_3D_for_ADC_threshold(df_temp, data, threshold, path)
    # Save animation
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files
             if file[-9:] != '.DS_Store' and file != '.gitignore']
    animation_path = os.path.join(dir_name,
                                  '../../Results/Animations/ADC_sweep_3D.gif')
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(animation_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)


def hist_3D_for_ADC_threshold(df, data, threshold, path):
    # Get maximum and minimum in data
    full_data = np.concatenate((data['top'][1],
                                data['bottom'][1],
                                data['middle'][1],
                                data['back'][1],
                                data['reference'][1]),
                               axis=None)
    maximum = max(full_data)
    minimum = min(full_data)
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Initiate 'voxel_id -> (x, y, z)'-mapping
    detector_vec = get_detector_mappings()
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wCh', 'gCh', 'Bus']].values,
                              bins=(80, 40, 9),
                              range=((0, 80), (80, 120), (0, 9))
                              )
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 9):
                over_min = H[wCh, gCh-80, bus] > min_count
                under_max = H[wCh, gCh-80, bus] <= max_count
                if over_min and under_max:
                    detector = detector_vec[bus//3]
                    coord = detector[flip_bus(bus % 3), gCh, flip_wire(wCh)]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    loc += 1
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
                                           cmin=0,
                                           cmax=2.5
                                           ),
                               showlegend=False,
                               scene='scene1'
                               )
    MG_3D_trace_2 = go.Scatter3d(x=hist[2],
                                 y=hist[0],
                                 z=hist[1],
                                 mode='markers',
                                 marker=dict(size=5,
                                             color=np.log10(hist[3]),
                                             colorscale='Jet',
                                             opacity=1,
                                             cmin=0,
                                             cmax=2.5
                                             ),
                                 showlegend=False,
                                 scene='scene2'
                                 )
    # Produce 2D-plot with data about counts in high and low region
    top_trace = go.Scatter(x=data['top'][0],
                           y=data['top'][1],
                           mode='lines+markers',
                           marker=dict(
                                       #color='rgb(255, 0, 0)',
                                       size=6),
                           #fillcolor='rgba(255, 0, 0, .5)',
                           name='Top',
                           #line=dict(width=2)
                           )
    bottom_trace = go.Scatter(x=data['bottom'][0],
                              y=data['bottom'][1],
                              marker=dict(
                                          #color='rgb(0, 0, 255)',
                                          size=6),
                              mode='lines+markers',
                              #fillcolor='rgba(0, 0, 255, .5)',
                              #line=dict(width=2),
                              name='Bottom',
                              )
    middle_trace = go.Scatter(x=data['middle'][0],
                              y=data['middle'][1],
                              marker=dict(
                                          #color='rgb(0, 0, 255)',
                                          size=6),
                              mode='lines+markers',
                              #fillcolor='rgba(0, 0, 255, .5)',
                              #line=dict(width=2),
                              name='Middle',
                              )
    back_trace = go.Scatter(x=data['back'][0],
                            y=data['back'][1],
                            marker=dict(
                                        #color='rgb(0, 0, 255)',
                                        size=6),
                            mode='lines+markers',
                            #fillcolor='rgba(0, 0, 255, .5)',
                            #line=dict(width=2),
                            name='Back',
                            )
    reference_trace = go.Scatter(x=data['reference'][0],
                                 y=data['reference'][1],
                                 marker=dict(
                                             #color='rgb(0, 0, 255)',
                                             size=6),
                                 mode='lines+markers',
                                 #fillcolor='rgba(0, 0, 255, .5)',
                                 #line=dict(width=2),
                                 name='Reference',
                                 )
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=3,
                                 specs=[[{'is_3d': False},
                                         {'is_3d': True},
                                         {'is_3d': True}]]
                                 )
    # Insert histogram
    fig.append_trace(top_trace, 1, 1)
    fig.append_trace(bottom_trace, 1, 1)
    fig.append_trace(middle_trace, 1, 1)
    fig.append_trace(back_trace, 1, 1)
    fig.append_trace(reference_trace, 1, 1)
    fig.append_trace(MG_3D_trace, 1, 2)
    fig.append_trace(MG_3D_trace_2, 1, 3)
    # Assign layout with axis labels, title and camera angle
    a = 1.5
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=-2*a, y=-0.5*a, z=1.3*a)
                  )
    camera_2 = dict(up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=2*a, y=0.5*a, z=1.3*a)
                    )
    # Front view
    fig['layout']['scene1']['xaxis'].update(title='z [m]')
    fig['layout']['scene1']['yaxis'].update(title='x [m]')
    fig['layout']['scene1']['zaxis'].update(title='y [m]')
    fig['layout']['scene1']['camera'].update(camera)
    # Back view
    fig['layout']['scene2']['xaxis'].update(title='z [m]')
    fig['layout']['scene2']['yaxis'].update(title='x [m]')
    fig['layout']['scene2']['zaxis'].update(title='y [m]')
    fig['layout']['scene2']['camera'].update(camera_2)
    # Data
    fig['layout']['xaxis1'].update(title='Threshold [ADC channels]',
                                   showgrid=True)
    fig['layout']['yaxis1'].update(title='Counts/voxel',
                                   range=[0.9*np.log10(minimum),
                                          1.02*np.log10(maximum)],
                                   showgrid=True, type='log')
    fig.layout.legend = dict(x=0.3, y=1)
    fig['layout'].update(title=('Counts/voxel                                                   ' +
                                'Front view                                                     ' +
                                'Back view'),
                         height=600, width=1400)
    shapes = [{'type': 'line', 'x0': threshold, 'y0': -1000,
               'x1': threshold, 'y1': 20000000,
               'line':  {'color': 'rgb(0, 0, 0)', 'width': 5},
               'opacity': 0.7}
              ]
    fig['layout'].update(shapes=shapes)
    pio.write_image(fig, path)


# =============================================================================
# Uncertainty calculation - all calibrations
# =============================================================================

def calculate_all_uncertainties():
    # Declare parameters
    HR_calibrations, HF_calibrations = get_all_calibrations()
    HR_Ei, HF_Ei = get_all_energies()
    HR_results = {'Ei': HR_Ei,
                  'mean': np.zeros(len(HR_calibrations)),
                  'STD': np.zeros(len(HR_calibrations))}
    HF_results = {'Ei': HF_Ei,
                  'mean': np.zeros(len(HF_calibrations)),
                  'STD': np.zeros(len(HF_calibrations))}
    # Declare output folder
    dir_name = os.path.dirname(__file__)
    output_folder = os.path.join(dir_name, '../../Results/Uncertainty/')
    # Iterate through all calibrations
    for calibrations, results in zip([HR_calibrations, HF_calibrations],
                                     [HR_results, HF_results]):
        for i, calibration in enumerate(calibrations):
            fig, mean, STD = calculate_uncertainty(calibration)
            results['mean'][i] = mean
            results['STD'][i] = STD
            fig.savefig(output_folder + calibration + '.pdf',
                        bbox_inches='tight')
            plt.close()
    # Plot summary of all mean and STD
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('$Ei$ [meV]')
    plt.ylabel('Mean value with STD')
    plt.xscale('log')
    plt.title('Uncertainty in isotropic distribution of V-scattering')
    for calibration, results in zip(['HR', 'HF'], [HR_results, HF_results]):
        print('Array lengths')
        print(len(results['Ei']))
        print(len(results['mean']))
        print(len(results['STD']))
        plt.errorbar(results['Ei'], results['mean'], results['STD'],
                     fmt='.', capsize=5, zorder=5, linestyle='-',
                     label=calibration)
        print(calibration)
        print(results['STD'])
    plt.legend()
    fig.savefig(output_folder + 'summary.pdf', bbox_inches='tight')
    return HR_results['STD'], HF_results['STD']

















