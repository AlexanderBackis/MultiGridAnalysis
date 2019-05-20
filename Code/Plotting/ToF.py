import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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
                                      get_detector,
                                      get_module_to_exclude
                                      )


# =============================================================================
# ToF - MG
# =============================================================================


def ToF_histogram(df, data_sets, window):
    # Filter clusters
    df = filter_ce_clusters(window, df)
    # Get parameters
    number_bins = int(window.tofBins.text())
    min_val = 0
    max_val = 16667
    # Produce histogram and plot
    fig = plt.figure()
    plt.hist(df.ToF * 62.5e-9 * 1e6, bins=number_bins,
             range=[min_val, max_val],
             log=True, color='black', zorder=4,
             histtype='step', label='MG'
             )
    title = 'ToF - histogram\n%s' % data_sets
    x_label = 'ToF [$\mu$s]'
    y_label = 'Counts'
    fig = stylize(fig, x_label=x_label, y_label=y_label, title=title,
                  yscale='log', grid=True)
    return fig


# =============================================================================
# ToF - MG (normalized on area and time) and He3
# =============================================================================


def ToF_compare_MG_and_He3(df, calibration, Ei, MG_time, He3_time,
                           MG_area, He3_area, window):
    # Declare parameters
    number_bins = int(window.tofBins.text())
    frame_shift = get_frame_shift(Ei) * 1e6
    df_He3 = load_He3_h5(calibration)
    # Filter data
    df_MG = filter_ce_clusters(window, df)
    # Produce histogram data
    He3_hist, He3_bins = np.histogram(df_He3.ToF, bins=number_bins)
    MG_hist, MG_bins = np.histogram(df_MG.ToF * 62.5e-9 * 1e6 + frame_shift,
                                    bins=number_bins)
    He3_bin_centers = 0.5 * (He3_bins[1:] + He3_bins[:-1])
    MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
    # Normalize Multi-Grid data based on area and time
    norm_area_time = (He3_area/MG_area) * (He3_time/MG_time)
    MG_hist_normalized = MG_hist * norm_area_time
    # Get background estimation
    MG_back_bin_centers, MG_back_hist, __, __ = get_ToF_background(window,
                                                                   calibration,
                                                                   Ei, MG_time,
                                                                   He3_time,
                                                                   MG_area,
                                                                   He3_area,
                                                                   number_bins
                                                                   )
    # Plot data
    fig = plt.figure()
    plt.plot(He3_bin_centers, He3_hist, color='blue',
             label='$^3$He-tubes', zorder=4)
    plt.plot(MG_bin_centers, MG_hist_normalized, color='red',
             label='Multi-Grid', zorder=3)
    plt.plot(MG_back_bin_centers, MG_back_hist, color='green',
             label='Background (MG)', zorder=5)
    plt.xlabel('ToF [$\mu$s]')
    plt.ylabel('Normalized Counts')
    plt.yscale('log')
    plt.legend(loc=1)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.title('ToF%s\n%s' % (get_detector(window), calibration))
    # Export histograms to text-files
    export_ToF_histograms_to_text(calibration, MG_bin_centers, He3_bin_centers,
                                  MG_hist_normalized, He3_hist, MG_back_hist)
    return fig


# =============================================================================
# ToF - MG background
# =============================================================================

def get_ToF_background(window, calibration, Ei, MG_time, He3_time,
                       MG_area, He3_area, number_bins=100):
    # Declare parameters
    frame_shift = get_frame_shift(Ei) * 1e6
    # Import background data
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, '../../Clusters/MG/Background.h5')
    df_back = pd.read_hdf(path, 'coincident_events')
    # Filter data
    df_back = filter_ce_clusters(window, df_back)
    df_back = df_back[df_back.Time < 1.5e12]
    module_to_exclude = get_module_to_exclude(calibration, Ei)
    if module_to_exclude is not None:
        df_back = df_back[df_back.Bus != module_to_exclude]
    # Calculate background duration
    start_time = df_back.head(1)['Time'].values[0]
    end_time = df_back.tail(1)['Time'].values[0]
    duration = (end_time - start_time) * 62.5e-9
    # Get normalization based on time and area
    norm_area_time = (He3_area/MG_area) * (He3_time/MG_time)
    # Calculate weights
    number_of_events = df_back.shape[0]
    events_per_s = number_of_events / duration
    events_s_norm = events_per_s / number_of_events
    back_weight = events_s_norm * MG_time * norm_area_time
    # Histogram background
    MG_back_hist, MG_bins = np.histogram(df_back.ToF*62.5e-9*1e6+frame_shift,
                                         bins=number_bins)
    MG_back_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
    return MG_back_bin_centers, MG_back_hist*back_weight, df_back, back_weight


# =============================================================================
# ToF - Compare background level in MG and He3
# =============================================================================


def compare_MG_and_He3_background(MG_interval, MG_back_interval, He3_interval,
                                  df_MG, df_MG_back, df_He3, calibration, Ei,
                                  window, MG_time, He3_time, MG_area, He3_area,
                                  back_weight):
    def get_counts_per_us(df, min_val, max_val, isMG=True, frame_shift=0):
        if isMG:
            counts = df[((df.ToF * 62.5e-9 * 1e6 + frame_shift) >= min_val) &
                        ((df.ToF * 62.5e-9 * 1e6 + frame_shift) <= max_val)
                        ].shape[0] / (max_val - min_val)
        else:
            counts = df[(df.ToF >= min_val) & (df.ToF <= max_val)
                        ].shape[0] / (max_val - min_val)
        return counts

    def get_statistical_uncertainty(value, a, b):
        da = np.sqrt(a)
        db = np.sqrt(b)
        return np.sqrt((da/a) ** 2 + (db/b) ** 2) * value

    # Declare parameters
    frame_shift = get_frame_shift(Ei) * 1e6
    # Get normalization based on time and area
    norm_area_time = (He3_area/MG_area) * (He3_time/MG_time)
    # Count events within interval for MG, MG background estimation and He3
    counts_MG = get_counts_per_us(df_MG, MG_interval[0], MG_interval[1],
                                  isMG=True, frame_shift=frame_shift)
    counts_MG_back = get_counts_per_us(df_MG_back, MG_back_interval[0],
                                       MG_back_interval[1], isMG=True,
                                       frame_shift=0)
    counts_He3 = get_counts_per_us(df_He3, He3_interval[0], He3_interval[1],
                                   isMG=False, frame_shift=0)
    # Calculate ratios
    MG_over_MG_back = (counts_MG * norm_area_time) / counts_MG_back
    MG_back_over_He3 = (counts_MG_back * back_weight) / counts_He3
    MG_over_He3 = (counts_MG * norm_area_time) / counts_He3
    # Calculate statistical uncertainties
    MG = counts_MG * (MG_interval[1] - MG_interval[0])
    MG_back = counts_MG_back * (MG_back_interval[1] - MG_back_interval[0])
    He3 = counts_He3 * (He3_interval[1] - He3_interval[0])
    error_MG_over_MG_back = get_statistical_uncertainty(MG_over_MG_back,
                                                        MG, MG_back)
    error_MG_back_over_He3 = get_statistical_uncertainty(MG_back_over_He3,
                                                         MG_back, He3)
    error_MG_over_He3 = get_statistical_uncertainty(MG_over_He3, MG, He3)
    return ({'MG_over_MG_back': [MG_over_MG_back, error_MG_over_MG_back],
             'MG_back_over_He3': [MG_back_over_He3, error_MG_back_over_He3],
             'MG_over_He3': [MG_over_He3, error_MG_over_He3]
             })

# =============================================================================
# ToF - Iterate through all energies (and compare background)
# =============================================================================


def plot_all_energies_ToF(window):
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
    output_folder = os.path.join(dir_name, '../../Results/ToF/')
    # Declare ToF-intervals (us) for background comparison
    intervals = get_ToF_intervals()
    # Calculate He3-area and number of bins
    He3_area, __ = get_He3_tubes_area_and_solid_angle()
    number_bins = int(window.tofBins.text())
    # Declare vectors for data
    ratios_HR = {'MG_over_MG_back': [[], [], []],
                 'MG_back_over_He3': [[], [], []],
                 'MG_over_He3': [[], [], []]
                 }
    ratios_HF = {'MG_over_MG_back': [[], [], []],
                 'MG_back_over_He3': [[], [], []],
                 'MG_over_He3': [[], [], []]
                 }
    data = {'HR': ratios_HR, 'HF': ratios_HF}
    # Iterate through all files and produce ToF-histograms
    for input_path, intervals in zip(input_paths, intervals):
        # Exctract intervals
        MG_interval = [intervals[0], intervals[1]]
        MG_back_interval = [intervals[2], intervals[3]]
        He3_interval = [intervals[4], intervals[5]]
        # Import meta data
        Ei = import_MG_Ei(input_path)
        calibration = import_MG_calibration(input_path)
        MG_time = import_MG_measurement_time(input_path)
        # Import data
        df_MG = import_MG_coincident_events(input_path)
        MG_area, __ = get_multi_grid_area_and_solid_angle(window, calibration,
                                                          Ei)
        df_He3 = load_He3_h5(calibration)
        He3_time = get_He3_duration(calibration)
        __, __, df_MG_back, back_weight = get_ToF_background(window,
                                                             calibration,
                                                             Ei, MG_time,
                                                             He3_time,
                                                             MG_area,
                                                             He3_area,
                                                             number_bins
                                                             )
        # Plot ToF-histogram (MG and He3 comparison)
        fig = ToF_compare_MG_and_He3(df_MG, calibration, Ei, MG_time, He3_time,
                                     MG_area, He3_area, window)
        # Get background ratios
        ratios = compare_MG_and_He3_background(MG_interval, MG_back_interval,
                                               He3_interval, df_MG, df_MG_back,
                                               df_He3, calibration, Ei, window,
                                               MG_time, He3_time, MG_area,
                                               He3_area,
                                               back_weight)

        fig.savefig(output_folder + 'ToF' + calibration + '.pdf')
        # Extract ratios
        setting = get_chopper_setting(calibration)
        for key, value in zip(ratios.keys(), ratios.values()):
            data[setting][key][0].append(Ei)
            data[setting][key][1].append(value[0])
            data[setting][key][2].append(value[1])
        plt.close()

    # Plot summary of all ratios
    keys = ['MG_over_MG_back', 'MG_back_over_He3', 'MG_over_He3']
    titles = ['MG over MG-background', 'MG-background over He3', 'MG over He3']
    for key, title in zip(keys, titles):
        ratio_HR = data['HR'][key]
        ratio_HF = data['HF'][key]
        # Get averages
        average_HR = sum(ratio_HR[1])/len(ratio_HR[1])
        average_HF = sum(ratio_HF[1])/len(ratio_HF[1])
        fig = plt.figure()
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.errorbar(ratio_HR[0], ratio_HR[1], ratio_HR[2],
                     color='blue', ecolor='blue', capsize=5,
                     label='V_3x3_HR (average: %.2f)' % average_HR,
                     zorder=5, fmt='o', linestyle='-'
                     )
        plt.errorbar(ratio_HF[0], ratio_HF[1], ratio_HF[2],
                     color='red', ecolor='red', capsize=5,
                     label='V_3x3_HF (average: %.2f)' % average_HF,
                     zorder=5, fmt='o', linestyle='-'
                     )

        plt.xlabel('Energy [meV]')
        plt.ylabel('Ratio %s' % title)
        plt.title('Comparison between %s in flat ToF region' % title)
        plt.xscale('log')
        plt.legend()
        fig.savefig(output_folder + key + '.pdf')
        plt.close()















