import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
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
                                      get_charge,
                                      export_dE_histograms_to_text,
                                      get_peak_area_and_FWHM,
                                      import_HR_energies,
                                      import_HF_energies,
                                      import_efficiency_correction,
                                      import_efficiency_theoretical,
                                      import_C4H2I2S_MG_files,
                                      import_C4H2I2S_He3_files,
                                      get_detector,
                                      get_calibration
                                      )

from Plotting.Miscellaneous import calculate_all_uncertainties

# =============================================================================
# Energy transfer - MG
# =============================================================================


def energy_transfer_histogram(df, calibration, Ei, window):
    # Declare parameters
    number_bins = int(window.dE_bins.text())
    calibration = get_calibration(calibration, Ei)
    # Filter Multi-Grid data
    df = filter_ce_clusters(window, df)
    # Calculate Multi-Grid energy transfer
    dE = get_MG_dE(df, calibration, Ei)
    # Histogram dE data
    dE_hist, bins = np.histogram(dE, bins=number_bins, range=[-Ei, Ei])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Plot data
    fig = plt.figure()
    plt.plot(bin_centers, dE_hist, '.-', color='black', zorder=5)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Energy transfer\nData set: %s' % calibration)
    return fig


# =============================================================================
# Energy transfer - MG (normalized on proton charge and solid angle) and He3
# =============================================================================


def energy_transfer_compare_MG_and_He3(df_MG, calibration, Ei, MG_solid_angle,
                                       He3_solid_angle, p0, window):
    # Declare parameters
    number_bins = int(window.dE_bins.text())
    # Filter Multi-Grid data
    df_MG = filter_ce_clusters(window, df_MG)
    # Calculate Multi-Grid and He3 energy transfer
    dE_MG = get_MG_dE(df_MG, calibration, Ei)
    dE_He3 = get_raw_He3_dE(calibration, Ei)
    # Histogram dE data
    dE_MG_hist, __ = np.histogram(dE_MG, bins=number_bins, range=[-Ei, Ei])
    dE_He3_hist, bins = np.histogram(dE_He3, bins=number_bins, range=[-Ei, Ei])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Calculate normalization based on accumulated charge and solid angle
    MG_charge, He3_charge = get_charge(calibration)
    norm_charge_solid_angle = ((He3_solid_angle/MG_solid_angle)
                               * (He3_charge/MG_charge))
    dE_MG_hist_normalized = dE_MG_hist * norm_charge_solid_angle
    # Calculate elastic peak area ratio and FHWM
    MG_peak_area, MG_FWHM, p0 = get_peak_area_and_FWHM(bin_centers,
                                                       dE_MG_hist_normalized,
                                                       calibration, p0, 'MG',
                                                       window)
    He3_peak_area, He3_FWHM, p0 = get_peak_area_and_FWHM(bin_centers,
                                                         dE_He3_hist,
                                                         calibration, p0,
                                                         'He3', window)
    peak_ratio = MG_peak_area/He3_peak_area
    # Plot data
    fig = plt.figure()
    plt.plot(bin_centers, dE_MG_hist_normalized, '-', color='red',
             label='Multi-Grid', zorder=5)
    plt.plot(bin_centers, dE_He3_hist, '-', color='blue',
             label='$^3$He-tubes', zorder=5)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Normalized Counts')
    plt.yscale('log')
    plt.title('Energy transfer%s\nData set: %s' % (get_detector(window),
                                                   calibration))
    plt.legend(loc=1)
    # Export histograms to text-files
    export_dE_histograms_to_text(calibration, bin_centers, dE_MG_hist,
                                 dE_He3_hist)
    return [fig, peak_ratio, MG_FWHM, He3_FWHM, p0]


# =============================================================================
# Energy transfer - Iterate through all energies
# =============================================================================

def plot_all_energies_dE(window):
    def get_energy(element):
        start = element.find('Calibration_')+len('Calibration_')
        stop = element.find('_meV')
        return float(element[start:stop])

    def append_folder_and_files(folder, files):
        folder_vec = np.array(len(files)*[folder])
        return np.core.defchararray.add(folder_vec, files)

    # Declare all input-paths
    dir_name = os.path.dirname(__file__)
    HF_folder = os.path.join(dir_name, '../../Clusters/MG_new/HF/')
    HF_files = np.array([file for file in os.listdir(HF_folder)
                         if file[-3:] == '.h5'])
    HF_files_sorted = sorted(HF_files, key=lambda element: get_energy(element))
    Van_3x3_HF_clusters = append_folder_and_files(HF_folder, HF_files_sorted)
    HR_folder = os.path.join(dir_name, '../../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder)
                         if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: get_energy(element))
    Van_3x3_HR_clusters = append_folder_and_files(HR_folder, HR_files_sorted)
    input_paths = np.concatenate((Van_3x3_HR_clusters, Van_3x3_HF_clusters),
                                 axis=None)
    # Declare output folder
    output_folder = os.path.join(dir_name, '../../Results/Energy_transfer/')
    # Calculate He3-solid angle
    __, He3_solid_angle = get_He3_tubes_area_and_solid_angle()
    # Declare data vectors
    energies = {'HR': [], 'HF': []}
    peak_ratios = {'HR': [], 'HF': []}
    MG_FWHMs = {'HR': [], 'HF': []}
    He3_FWHMs = {'HR': [], 'HF': []}
    # Iterate through all files and produce ToF-histograms
    p0 = [1.20901528e+04, 5.50978749e-02, 1.59896619e+00]
    for input_path in input_paths:
        # Import meta data
        Ei = import_MG_Ei(input_path)
        calibration = import_MG_calibration(input_path)
        # Import data
        df_MG = import_MG_coincident_events(input_path)
        __, MG_solid_angle = get_multi_grid_area_and_solid_angle(window,
                                                                 calibration,
                                                                 Ei)
        # Plot dE-histogram (MG and He3 comparison)
        values = energy_transfer_compare_MG_and_He3(df_MG, calibration, Ei,
                                                    MG_solid_angle,
                                                    He3_solid_angle, p0,
                                                    window)
        fig, peak_ratio, MG_FWHM, He3_FWHM, p0 = values
        # Store values in vectors
        setting = get_chopper_setting(calibration)
        energies[setting].append(Ei)
        peak_ratios[setting].append(peak_ratio)
        MG_FWHMs[setting].append(MG_FWHM)
        He3_FWHMs[setting].append(He3_FWHM)
        # Save figure
        fig.savefig(output_folder + calibration + '.pdf')
        plt.close()

    # Export values
    for setting in ['HR', 'HF']:
        overview_folder = os.path.join(dir_name,
                                       '../../Results/Summary/'
                                       + setting + '_overview/')
        np.savetxt(overview_folder + 'energies.txt',
                   energies[setting], delimiter=",")
        np.savetxt(overview_folder + 'peak_ratios.txt',
                   peak_ratios[setting], delimiter=",")
        np.savetxt(overview_folder + 'He3_FWHMs.txt',
                   He3_FWHMs[setting], delimiter=",")
        np.savetxt(overview_folder + 'MG_FWHMs.txt',
                   MG_FWHMs[setting], delimiter=",")


# =============================================================================
# Efficiency
# =============================================================================

def plot_efficiency():
    def energy_correction(energy):
        A = 0.99827
        b = 1.8199
        return A/(1-np.exp(-b*meV_to_A(energy)))

    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    # Import energies
    HR_energies = import_HR_energies()
    HF_energies = import_HF_energies()
    # Import uncertainties
    HR_uncertainty, HF_uncertainty = calculate_all_uncertainties()
    uncertainties = [HR_uncertainty, HF_uncertainty]
    # Import theoretical equation
    eff_theo = import_efficiency_theoretical()
    shift = 1  # 0.7
    energies = np.array([HR_energies, HF_energies])
    # Declare parameters
    dir_name = os.path.dirname(__file__)
    data_set_names = ['HR', 'HF']
    color_vec = ['blue', 'red']
    # Plot data
    fig = plt.figure()
    plt.plot(eff_theo[0], eff_theo[1], color='black', label='Theoretical',
             zorder=3)
    for i, data_set_name in enumerate(data_set_names):
        overview_folder = os.path.join(dir_name, '../../Results/Summary/'
                                                 + data_set_name
                                                 + '_overview/')
        ratios = np.loadtxt(overview_folder + 'peak_ratios.txt', delimiter=",")
        efficiency = (ratios / (energy_correction(energies[i]))) * shift
        uncertainty = efficiency * uncertainties[i]
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Efficiency vs Energy\n(Peak area comparrison, MG/He3,'
                  + 'including efficiency correction for He3)')
        #plt.plot(energies[i], efficiency, '-x', color=color_vec[i],
        #         label=data_set_name + '(Scaled by: %.2f)' % shift,
        #         zorder=5)
        plt.errorbar(energies[i], efficiency, uncertainty,
                     fmt='.', capsize=5, zorder=5, linestyle='-',
                     label=data_set_name + '(Scaled by: %.2f)' % shift,
                     color=color_vec[i])
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        plt.ylabel('Efficiency')
        plt.legend()
    plt.tight_layout()
    fig.show()
    #plt.close()
    #fig = plt.figure()
    #plt.grid(True, which='major', zorder=0)
    #plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.plot(np.concatenate((HR_energies, HF_energies), axis=0),
    #         energy_correction(np.concatenate((HR_energies, HF_energies),
    #                                          axis=0)),
    #         'o', color='black')
    #plt.xlabel('$E_i$ [meV]')
    #plt.ylabel('Correction')
    #plt.title('Efficiency correction for $^3$He-tubes at SEQUOIA')
    #fig.show()


# =============================================================================
# FWHM
# =============================================================================

def plot_FWHM():
    # Import energies
    HR_energies = import_HR_energies()
    HF_energies = import_HF_energies()
    energies = [HR_energies, HF_energies]
    # Declare parameters
    dir_name = os.path.dirname(__file__)
    data_set_names = ['HR', 'HF']
    # Plot data
    fig = plt.figure()
    for i, data_set_name in enumerate(data_set_names):
        overview_folder = os.path.join(dir_name, '../../Results/Summary/'
                                                 + data_set_name
                                                 + '_overview/')
        He3_FWHM = np.loadtxt(overview_folder + 'He3_FWHMs.txt',
                              delimiter=",")
        MG_FWHM = np.loadtxt(overview_folder + 'MG_FWHMs.txt',
                             delimiter=",")
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('FWHM vs Energy')
        plt.plot(energies[i], MG_FWHM, '-x', label=data_set_name + ', MG',
                 zorder=5)
        plt.plot(energies[i], He3_FWHM, '-x', label=data_set_name + ', He3',
                 zorder=5)
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('FWHM [meV]')
        plt.legend()
    plt.tight_layout()
    fig.show()


# =============================================================================
# C4H2I2S - Compare all energies
# =============================================================================

def C4H2I2S_compare_all_energies(window):
    # Declare folder paths
    dir_name = os.path.dirname(__file__)
    MG_folder = os.path.join(dir_name, '../../Clusters/MG/')
    He3_folder = os.path.join(dir_name, '../../Archive/2019_01_10_SEQ_Diiodo/')
    # Import file names
    MG_files = import_C4H2I2S_MG_files()
    He3_files = import_C4H2I2S_He3_files()
    # Declare parameters
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    number_of_bins = 290
    Energies = [21, 35, 50.12, 70.13, 99.9, 197.3, 296.1, 492]
    calibrations = ['C4H2I2S_21.0', 'C4H2I2S_35.0', 'C4H2I2S_50.0',
                    'C4H2I2S_70.0', 'C4H2I2S_100.0', 'C4H2I2S_200.0',
                    'C4H2I2S_300.0', 'C4H2I2S_500.0']
    # Iterate through all energies
    for MG_file, He3_files_short, Ei, calibration, color in zip(MG_files,
                                                                He3_files,
                                                                Energies,
                                                                calibrations,
                                                                colors):
        # Import He3 data and histogram dE
        He3_dE_hist = np.zeros(number_of_bins)
        for file_number, measurement_id in enumerate(He3_files_short):
            nxs_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxs'
            nxs_path = He3_folder + nxs_file
            nxs = h5py.File(nxs_path, 'r')
            he3_bins = (nxs['mantid_workspace_1']['event_workspace']
                           ['axis1'].value)
            he3_min = he3_bins[0]
            he3_max = he3_bins[-1]
            He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
            dE = nxs['mantid_workspace_1']['event_workspace']['tof'].value
            He3_dE_hist_small, __ = np.histogram(dE, bins=number_of_bins,
                                                 range=[he3_min, he3_max])
            He3_dE_hist += He3_dE_hist_small
        # Import MG data and calculate energ transfer
        df_MG = pd.read_hdf(MG_folder+MG_file, 'coincident_events')
        df_MG = filter_ce_clusters(window, df_MG)
        dE_MG = get_MG_dE(df_MG, calibration, Ei)
        # Histogram dE data
        MG_dE_hist, __ = np.histogram(dE_MG, bins=number_of_bins,
                                      range=[he3_min, he3_max])
        # Normalize MG data
        norm = sum(He3_dE_hist)/sum(MG_dE_hist)
        MG_dE_hist_normalized = MG_dE_hist * norm
        # Plot data
        fig = plt.figure()
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('C$_4$H$_2$I$_2$S\nE$_i$: %.2f' % Ei)
        plt.plot(He3_bin_centers, MG_dE_hist_normalized,
                 color=color,
                 label='Multi-Grid (100 meV)', zorder=5)
        plt.plot(He3_bin_centers, He3_dE_hist,
                 linestyle='--',
                 color='black',
                 label='$^3$He-tubes', zorder=5)
        plt.legend(loc=1)
        plt.xlabel('$E_i$ - $E_f$ [meV]')
        plt.ylabel('Normalized counts')
        plt.yscale('log')
        file_name = 'C4H2I2S_%d_meV.pdf' % int(Ei)
        fig.savefig(os.path.join(dir_name, '../../Results/C4H2I2S/%s'
                                           % file_name))
        plt.close()





