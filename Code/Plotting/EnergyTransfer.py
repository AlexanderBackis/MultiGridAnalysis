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
                                      get_MG_dE,
                                      get_charge,
                                      export_dE_histograms_to_text,
                                      get_peak_area_and_FWHM,
                                      import_HR_energies,
                                      import_HF_energies,
                                      import_efficiency_correction,
                                      import_efficiency_theoretical
                                      )

# =============================================================================
# Energy transfer - MG
# =============================================================================


def energy_transfer_histogram(df, calibration, Ei, window):
    # Declare parameters
    number_bins = int(window.dE_bins.text())
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
                                                       calibration, p0, 'MG')
    He3_peak_area, He3_FWHM, p0 = get_peak_area_and_FWHM(bin_centers,
                                                         dE_He3_hist,
                                                         calibration, p0,
                                                         'He3')
    peak_ratio = MG_peak_area/He3_peak_area
    # Plot data
    fig = plt.figure()
    plt.plot(bin_centers, dE_MG_hist_normalized, '.-', color='red',
             label='Multi-Grid', zorder=5)
    plt.plot(bin_centers, dE_He3_hist, '.-', color='blue',
             label='$^3$He-tubes', zorder=5)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('$E_i$ - $E_f$ [meV]')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Energy transfer\nData set: %s' % calibration)
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
    # Import theoretical equation
    eff_theo = import_efficiency_theoretical()
    shift = 1.35
    energies = [HR_energies, HF_energies]
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
        efficiency = ratios / (energy_correction(energies[i]) * shift)
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Efficiency vs Energy\n(Peak area comparrison, MG/He3,'
                  + 'including efficiency correction for He3)')
        plt.plot(energies[i], efficiency, '-x', color=color_vec[i],
                 label=data_set_name, zorder=5)
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        plt.ylabel('Efficiency')
        plt.legend()
    plt.tight_layout()
    fig.show()


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





