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
    calibration = get_calibration(calibration, Ei)
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
    output_folder = os.path.join(dir_name, '../../Results/Energy_transfer/')
    # Calculate He3-solid angle
    He3_area, He3_solid_angle = get_He3_tubes_area_and_solid_angle()
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
        calibration = get_calibration(calibration, Ei)
        # Import data
        df_MG = import_MG_coincident_events(input_path)
        MG_area, MG_solid_angle = get_multi_grid_area_and_solid_angle(window,
                                                                      calibration,
                                                                      Ei)
        print('Area ratio: ' + str(He3_area/MG_area))
        print('Solid angle ratio: ' + str(He3_solid_angle/MG_solid_angle))
        # Plot dE-histogram (MG and He3 comparison)
        values = energy_transfer_compare_MG_and_He3(df_MG, calibration,
                                                    Ei,
                                                    MG_solid_angle,
                                                    He3_solid_angle,
                                                    p0,
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
        # Plot and save just MG
        #fig = energy_transfer_histogram(df_MG, calibration, Ei, window)
        #fig.savefig(output_folder + '/MG/' + calibration + '.pdf')
        #plt.close()

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
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def energy_correction(energy):
        A = 0.99827
        b = 1.8199
        return A/(1-np.exp(-b*meV_to_A(energy)))

    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    def A_to_meV(wavelength):
        return (81.81/(wavelength ** 2))

    # Declare parameters
    dir_name = os.path.dirname(__file__)
    data_set_names = ['HR', 'HF']
    color_vec = ['blue', 'red']
    # Import energies
    HR_energies = import_HR_energies()
    HF_energies = import_HF_energies()
    # Import uncertainties
    uncertainties = []
    overview_folder = os.path.join(dir_name, '../../Results/Summary/')
    for setting in ['HR', 'HF']:
        path = overview_folder + setting + '_overview/uncertainties.txt'
        uncertainties.append(np.loadtxt(path, delimiter=","))
    # Import theoretical equation
    eff_theo = import_efficiency_theoretical()
    energies = np.array([HR_energies, HF_energies])
    new_theo_path = os.path.join(dir_name, '../../Tables/boron10data.txt')
    new_theo_array = np.transpose(np.loadtxt(new_theo_path, delimiter=","))
    # Import He3 efficiency
    He3_theo_path = os.path.join(dir_name, '../../Tables/He3eff_122mm.txt')
    He3_theo_array = np.transpose(np.loadtxt(He3_theo_path, delimiter="\t"))
    He3_energies = A_to_meV(He3_theo_array[0])
    He3_effiency = He3_theo_array[1]
    # Plot data
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(14)
    #plt.plot(He3_energies, He3_effiency, label='He3-theo')
    #plt.plot(A_to_meV(new_theo_array[0]), new_theo_array[1]/100, color='black',
    #         label='Analytical - PYTHON', #linestyle='dotted',
    #         zorder=10)
    #plt.plot(eff_theo[0], eff_theo[1], label='Analytical - MATLAB',
    #         color='green')
    #plt.plot(energies[0][:12], 1/(energy_correction(energies[0][0:12])),
    #             color='black', linestyle='--', label=None)
    #plt.plot(energies[1], 1/(energy_correction(energies[1])),
    #         color='black', linestyle='--', label='Theoretical - $^3$He')
    scaling_factor = 1/1.210490
    for i, data_set_name in enumerate(data_set_names):
        He3_efficiency = []
        B_10_calculation_data_points = []
        for energy in energies[i]:
            # Save He3 efficiencies for data points
            idx = find_nearest_idx(He3_energies, energy)
            He3_efficiency.append(He3_effiency[idx])
            # Save B-10 efficiencies for data points
            idx_B_10 = find_nearest_idx(eff_theo[0], energy)
            B_10_calculation_data_points.append(eff_theo[1][idx_B_10])
        He3_efficiency = np.array(He3_efficiency)
        B_10_calculation_data_points = np.array(B_10_calculation_data_points)
        print('He3 efficiency')
        print(He3_efficiency)
        overview_folder = os.path.join(dir_name, '../../Results/Summary/'
                                                 + data_set_name
                                                 + '_overview/')
        ratios = np.loadtxt(overview_folder + 'peak_ratios.txt', delimiter=",")
        used_He3_calculation = He3_efficiency # (1 / (energy_correction(energies[i]))) #
        efficiency = (ratios * used_He3_calculation)
        uncertainty = efficiency * uncertainties[i]
        plt.subplot(1, 3, 1)
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Measurement')
        plt.errorbar(energies[i], efficiency, uncertainty,
                     fmt='.', capsize=5, zorder=5, linestyle='-',
                     label='Measurement - ' + data_set_name,
                     color=color_vec[i])
        plt.xlabel('$E_i$ [meV]')
        plt.xscale('log')
        #plt.xlim(0, 6.5)
        plt.ylabel('Efficiency')
        if i == 0:
            plt.plot(eff_theo[0], eff_theo[1], label='Analytical - MATLAB',
                     color='green')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.plot(energies[i], efficiency/B_10_calculation_data_points, '.',
                 color='black')
        plt.xlabel('E_i [meV]')
        plt.ylabel('Ratios (measurement/calculation)')
        plt.title('Discrepency with Calculation')
        plt.xscale('log')
        plt.subplot(1, 3, 3)
        plt.grid(True, which='major', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        if i == 1:
            plt.plot(energies[i], used_He3_calculation, '.',
                     color='red', label='He-3 efficiency')
            plt.plot(energies[i], (1/(efficiency/B_10_calculation_data_points)) * used_He3_calculation, '.',
                     color='blue', label='Rescaled He-3 efficiency')
        else:
            plt.plot(energies[i], used_He3_calculation, '.',
                     color='red', label=None)
            plt.plot(energies[i], (1/(efficiency/B_10_calculation_data_points)) * used_He3_calculation, '.',
                     color='blue', label=None)
        plt.legend()
        plt.xlabel('E_i [meV]')
        plt.ylabel('Efficiency')
        plt.title('Rescaled He-3, using discrepency')
        plt.xscale('log')
        print('Testing new procedure')
        print('measurement')
        print(efficiency)
        print('calculation')
        print(B_10_calculation_data_points)
    plt.tight_layout()
    fig.show()

    #fig = plt.figure()
    #plt.grid(True, which='major', zorder=0)
    #plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.plot(np.concatenate((HR_energies, HF_energies), axis=0),
    #         1/energy_correction(np.concatenate((HR_energies, HF_energies),
    #                                             axis=0)),
    #         'o', color='black')
    #plt.xlabel('$\lambda$ [Å]')
    #plt.ylabel('Efficiency')
    #plt.title('Efficiency of $^3$He-tubes at SEQUOIA')
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
    print(HR_energies)
    plt.tight_layout()
    fig.show()


# =============================================================================
# C4H2I2S - Compare all energies
# =============================================================================

def C4H2I2S_compare_all_energies(window):
    # Declare folder paths
    dir_name = os.path.dirname(__file__)
    MG_folder = os.path.join(dir_name, '../../Clusters/MG_old/')
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
        # Import MG data, calculate energy transfer and filter
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
                 label='Multi-Grid', zorder=5)
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
