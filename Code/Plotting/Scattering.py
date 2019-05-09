import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import imageio
import scipy
from Plotting.HelperFunctions import (stylize, filter_ce_clusters, get_MG_dE,
                                      get_raw_He3_dE, get_calibration,
                                      Gaussian_fit, find_nearest,
                                      detector_filter, mkdir_p, get_energy,
                                      get_chopper_setting,
                                      append_folder_and_files, get_charge,
                                      get_He3_tubes_area_and_solid_angle,
                                      get_multi_grid_area_and_solid_angle,
                                      get_all_energies,
                                      get_detector
                                      )

# =============================================================================
# Peak shape comparison - 5x5
# =============================================================================


def compare_all_shoulders_5x5(window):
    # Declare parameters
    number_bins = 150
    sigma_interval = [3, 8]
    detectors = ['ILL', 'ESS_CLB', 'ESS_PA']
    FoMs = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    colors = {'ILL': 'blue', 'ESS_CLB': 'red', 'ESS_PA': 'green'}
    energies = []
    errors = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    count = 0
    # Get input-file paths
    dir_name = os.path.dirname(__file__)
    folder = os.path.join(dir_name, '../../Clusters/MG_V_5x5_HR/')
    files = np.array([file for file in os.listdir(folder)
                      if file[-3:] == '.h5'
                      ])
    files_sorted = sorted(files, key=lambda element: float(element[element.find('V_5x5_HR_')+len('V_5x5_HR_'):element.find('_meV.h5')]))
    paths = np.core.defchararray.add(np.array(len(files_sorted)*[folder]), files_sorted)
    # Get output path
    temp_folder = os.path.join(dir_name, '../../Results/Shoulder/temp/')
    mkdir_p(temp_folder)
    for run in range(2):
        for detector in detectors:
            for i, path in enumerate(paths):
                # Import parameters
                E_i = pd.read_hdf(path, 'E_i')['E_i'].iloc[0]
                calibration = pd.read_hdf(path, 'calibration')['calibration'].iloc[0]
                calibration = get_calibration(calibration, E_i)
                df_MG_temp = pd.read_hdf(path, 'coincident_events')
                df_MG = filter_ce_clusters(window, df_MG_temp)
                # Get MG dE-data
                df_MG_detector = detector_filter(df_MG, detector)
                dE_MG = get_MG_dE(df_MG_detector, calibration, E_i)
                # Get He3 dE-data
                dE_He3 = get_raw_He3_dE(calibration, E_i)
                # Produce histograms
                hist_MG, bins = np.histogram(dE_MG, bins=number_bins,
                                             range=[-E_i/5, E_i/5])
                hist_He3, __ = np.histogram(dE_He3, bins=number_bins,
                                            range=[-E_i/5, E_i/5])
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                # Normalize data
                p0 = None
                area_MG, FWHM_MG, __, __, __, popt_MG, l_p_idx_MG, r_p_idx_MG, back_MG = Gaussian_fit(bin_centers, hist_MG, p0)
                area_He3, FWHM_He3, __, __, __, popt_He3, l_p_idx_He3, r_p_idx_He3, back_He3 = Gaussian_fit(bin_centers, hist_He3, p0)
                norm_area = area_He3/area_MG
                normalized_hist_MG = hist_MG * norm_area
                # THIS IS JUST TEMPORARY REMOVE AFTER USE
                fig = plt.figure()
                plt.plot(bin_centers, normalized_hist_MG, '-', color='red',
                         label='Multi-Grid', zorder=5)
                plt.plot(bin_centers, hist_He3, '-', color='blue',
                         label='$^3$He-tubes', zorder=5)
                plt.grid(True, which='major', linestyle='--', zorder=0)
                plt.grid(True, which='minor', linestyle='--', zorder=0)
                plt.xlabel('$E_i$ - $E_f$ [meV]')
                plt.ylabel('Normalized Counts')
                plt.yscale('log')
                plt.title('Energy transfer - All detectors (5x5 V-sample)\nData set: %s' % calibration)
                plt.legend(loc=1)
                output_path_pdf = os.path.join(dir_name,
                                               '../../Results/pdf_files/%s.pdf' % count)
                fig.savefig(output_path_pdf, bbox_inches='tight')
                plt.close()


                # Compare difference at certain interval
                x0_MG = popt_MG[1]
                sigma_MG = abs(popt_MG[2])
                x0_He3 = popt_He3[1]
                sigma_He3 = abs(popt_He3[2])
                left_idx_MG = find_nearest(bin_centers, x0_MG+sigma_interval[0]*sigma_MG)
                right_idx_MG = find_nearest(bin_centers, x0_MG+sigma_interval[1]*sigma_MG)
                indices_MG = np.arange(left_idx_MG, right_idx_MG+1, 1)
                left_idx_He3 = find_nearest(bin_centers, x0_He3+sigma_interval[0]*sigma_He3)
                right_idx_He3 = find_nearest(bin_centers, x0_He3+sigma_interval[1]*sigma_He3)
                indices_He3 = np.arange(left_idx_He3, right_idx_He3+1, 1)
                area_MG = sum(normalized_hist_MG[indices_MG]) - len(indices_MG) * back_MG
                area_He3 = sum(hist_He3[indices_He3]) - len(indices_He3) * back_He3

                area_diff = sum(normalized_hist_MG[indices_MG] - hist_He3[indices_MG])
                He3_peak_area = sum(hist_He3[l_p_idx_He3:r_p_idx_He3]
                                    - np.ones(len(bin_centers[l_p_idx_He3:r_p_idx_He3])) * back_He3
                            )
                FoM = area_diff/He3_peak_area
                # Find uncertainty
                a = sum(hist_MG[indices_MG])
                b = sum(hist_He3[indices_MG])
                c = sum(hist_He3[l_p_idx_He3:r_p_idx_He3])
                d = sum(np.ones(len(bin_centers[l_p_idx_He3:r_p_idx_He3])) * back_He3)
                da = np.sqrt(a)
                db = np.sqrt(b)
                dc = np.sqrt(c)
                dd = np.sqrt(d)
                uncertainty = np.sqrt((np.sqrt(da ** 2 + db ** 2)/(a-b)) ** 2 + (np.sqrt(dc ** 2 + dd ** 2)/(c-d)) ** 2)
                error = uncertainty * FoM
                
                if run == 0:
                    FoMs[detector].append(FoM)
                    errors[detector].append(error)
                    if detector == 'ILL':
                        energies.append(E_i)
                else:
                    # Plot data
                    fig = plt.figure()
                    #main_title = 'Peak shape comparison $^3$He-tubes and Multi-Grid'
                    #main_title += '\nInterval: ' + str(sigma_interval) + '$\sigma$'
                    #fig.suptitle(main_title, x=0.5, y=1.02)
                    fig.set_figheight(6)
                    fig.set_figwidth(14)
                    plt.subplot(1, 2, 1)
                    plt.plot(bin_centers, normalized_hist_MG, color='red',
                             label='Multi-Grid', zorder=5)
                    plt.plot(bin_centers, hist_He3, color='blue', label='$^3$He-tubes',
                             zorder=5)
                    plt.fill_between(bin_centers[indices_MG],
                                     normalized_hist_MG[indices_MG],
                                     hist_He3[indices_MG],
                                     facecolor='orange',
                                     label='Shoulder area',
                                     alpha=1, zorder=2
                                     )
                    plt.fill_between(bin_centers[l_p_idx_He3:r_p_idx_He3],
                                     hist_He3[l_p_idx_He3:r_p_idx_He3],
                                     np.ones(len(bin_centers[l_p_idx_He3:r_p_idx_He3]))*back_He3,
                                     facecolor='purple',
                                     label='$^3$He-peak area',
                                     alpha=1, zorder=2
                                     )
                    plt.title(detector + ': ' + calibration)
                    plt.xlabel('$E_i$ - $E_f$ [meV]')
                    plt.ylabel('Normalized counts')
                    plt.yscale('log')
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.legend(loc=1)
                    plt.subplot(1, 2, 2)
                    for detector_temp in detectors:
                        plt.errorbar(energies, FoMs[detector_temp],
                                     errors[detector_temp],
                                     ecolor=colors[detector_temp],
                                     fmt='.',
                                     capsize=5,
                                     color=colors[detector_temp],
                                     label=detector_temp, zorder=5,
                                     linestyle='-'
                                     )
                     #   plt.plot(energies, FoMs[detector_temp], '.-',
                     #            color=colors[detector_temp],
                      #           label=detector_temp, zorder=5)
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.xlabel('$E_i$ [meV]')
                    plt.ylabel('Figure-of-Merit')
                    plt.title('Figure-of-Merit')
                    if ((detector == 'ESS_PA') and (i == 4)):
                        pass
                    else:
                        plt.axvline(x=energies[i], color=colors[detector],
                                    zorder=5)
                    plt.legend(loc=2)
                    # Save fig
                    count += 1
                    output_path = temp_folder + str(count) + '.png'
                    plt.tight_layout()
                    fig.savefig(output_path, bbox_inches='tight')
                    output_path_pdf = os.path.join(dir_name,
                                                   '../../Results/pdf_files/%s.pdf' % count)
                    #fig.savefig(output_path_pdf, bbox_inches='tight')

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name,
                               '../../Results/Shoulder/FOM_sweep_shoulder.gif')

    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images)  #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)

# =============================================================================
# Peak shape comparison
# =============================================================================


def compare_all_shoulders(window):

    def get_statistical_uncertainty(value, a, b, c, d):
        da = np.sqrt(a)
        db = np.sqrt(b)
        dc = np.sqrt(c)
        dd = np.sqrt(d)
        dA = np.sqrt(da ** 2 + dc ** 2)
        dB = np.sqrt(db ** 2 + dd ** 2)
        return np.sqrt((dA/(a-c)) ** 2 + (dB/(c-d)) ** 2) * value
    # Declare parameters
    number_bins = int(window.dE_bins.text())
    sigma_interval = [3, 8]
    # Declare all input-paths
    dir_name = os.path.dirname(__file__)
    HR_folder = os.path.join(dir_name, '../../Clusters/MG_new/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder)
                         if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: get_energy(element))
    paths = append_folder_and_files(HR_folder, HR_files_sorted)
    # Get output path
    temp_folder = os.path.join(dir_name, '../../Results/Shoulder/temp/')
    mkdir_p(temp_folder)
    # Calculate He3-solid angle
    __, He3_solid_angle = get_He3_tubes_area_and_solid_angle()
    # Declare vectors where results will be stored
    detectors = ['ILL', 'ESS_CLB', 'ESS_PA']
    FoMs = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    colors = {'ILL': 'blue', 'ESS_CLB': 'red', 'ESS_PA': 'green'}
    energies = []
    errors = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    count = 0
    # Iterate through all data and calculate FoM
    for run in range(2):
        for detector in detectors:
            for i, path in enumerate(paths[8:20]):  # Only look at 2 meV to 30 meV 8->16
                # Import parameters
                E_i = pd.read_hdf(path, 'E_i')['E_i'].iloc[0]
                calibration = (pd.read_hdf(path, 'calibration')
                               ['calibration'].iloc[0])
                df_MG_temp = pd.read_hdf(path, 'coincident_events')
                df_MG = filter_ce_clusters(window, df_MG_temp)
                # Get MG dE-data
                df_MG_detector = detector_filter(df_MG, detector)
                dE_MG = get_MG_dE(df_MG_detector, calibration, E_i)
                # Get He3 dE-data
                dE_He3 = get_raw_He3_dE(calibration, E_i)
                # Produce histograms
                hist_MG, bins = np.histogram(dE_MG, bins=number_bins,
                                             range=[-E_i/5, E_i/5])
                hist_He3, __ = np.histogram(dE_He3, bins=number_bins,
                                            range=[-E_i/5, E_i/5])
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                # Fit data
                p0 = None
                area_MG, FWHM_MG, __, __, __, popt_MG, l_p_idx_MG, r_p_idx_MG, back_MG, MG_back_indices = Gaussian_fit(bin_centers, hist_MG, p0)
                area_He3, FWHM_He3, __, __, __, popt_He3, l_p_idx_He3, r_p_idx_He3, back_He3, He3_back_indices = Gaussian_fit(bin_centers, hist_He3, p0)
                # Normalize data
                norm_peak_area = area_He3/area_MG
                normalized_hist_MG = hist_MG * norm_peak_area
                normalized_MG_back = back_MG * norm_peak_area
                # Compare difference at certain interval
                x0_MG = popt_MG[1]
                sigma_MG = abs(popt_MG[2])
                x0_He3 = popt_He3[1]
                sigma_He3 = abs(popt_He3[2])
                left_idx_MG = find_nearest(bin_centers, x0_MG+sigma_interval[0]*sigma_MG)
                right_idx_MG = find_nearest(bin_centers, x0_MG+sigma_interval[1]*sigma_MG)
                indices_MG = np.arange(left_idx_MG, right_idx_MG+1, 1)
                left_idx_He3 = find_nearest(bin_centers, x0_He3+sigma_interval[0]*sigma_He3)
                right_idx_He3 = find_nearest(bin_centers, x0_He3+sigma_interval[1]*sigma_He3)
                indices_He3 = np.arange(left_idx_He3, right_idx_He3+1, 1)
                area_MG = sum(normalized_hist_MG[indices_MG]) - len(indices_MG) * normalized_MG_back
                area_He3 = sum(hist_He3[indices_He3]) - len(indices_He3) * back_He3
                FoM = area_MG/area_He3
                # Find uncertainty
                error = get_statistical_uncertainty(FoM,
                                                    sum(hist_He3[indices_He3]),
                                                    sum(hist_MG[indices_MG]),
                                                    sum(hist_MG[MG_back_indices]),
                                                    sum(hist_He3[He3_back_indices])
                                                    )
                if run == 0:
                    FoMs[detector].append(FoM)
                    errors[detector].append(error)
                    if detector == 'ILL':
                        energies.append(E_i)
                else:
                    # Plot data
                    
                    fig = plt.figure()
                    #main_title = 'Peak shape comparison $^3$He-tubes and Multi-Grid'
                    #main_title += '\nInterval: ' + str(sigma_interval) + '$\sigma$'
                    #fig.suptitle(main_title, x=0.5, y=1.02)
                    fig.set_figheight(6)
                    fig.set_figwidth(20)
                    plt.subplot(1, 3, 1)
                    plt.plot(bin_centers, normalized_hist_MG, color='red',
                             label='Multi-Grid', zorder=5)
                    plt.plot(bin_centers[MG_back_indices],
                             normalized_hist_MG[MG_back_indices],
                             color='black', label=None, zorder=10)
                    plt.fill_between(bin_centers[indices_MG],
                                     normalized_hist_MG[indices_MG],
                                     np.ones(len(indices_MG))*normalized_MG_back,
                                     facecolor='orange',
                                     label='Shoulder area',
                                     alpha=1, zorder=2
                                     )
                    plt.title(detector + ': ' + calibration)
                    plt.xlabel('$E_i$ - $E_f$ [meV]')
                    plt.ylabel('Normalized counts')
                    plt.yscale('log')
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.legend(loc=1)
                    plt.subplot(1, 3, 2)
                    plt.plot(bin_centers[He3_back_indices],
                             hist_He3[He3_back_indices],
                             color='black', label=None, zorder=6)
                    plt.plot(bin_centers, hist_He3, color='blue',
                             label='$^3$He-tubes', zorder=5)
                    plt.fill_between(bin_centers[indices_He3],
                                     hist_He3[indices_He3],
                                     np.ones(len(indices_He3))*back_He3,
                                     facecolor='purple',
                                     label='Shoulder area',
                                     alpha=1, zorder=2
                                     )
                    plt.title('$^3$He-tubes: ' + calibration)
                    plt.xlabel('$E_i$ - $E_f$ [meV]')
                    plt.ylabel('Normalized counts')
                    plt.yscale('log')
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.legend(loc=1)
                    plt.subplot(1, 3, 3)
                    for detector_temp in detectors:
                        plt.errorbar(energies, FoMs[detector_temp],
                                     errors[detector_temp],
                                     ecolor=colors[detector_temp],
                                     fmt='.',
                                     capsize=5,
                                     color=colors[detector_temp],
                                     label=detector_temp, zorder=5,
                                     linestyle='-'
                                     )
                     #   plt.plot(energies, FoMs[detector_temp], '.-',
                     #            color=colors[detector_temp],
                      #           label=detector_temp, zorder=5)
                    plt.grid(True, which='major', linestyle='--', zorder=0)
                    plt.grid(True, which='minor', linestyle='--', zorder=0)
                    plt.xlabel('$E_i$ [meV]')
                    plt.ylabel('Figure-of-Merit')
                    plt.yscale('log')
                    plt.title('Figure-of-Merit')
                    if ((detector == 'ESS_PA') and (i == 4)):
                        pass
                    else:
                        plt.axvline(x=energies[i], color=colors[detector],
                                    zorder=5)
                    plt.legend(loc=2)
                    # Save fig
                    count += 1
                    output_path = temp_folder + str(count) + '.png'
                    plt.tight_layout()
                    fig.savefig(output_path, bbox_inches='tight')
                    output_path_pdf = os.path.join(dir_name,
                                                   '../../Results/pdf_files/%s.pdf' % count)
                    fig.savefig(output_path_pdf, bbox_inches='tight')
                    plt.close()

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name,
                               '../../Results/Shoulder/FOM_sweep_shoulder.gif')

    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images)  #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)


# =============================================================================
# Peak shape comparison - 3x3, at 2.5 sigma
# =============================================================================

def analyze_lineshape(window):
    def Gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    def fit_data(bin_centers, dE_hist, p0):
        center_idx = len(bin_centers)//2
        zero_idx = find_nearest(bin_centers[center_idx-20:center_idx+20], 0)
        zero_idx += center_idx - 20
        fit_bins = np.arange(zero_idx-20, zero_idx+20+1, 1)
        elastic_peak_max = max(dE_hist[fit_bins])
        popt, __ = scipy.optimize.curve_fit(Gaussian, bin_centers[fit_bins],
                                            dE_hist[fit_bins], p0=p0
                                            )
        sigma = abs(popt[2])
        x0 = popt[1]
        return sigma, x0, elastic_peak_max

    # Declare all input-paths
    dir_name = os.path.dirname(__file__)
    HR_folder = os.path.join(dir_name, '../../Clusters/MG/HR/')
    HR_files = np.array([file for file in os.listdir(HR_folder)
                         if file[-3:] == '.h5'])
    HR_files_sorted = sorted(HR_files, key=lambda element: get_energy(element))
    input_paths = append_folder_and_files(HR_folder, HR_files_sorted)
    # Declare output folder
    output_folder = os.path.join(dir_name, '../../Results/Shoulder/')
    # Declare parameters
    colors = {'ILL': 'blue', 'ESS_CLB': 'red', 'ESS_PA': 'green'}
    p0 = None
    number_bins = 150
    # Declare vectors where results will be stored
    data = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    HR_energies, __ = get_all_energies()
    # Iterate through all calibrations
    for input_path in input_paths:
        # Import parameters
        E_i = pd.read_hdf(input_path, 'E_i')['E_i'].iloc[0]
        calibration = (pd.read_hdf(input_path, 'calibration')
                       ['calibration'].iloc[0])
        df_MG_temp = pd.read_hdf(input_path, 'coincident_events')
        for detector in data.keys():
            # Filter Multi-Grid data
            df_MG = filter_ce_clusters(window, df_MG_temp)
            df_MG_detector = detector_filter(df_MG, detector)
            # Get energy transfer data
            dE_MG = get_MG_dE(df_MG_detector, calibration, E_i)
            dE_He3 = get_raw_He3_dE(calibration, E_i)
            # Produce histograms
            hist_MG, bins = np.histogram(dE_MG, bins=number_bins,
                                         range=[-E_i/5, E_i/5])
            hist_He3, __ = np.histogram(dE_He3, bins=number_bins,
                                        range=[-E_i/5, E_i/5])
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            # Fit data
            sigma_MG, x0_MG, peak_MG = fit_data(bin_centers, hist_MG, p0)
            sigma_He3, x0_He3, peak_He3 = fit_data(bin_centers, hist_He3, p0)
            # Normalize data
            norm = peak_He3/peak_MG
            hist_MG_normalized = hist_MG * norm
            # Get value at 2.5 sigma from the peak center
            idx_MG = find_nearest(bin_centers, x0_MG+2.5*sigma_MG)
            idx_He3 = find_nearest(bin_centers, x0_He3+2.5*sigma_He3)
            value_MG = hist_MG_normalized[idx_MG]
            err_MG = np.sqrt(hist_MG[idx_MG]) * norm
            value_He3 = hist_He3[idx_He3]
            err_He3 = np.sqrt(hist_He3[idx_He3])
            # Plot data
            fig = plt.figure()
            plt.title('Line Shape Investigation')
            plt.plot(bin_centers, hist_MG_normalized, color=colors[detector],
                     label='Value at 2.5σ: %.2f ± %.2f' % (value_MG, err_MG))
            plt.plot(bin_centers, hist_He3, color='black',
                     label='Value at 2.5σ: %.2f ± %.2f' % (value_He3, err_He3))
            plt.xlabel('∆E [meV]')
            plt.yscale('log')
            plt.ylabel('Normalized counts')
            plt.grid(True, which='major', linestyle='--', zorder=0)
            plt.grid(True, which='minor', linestyle='--', zorder=0)
            # Save data
            output_path = output_folder + calibration + detector + '.pdf'
            fig.savefig(output_path, bbox_inches='tight')
            plt.close()
            data[detector].append()













