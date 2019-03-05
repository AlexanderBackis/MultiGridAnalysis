import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import imageio
from Plotting.HelperFunctions import stylize, filter_ce_clusters, get_MG_dE
from Plotting.HelperFunctions import get_raw_He3_dE, get_calibration
from Plotting.HelperFunctions import Gaussian_fit, find_nearest
from Plotting.HelperFunctions import detector_filter, mkdir_p
    
# =============================================================================
# Peak shape comparison
# =============================================================================


def compare_all_shoulders(window):
    # Declare parameters
    number_bins = 150
    sigma_interval = [3, 8]
    detectors = ['ILL', 'ESS_CLB', 'ESS_PA']
    FoMs = {'ILL': [], 'ESS_CLB': [], 'ESS_PA': []}
    colors = {'ILL': 'red', 'ESS_CLB': 'blue', 'ESS_PA': 'green'}
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
                # Compare difference at certain interval
                x0_MG = popt_MG[1]
                sigma_MG = abs(popt_MG[2])
                x0_He3 = popt_He3[1]
                sigma_He3 = abs(popt_He3[2])
                print(sigma_MG)
                print(sigma_He3)
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
                    fig.savefig(output_path_pdf, bbox_inches='tight')

    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' 
             and file != '.gitignore']
    
    output_path = os.path.join(dir_name, '../../Results/Shoulder/FOM_sweep_shoulder.gif')
    
    for filename in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + filename + '.png'))
    imageio.mimsave(output_path, images)  #format='GIF', duration=0.33)
    shutil.rmtree(temp_folder, ignore_errors=True)
    window.update()
    window.app.processEvents()






