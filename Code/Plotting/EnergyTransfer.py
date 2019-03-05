import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
    
# =============================================================================
# Helium3 data
# =============================================================================


def plot_He3_data(df, data_sets, calibration, measurement_time, E_i, calcFWHM,
                  vis_help, back_yes, window, isPureAluminium=False,
                  isRaw=False, isFiveByFive=False, useGaussianFit=False,
                  p0=None, isCLB=False, isCorrected=False):

    
    df = detector_filter(df, detector)
    fig = plt.figure()
    MG_SNR = 0
    He3_SNR = 0
    
    # Import He3 data
    measurement_id = find_He3_measurement_id(calibration)
    dirname = os.path.dirname(__file__)
    folder = '../Archive/2018_06_06_SEQ_He3/'
    nxs_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxs'
    nxspe_file = 'SEQ_' + str(measurement_id) + '_autoreduced.nxspe'
    nxs_path = os.path.join(dirname, folder + nxs_file)
    nxspe_path = os.path.join(dirname, folder + nxspe_file)
    nxs = h5py.File(nxs_path, 'r')
    nxspe = h5py.File(nxspe_path, 'r')
    # Extract values
    he3_bins = nxs['mantid_workspace_1']['event_workspace']['axis1'].value
    he3_min = he3_bins[0]
    he3_max = he3_bins[-1]
    He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
    He3_dE_hist = None
    if isRaw:
        df_He3 = load_He3_h5(calibration)
        T_0_raw = get_T0(calibration, E_i) * np.ones(len(df_He3.ToF))
        t_off_raw = get_t_off_He3(calibration) * np.ones(len(df_He3.ToF))
        E_i_raw = E_i * np.ones(len(df_He3.ToF))
        dE_raw, t_f_raw = get_dE_He3(E_i_raw, df_He3.ToF, df_He3.distance, T_0_raw, t_off_raw)
        df_temp_raw = pd.DataFrame(data={'dE': dE_raw, 't_f': t_f_raw})
        dE_raw = df_temp_raw[df_temp_raw['t_f'] > 0].dE
        He3_dE_hist, __ = np.histogram(dE_raw, bins=390, range=[he3_min, he3_max])


    elif isCorrected:
        corrected_path = ''
        uncorrected_path = ''
        if calibration[0:30] == 'Van__3x3_High_Flux_Calibration':
            corrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_trueHighFlux/')
            file_name = 'van_eff_trueHighFlux_' + str(int(E_i)) + 'p00.nxspe'
            corrected_path = corrected_folder + file_name
            uncorrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_falseHighFlux/')
            uncorrected_file = 'van_eff_falseHighFlux_' + str(int(E_i)) + 'p00.nxspe'
            uncorrected_path = uncorrected_folder + uncorrected_file
        else:
            corrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_true/')
            file_name = 'van_eff_true_' + str(int(E_i)) + 'p00.nxspe'
            corrected_path = corrected_folder + file_name
            uncorrected_folder = os.path.join(dirname, '../Archive/2019_02_04_3He_eff_correction/van_eff_false/')
            uncorrected_file = 'van_eff_false_' + str(int(E_i)) + 'p00.nxspe'
            uncorrected_path = uncorrected_folder + uncorrected_file
        corrected_He3 = h5py.File(corrected_path, 'r')
        uncorrected_He3 = h5py.File(uncorrected_path, 'r')
        corrected_data = corrected_He3['data']['data']['data']
        uncorrected_data = uncorrected_He3['data']['data']['data']
        he3_bins = corrected_He3['data']['data']['energy']
        he3_min = he3_bins[0]
        he3_max = he3_bins[-1]
        He3_bin_centers = 0.5 * (he3_bins[1:] + he3_bins[:-1])
        #print(corrected_data)
        #print(He3_bin_centers)
        He3_dE_hist = np.zeros(len(He3_bin_centers), dtype='float')
        He3_dE_hist_uncorr = np.zeros(len(He3_bin_centers), dtype='float')
        for i, row in enumerate(corrected_data):
            if np.isnan(np.array(row)).any():
                print(i)
            else:
                He3_dE_hist += np.array(row)

        for i, row in enumerate(uncorrected_data):
            if np.isnan(np.array(row)).any():
                pass
            else:
                He3_dE_hist_uncorr += np.array(row)



    else:
        dE = nxs['mantid_workspace_1']['event_workspace']['tof'].value
        He3_dE_hist, __ = np.histogram(dE, bins=390, range=[he3_min, he3_max])
    
    He_duration = nxs['mantid_workspace_1']['logs']['duration']['value'].value[0]
    
    # Calculate MG spectrum
    df = df[df.d != -1]
    t_off = np.ones(df.shape[0]) * get_t_off(calibration)
    T_0 = np.ones(df.shape[0]) * get_T0(calibration, E_i)
    frame_shift = np.ones(df.shape[0]) * get_frame_shift(E_i)
    E_i = np.ones(df.shape[0]) * E_i #get_Ei(measurement_id)
    ToF = df.ToF.values
    d = df.d.values
    dE, t_f = get_dE(E_i, ToF, d, T_0, t_off, frame_shift) 
    df_temp = pd.DataFrame(data={'dE': dE, 't_f': t_f})
    dE = df_temp[df_temp['t_f'] > 0].dE
    # Get MG dE histogram
    dE_bins = len(He3_bin_centers)
    dE_range = [he3_min, he3_max]
    MG_dE_hist, MG_bins = np.histogram(dE, bins=dE_bins, range=dE_range)
    MG_bin_centers = 0.5 * (MG_bins[1:] + MG_bins[:-1])
    # Get He3 offset
    He3_offset = 0
    if isRaw is False:
        He3_offset = get_He3_offset(calibration)
    # Use Gaussian fitting procedure instead, if selected
    if useGaussianFit:
        norm_MG, MG_FWHM, MG_gaussian, MG_fit_x, MG_MAX, p0, MG_left, MG_right = Gaussian_fit(MG_bin_centers,
                                                                                      		  MG_dE_hist, p0)
        norm_He3, He3_FWHM, He3_gaussian, He3_fit_x, He3_MAX, p0, He3_left, He3_right = Gaussian_fit(He3_bin_centers+He3_offset,
                                                                                            		 He3_dE_hist, p0)
    else:
        # Get MG and He3 peak edges
        MG_left, MG_right, He3_left, He3_right = get_peak_edges(calibration)
        if isRaw:
            MG_left, MG_right, He3_left, He3_right = get_peak_edges_raw(calibration)
        # Get MG and He3 normalisation
        norm_MG = calculate_peak_norm(MG_bin_centers, MG_dE_hist, MG_left,
                                      MG_right)
        norm_He3 = calculate_peak_norm(He3_bin_centers, He3_dE_hist, He3_left,
                                       He3_right)

    
    # Calculate total normalization
    area_time_norm = get_area_time_norm(calibration, E_i[0], isCLB,
                                        isPureAluminium,
                                        measurement_time, He_duration)
    proton_solid_norm = get_charge_solid_norm(calibration, E_i[0], isCLB,
                                              isPureAluminium)
    
    

    if isFiveByFive:
        tot_norm = sum(MG_dE_hist)/sum(He3_dE_hist)
    
    # Plot background level
    hist_back = plot_dE_background(E_i[0], calibration, measurement_time,
                                   area_time_norm, he3_min, he3_max, back_yes,
                                   area_time_norm, window, isCLB=isCLB,
                                   isPureAluminium=isPureAluminium,
                                   numberBins=dE_bins)

    # Plot MG and He3
    if vis_help is not True:
        MG_dE_hist = MG_dE_hist/proton_solid_norm
        He3_dE_hist = He3_dE_hist

    if back_yes:
        pass
    else:
        hist_back = plot_dE_background(E_i[0], calibration, measurement_time,
                                       proton_solid_norm, he3_min, he3_max, back_yes,
                                       proton_solid_norm, window, isCLB=isCLB,
                                       isPureAluminium=isPureAluminium,
                                       numberBins=dE_bins)

        MG_dE_hist = MG_dE_hist - hist_back
        
    if isRaw:
        plt.plot(MG_bin_centers, MG_dE_hist, label='Multi-Grid', color='red')
        plt.plot(He3_bin_centers, He3_dE_hist, label='$^3$He-tubes', color='blue')
        if back_yes:
            plt.plot(MG_bin_centers, hist_back, color='green', label='Background estimation', zorder=5)
    else:
        plt.plot(MG_bin_centers, MG_dE_hist, label='Multi-Grid', color='red', zorder=10)
        plt.plot(He3_bin_centers+He3_offset, He3_dE_hist, label='$^3$He tubes', color='blue')
        plt.plot(He3_bin_centers+He3_offset, He3_dE_hist_uncorr, label='$^3$He tubes (uncorrected)', color='orange')
        if back_yes:
        	plt.plot(MG_bin_centers, hist_back, color='green', label='MG background', zorder=5)
    
    
    # Calculate FWHM
    if calcFWHM:
        if useGaussianFit:
            MG_FWHM = str(round(MG_FWHM, 4))
            He3_FWHM = str(round(He3_FWHM, 4))
        else:
            MG_FWHM, MG_SNR, MG_MAX = get_FWHM(MG_bin_centers, MG_dE_hist, MG_left, MG_right,
                                               vis_help, b_label='Background')
            MG_FWHM = str(round(MG_FWHM, 4))
            He3_FWHM, He3_SNR, He3_MAX = get_FWHM(He3_bin_centers, He3_dE_hist, He3_left,
                                                  He3_right, vis_help, b_label=None)
            He3_FWHM = str(round(He3_FWHM, 4))

    MG_peak_normalised = norm_MG/proton_solid_norm
    He3_peak_normalised = norm_He3
    MG_over_He3 = round(MG_peak_normalised/He3_peak_normalised, 4)
    MG_over_He3_max = 0
    if calcFWHM:
        MG_over_He3_max = round((MG_MAX/He3_MAX), 4)
    # Plot text box
    text_string = r"$\bf{" + '---MultiGrid---' + "}$" + '\n'
    text_string += 'Area: ' + str(round(MG_peak_normalised, 1)) + ' [counts]\n'
    text_string += 'FWHM: ' + MG_FWHM + ' [meV]\n'
    text_string += 'Duration: ' + str(round(measurement_time, 1)) + ' [s]\n'
    text_string += r"$\bf{" + '---He3---' + "}$" + '\n'
    text_string += 'Area: ' + str(round(He3_peak_normalised, 1)) + ' [counts]\n'
    text_string += 'FWHM: ' + He3_FWHM + ' [meV]\n'
    text_string += 'Duration: ' + str(round(He_duration, 1)) + '  [s]\n'
    text_string += r"$\bf{" + '---Comparison---' + "}$" + '\n'
    text_string += 'Area fraction: ' + str(MG_over_He3)

    #nearest_temp = find_nearest(He3_bin_centers+He3_offset, 0)
    #zero_val_He3_corr   = He3_dE_hist[nearest_temp]
    #zero_val_He3_uncorr = He3_dE_hist_uncorr[nearest_temp]

    #plt.plot([He3_bin_centers[nearest_temp], He3_bin_centers[nearest_temp]],
    #		 [zero_val_He3_corr, zero_val_He3_uncorr])

    #text_string += '\nEfficiency correction: %f' % round(zero_val_He3_corr/zero_val_He3_uncorr, 4)
    
    He3_hist_max = max(He3_dE_hist)
    MG_hist_max = max(MG_dE_hist)
    tot_max = max([He3_hist_max, MG_hist_max])
    plt.text(-0.7*E_i[0], tot_max * 0.07, text_string, ha='center', va='center', 
                 bbox={'facecolor':'white', 'alpha':0.9, 'pad':10}, fontsize=6,
                 zorder=50)
    
    # Visualize peak edges
    if useGaussianFit:
        plt.plot(MG_fit_x, MG_gaussian/proton_solid_norm, color='purple')
        plt.plot(He3_fit_x, He3_gaussian, color='orange')
        
        plt.plot([He3_bin_centers[He3_left]+He3_offset,
                  He3_bin_centers[He3_right]+He3_offset],
                 [He3_dE_hist[He3_left], He3_dE_hist[He3_right]], '-x',
                  color='black' ,
                  label='Peak edges', zorder=20)
        plt.plot([MG_bin_centers[MG_left], MG_bin_centers[MG_right]], 
                 [MG_dE_hist[MG_left], MG_dE_hist[MG_right]], '-x', 
                  color='black',
                  label=None, zorder=20)
        #pass
                
    plt.legend(loc='upper right').set_zorder(10)
    plt.yscale('log')
    plt.xlabel('E$_i$ - E$_f$ [meV]')
    plt.xlim(he3_min, he3_max)
    plt.ylim(1, 1.5*He3_hist_max)
    plt.ylabel('Normalized Counts')
    title = calibration + '_meV' 
    if back_yes is not True:
        title += '\n(Background subtracted)'
    if isPureAluminium:
        title += '\nPure Aluminium'
    if isFiveByFive:
        title += '\n(Van__5x5 sample for Multi-Grid)'
    if isCLB:
        title += '\nESS.CLB'
    plt.title(title)
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)


    # Export histograms to text-files
    if back_yes == True:
        dir_name = os.path.dirname(__file__)
        MG_path = os.path.join(dir_name, '../Results/Histograms/MG/dE/MG_%s_meV.txt' % calibration)
        He3_path = os.path.join(dir_name, '../Results/Histograms/He3/dE/He3_%s_meV.txt' % calibration)
        MG_dict = {'dE [meV]': MG_bin_centers,
                   'Signal [Normalized Counts]': MG_dE_hist,
                   'Background estimation [Normalized counts]': hist_back
                    }
        He3_dict = {'dE [meV]': He3_bin_centers,
                   'Signal [Normalized Counts]': He3_dE_hist
                    }
        MG_df = pd.DataFrame(MG_dict)
        He3_df = pd.DataFrame(He3_dict)
        MG_df.to_csv(MG_path, index=None, sep=' ', mode='w', encoding='ascii')
        He3_df.to_csv(He3_path, index=None, sep=' ', mode='w', encoding='ascii')
    return fig, MG_over_He3, MG_over_He3_max, He3_FWHM, MG_FWHM, p0