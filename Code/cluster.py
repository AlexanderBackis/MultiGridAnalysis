#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:18:00 2018

@author: alexanderbackis
"""

# =======  LIBRARIES  ======= #
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import struct
import re
import zipfile
import shutil

# =======    MASKS    ======= #
TypeMask      =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000
DataMask      =   0xF0000000     # 1111 0000 0000 0000 0000 0000 0000 0000

ChannelMask   =   0x00FFF000     # 0000 0000 1111 1111 1111 0000 0000 0000
BusMask       =   0x0F000000     # 0000 1111 0000 0000 0000 0000 0000 0000
ADCMask       =   0x00000FFF     # 0000 0000 0000 0000 0000 1111 1111 1111
TimeStampMask =   0x3FFFFFFF     # 0011 1111 1111 1111 1111 1111 1111 1111
NbrWordsMask  =   0x00000FFF     # 0000 0000 0000 0000 0000 1111 1111 1111
GateStartMask =   0x0000FFFF     # 0000 0000 0000 0000 1111 1111 1111 1111
ExTsMask      =   0x0000FFFF     # 0000 0000 0000 0000 1111 1111 1111 1111
TriggerMask   =   0xCF000000     # 1100 1111 0000 0000 0000 0000 0000 0000


# =======  DICTONARY  ======= #
Header        =   0x40000000     # 0100 0000 0000 0000 0000 0000 0000 0000
Data          =   0x00000000     # 0000 0000 0000 0000 0000 0000 0000 0000
EoE           =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000

DataBusStart  =   0x30000000     # 0011 0000 0000 0000 0000 0000 0000 0000
DataEvent     =   0x10000000     # 0001 0000 0000 0000 0000 0000 0000 0000
DataExTs      =   0x20000000     # 0010 0000 0000 0000 0000 0000 0000 0000

Trigger       =   0x41000000     # 0100 0001 0000 0000 0000 0000 0000 0000

# =======  BIT-SHIFTS  ======= #
ChannelShift  =   12
BusShift      =   24
ExTsShift     =   30

# =============================================================================
#                                IMPORT DATA
# =============================================================================

def import_data(file_path, max_size=np.inf):
    """ Goes to sister-folder '/Data/' and imports '.mesytec'-file with name
        'file_name'. Does this in three steps:

            1. Reads file as binary and saves data in 'content'
            2. Finds the end of the configuration text, i.e. '}\n}\n' followed
               by 0 to n spaces, then saves everything after this to
               'reduced_content'.
            3. Groups data into 'uint'-words of 4 bytes (32 bits) length

    Args:
        file_name (str): Name of '.mesytec'-file that contains the data

    Returns:
        data (tuple): A tuple where each element is a 32 bit mesytec word

    """

    with open(file_path, mode='rb') as bin_file:
        piece_size = 1000
        if max_size < 1000:
            piece_size = max_size

        content = bin_file.read(piece_size * (1 << 20))

        # Skip configuration text
        match = re.search(b'}\n}\n[ ]*', content)
        start = match.end()
        content = content[start:]

        data = struct.unpack('I' * (len(content)//4), content)

        # Import data
        moreData = True
        imported_data = piece_size
        while moreData and imported_data <= max_size:
            imported_data += piece_size
            piece = bin_file.read(piece_size * (1 << 20))
            if not piece:  # Reached end of file
                moreData = False
            else:
                data += struct.unpack('I' * (len(piece)//4), piece)
    return data


# =============================================================================
#                               CLUSTER DATA
# =============================================================================

def cluster_data(data, ADC_threshold=0, ILL_buses=[], progressBar=None,
                 dialog=None, app=None):
    """ Clusters the imported data and stores it two data frames: one for
        individual events and one for coicident events (i.e. candidate neutron
        events).

        Does this in the following fashion for coincident events:
            1. Reads one word at a time
            2. Checks what type of word it is (Header, BusStart, DataEvent,
               DataExTs or EoE).
            3. When a Header is encountered, 'isOpen' is set to 'True',
               signifying that a new event has been started. Data is then
               gathered into a single coincident event until a different bus is
               encountered (unless ILL exception), in which case a new event is
               started.
            4. When EoE is encountered the event is formed, and timestamp is
               assigned to it and all the created events under the current
               Header. This event is placed in the created dictionary.
            5. After the iteration through data is complete, the dictionary
               containing the coincident events is convereted to a DataFrame.

        And for events:
            1-2. Same as above.
            3. Every time a data word is encountered it is added as a new event
               in the intitally created dicitionary.
            4-5. Same as above

    Args:
        data (tuple)    : Tuple containing data, one word per element.
        ILL_buses (list): List containg all ILL buses

    Returns:
        data (tuple): A tuple where each element is a 32 bit mesytec word

        events_df (DataFrame): DataFrame containing one event (wire or grid)
                               per row. Each event has information about:
                               "Bus", "Time", "Channel", "ADC".

        coincident_events_df (DataFrame): DataFrame containing one neutron
                                          event per row. Each neutron event has
                                          information about: "Bus", "Time",
                                          "ToF", "wCh", "gCh", "wADC", "gADC",
                                          "wM", "gM", "Coordinate".



    """
    offset_1 = {'x': -0.907574, 'y': -3.162949, 'z': 5.384863}
    offset_2 = {'x': -1.246560, 'y': -3.161484, 'z': 5.317432}
    offset_3 = {'x': -1.579114, 'y': -3.164503,  'z': 5.227986}

    corners = {'ESS_2': {1: [-1.579114, -3.164503, 5.227986],
                         2: [-1.252877, -3.162614, 5.314108]},
               'ESS_1': {3: [-1.246560, -3.161484, 5.317432],
                         4: [-0.916552, -3.160360, 5.384307]},
               'ILL':   {5: [-0.907574, -3.162949, 5.384863],
                         6: [-0.575025, -3.162578, 5.430037]}
                }

    ILL_C = corners['ILL']
    ESS_1_C = corners['ESS_1']
    ESS_2_C = corners['ESS_2']

    theta_1 = np.arctan((ILL_C[6][2]-ILL_C[5][2])/(ILL_C[6][0]-ILL_C[5][0]))
    theta_2 = np.arctan((ESS_1_C[4][2]-ESS_1_C[3][2])/(ESS_1_C[4][0]-ESS_1_C[3][0]))
    theta_3 = np.arctan((ESS_2_C[2][2]-ESS_2_C[1][2])/(ESS_2_C[2][0]-ESS_2_C[1][0]))

    detector_1 = create_ill_channel_to_coordinate_map(theta_1, offset_1)
    detector_2 = create_ess_channel_to_coordinate_map(theta_2, offset_2)
    detector_3 = create_ess_channel_to_coordinate_map(theta_3, offset_3)

    detector_vec = [detector_1, detector_2, detector_3]

    size = len(data)
    coincident_event_parameters = ['Bus', 'Time', 'ToF', 'wCh', 'gCh',
                                   'wADC', 'gADC', 'wM', 'gM', 'ceM']
    coincident_events = create_dict(size, coincident_event_parameters)
    coincident_events.update({'d': np.zeros([size], dtype=float)})

    event_parameters = ['Bus', 'Time', 'Channel', 'ADC']
    events = create_dict(size, event_parameters)
    triggers = np.empty([size], dtype=int)
    #Declare variables
    TriggerTime = 0
    index = -1
    index_event = -1
    trigger_index = 0
    #Declare temporary variables
    isOpen = False
    isData = False
    isTrigger = False
    Bus = -1
    previousBus = -1
    maxADCw = 0
    maxADCg = 0
    nbrCoincidentEvents = 0
    nbrEvents = 0
    Time = 0
    extended_time_stamp = None
    number_words = len(data)
    #Five possibilities in each word: Header, DataBusStart, DataEvent,
    #DataExTs or EoE.
    for count, word in enumerate(data):
        if (word & TypeMask) == Header:
        #    print('Header')
            isOpen = True
            isTrigger = (word & TriggerMask) == Trigger
        elif ((word & DataMask) == DataBusStart) & isOpen:
        #    print('DataBusStart')
            Bus = (word & BusMask) >> BusShift
            isData = True
            if (previousBus in ILL_buses) and (Bus in ILL_buses):
                pass
            else:
                previousBus = Bus
                maxADCw = 0
                maxADCg = 0
                nbrCoincidentEvents += 1
                nbrEvents += 1
                index += 1

                coincident_events['wCh'][index] = -1
                coincident_events['gCh'][index] = -1
                coincident_events['Bus'][index] = Bus
        elif ((word & DataMask) == DataEvent) & isOpen:
            Channel = ((word & ChannelMask) >> ChannelShift)
            ADC = (word & ADCMask)
            if ADC > ADC_threshold:
                index_event += 1
                nbrEvents += 1
                events['Bus'][index_event] = Bus
                events['ADC'][index_event] = ADC
                if Channel >= 120:
                    pass
                elif Channel < 80:
                    coincident_events['Bus'][index] = Bus              # Remove if trigger is on wire
                    coincident_events['wADC'][index] += ADC
                    coincident_events['wM'][index] += 1
                    if ADC > maxADCw:
                        coincident_events['wCh'][index] = Channel ^ 1  # Shift odd and even Ch
                        maxADCw = ADC
                    events['Channel'][index_event] = Channel ^ 1       # Shift odd and even Ch
                    #print('Channel: %d' % (Channel ^ 1))
                else:
                    coincident_events['gADC'][index] += ADC
                    coincident_events['gM'][index] += 1
                    if ADC > maxADCg:
                        coincident_events['gCh'][index] = Channel
                        maxADCg = ADC
                    events['Channel'][index_event] = Channel
                    #print('Channel: %d' % Channel)
        elif ((word & DataMask) == DataExTs) & isOpen:
        #    print('DataExTs')
            extended_time_stamp = (word & ExTsMask) << ExTsShift
        elif ((word & TypeMask) == EoE) & isOpen:
        #    print('EoE')
            time_stamp = (word & TimeStampMask)
            if extended_time_stamp is not None:
                Time = extended_time_stamp | time_stamp
            else:
                Time = time_stamp

            if isTrigger:
                TriggerTime = Time
                triggers[trigger_index] = TriggerTime
                trigger_index += 1
            #Assign timestamp to coindicent events
            ToF = Time - TriggerTime
            for i in range(0, nbrCoincidentEvents):
                coincident_events['Time'][index-i] = Time
                coincident_events['ToF'][index-i] = ToF
            #Assign timestamp to events
            for i in range(0, nbrEvents):
                events['Time'][index_event-i] = Time
            #Assign d
            for i in range(0, nbrCoincidentEvents):
                wCh = coincident_events['wCh'][index-i]
                gCh = coincident_events['gCh'][index-i]
                coincident_events['ceM'][index-i] = nbrCoincidentEvents
                if (wCh != -1 and gCh != -1):
                    eventBus = coincident_events['Bus'][index]
                    ToF = coincident_events['ToF'][index-i]
                    d = get_d(eventBus, wCh, gCh, detector_vec)
                    coincident_events['d'][index-i] = d
                else:
                    coincident_events['d'][index-i] = -1



            #print('Time: %d' % Time)
            #print('ToF: %d' % ToF)
            #print()
            #Reset temporary variables
            nbrCoincidentEvents = 0
            nbrEvents = 0
            Bus = -1
            previousBus = -1
            isOpen = False
            isData = False
            isTrigger = False
            Time = 0
            length = count

        if count % 1000000 == 1:
            percentage_finished = round((count/number_words)*100)
            if progressBar is not None:
                progressBar.setValue(percentage_finished)
                dialog.update()
                app.processEvents()

    if percentage_finished != '100%':
        if progressBar is not None:
            progressBar.setValue(100)
            dialog.update()
            app.processEvents()

    #Remove empty elements and save in DataFrame for easier analysis
    for key in coincident_events:
        coincident_events[key] = coincident_events[key][0:index]
    coincident_events_df = pd.DataFrame(coincident_events)

    for key in events:
        events[key] = events[key][0:index_event]
    events_df = pd.DataFrame(events)

    triggers_df = None
    if trigger_index == 0:
        triggers_df = pd.DataFrame([0])
    else:
        triggers_df = pd.DataFrame(triggers[0:trigger_index-1])

    print('Coincident events')
    print(coincident_events_df)

    return coincident_events_df, events_df, triggers_df


def filter_data(ce_temp, e_temp, t_temp, discard_glitch, keep_only_ce,
                glitch_mid=0.5):
    ce = ce_temp
    e = e_temp
    t = t_temp
    measurement_time = 0

    if discard_glitch:
        ce_red = ce[(ce['wM'] >= 80) & (ce['gM'] >= 40)]
        if len(ce_red.index) == 0:
            start_time = ce_temp.head(1)['Time'].values[0]
            end_time = ce_temp.tail(1)['Time'].values[0]
            measurement_time = (end_time - start_time) * 62.5e-9
        else:
            # Filter glitch events in begining and end by diving data
            # in two parts (f: first, s: second) and finding first
            # and last glitch event.
            mid_point = ce.tail(1)['Time'].values[0] * glitch_mid
            ce_f = ce_temp[(ce_temp['Time'] < mid_point)]
            ce_s = ce_temp[(ce_temp['Time'] > mid_point)]
            ce_red_f = ce_f[(ce_f['wM'] >= 80) & (ce_f['gM'] >= 40)]
            ce_red_s = ce_s[(ce_s['wM'] >= 80) & (ce_s['gM'] >= 40)]

            data_start = None
            data_end = None
            if ce_red_f.shape[0] > 0:
                data_start = ce_red_f.tail(1)['Time'].values[0]
            else:
                data_start = ce.head(1)['Time'].values[0]

            if ce_red_s.shape[0] > 0:
                data_end = ce_red_s.head(1)['Time'].values[0]
            else:
                data_end = ce.tail(1)['Time'].values[0]

            ce = ce_temp[  (ce_temp['Time'] > data_start)
                         & (ce_temp['Time'] < data_end)]

            e = e_temp[  (e_temp['Time'] > data_start)
                        & (e_temp['Time'] < data_end)]

            if ce.shape[0] > 0:
                start_time = ce.head(1)['Time'].values[0]
                end_time = ce.tail(1)['Time'].values[0]
                measurement_time = (end_time - start_time) * 62.5e-9

    else:
        start_time = ce_temp.head(1)['Time'].values[0]
        end_time = ce_temp.tail(1)['Time'].values[0]
        measurement_time = (end_time - start_time) * 62.5e-9

    if keep_only_ce:
        e = pd.DataFrame()

    return ce, e, t, measurement_time


def unzip_data(zip_source):
    dirname = os.path.dirname(__file__)
    zip_temp_folder = os.path.join(dirname, '../zip_temp_folder/')
    mkdir_p(zip_temp_folder)
    file_temp_folder = os.path.join(dirname, '../')
    destination = ''
    with zipfile.ZipFile(zip_source, "r") as zip_ref:
        zip_ref.extractall(zip_temp_folder)
        temp_list = os.listdir(zip_temp_folder)
        source_file = None
        for temp_file in temp_list:
            if temp_file[-8:] == '.mvmelst':
                source_file = temp_file
        source = zip_temp_folder + source_file
        destination = file_temp_folder + source_file
        shutil.move(source, destination)
        shutil.rmtree(zip_temp_folder, ignore_errors=True)
    return destination


def load_data(clusters_path, window):
    window.load_progress.setValue(0)
    window.load_progress.show()
    # Load coincident events
    ce = pd.read_hdf(clusters_path, 'coincident_events')
    print(ce)
    window.load_progress.setValue(25)
    window.update()
    window.app.processEvents()
    # Load events
    e = pd.read_hdf(clusters_path, 'events')
    window.load_progress.setValue(50)
    window.update()
    window.app.processEvents()
    # Load Triggers
    window.Triggers = pd.read_hdf(clusters_path, 'triggers')
    window.load_progress.setValue(75)
    window.update()
    window.app.processEvents()
    # Load measurement time
    measurement_time = pd.read_hdf(clusters_path, 'measurement_time')['measurement_time'].iloc[0]
    # Write or append
    if window.write.isChecked():
        window.Coincident_events = ce
        window.Events = e
        window.measurement_time = measurement_time
    else:
        window.Coincident_events = window.Coincident_events.append(ce)
        window.Events = window.Events.append(e)
        window.measurement_time += measurement_time
    # Assign rest of parameters
    window.number_of_detectors = pd.read_hdf(clusters_path, 'number_of_detectors')['number_of_detectors'].iloc[0]
    module_order_df = pd.read_hdf(clusters_path, 'module_order')
    detector_types_df = pd.read_hdf(clusters_path, 'detector_types')
    detector_types = []
    for row in detector_types_df['detector_types']:
        detector_types.append(row)
    module_order = []
    for row in module_order_df['module_order']:
        module_order.append(row)
    window.detector_types = detector_types
    window.module_order = module_order
    window.E_i = pd.read_hdf(clusters_path, 'E_i')['E_i'].iloc[0]
    window.data_sets = pd.read_hdf(clusters_path, 'data_set')['data_set'].iloc[0]
    window.calibration = pd.read_hdf(clusters_path, 'calibration')['calibration'].iloc[0]
    window.load_progress.setValue(100)
    window.update()
    window.app.processEvents()
    window.load_progress.close()
    window.update()
    window.app.processEvents()


def save_data(coincident_events, events, triggers, number_of_detectors,
              module_order, detector_types, data_set, measurement_time,
              E_i, calibration, window, path):
    window.save_progress.setValue(0)
    window.save_progress.show()
    window.update()
    window.app.processEvents()

    coincident_events.to_hdf(path, 'coincident_events', complevel=9)
    window.save_progress.setValue(25)
    window.update()
    window.app.processEvents()
    events.to_hdf(path, 'events', complevel=9)
    window.save_progress.setValue(50)
    window.update()
    window.app.processEvents()
    triggers.to_hdf(path, 'triggers', complevel=9)
    window.save_progress.setValue(75)
    window.update()
    window.app.processEvents()

    number_det = pd.DataFrame({'number_of_detectors': [number_of_detectors]})
    mod_or     = pd.DataFrame({'module_order': module_order})
    det_types  = pd.DataFrame({'detector_types': detector_types})
    da_set     = pd.DataFrame({'data_set': [data_set]})
    mt         = pd.DataFrame({'measurement_time': [measurement_time]})
    ca         = pd.DataFrame({'calibration': [calibration]})
    ei = pd.DataFrame({'E_i': [E_i]})

    number_det.to_hdf(path, 'number_of_detectors', complevel=9)
    mod_or.to_hdf(path, 'module_order', complevel=9)
    det_types.to_hdf(path, 'detector_types', complevel=9)
    da_set.to_hdf(path, 'data_set', complevel=9)
    mt.to_hdf(path, 'measurement_time', complevel=9)
    ei.to_hdf(path, 'E_i', complevel=9)
    ca.to_hdf(path, 'calibration', complevel=9)
    window.save_progress.setValue(100)
    window.update()
    window.app.processEvents()
    window.save_progress.close()


def cluster_and_save_all_MG_data():
    # Declare parameters
    module_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    detector_types = ['ILL', 'ESS', 'ESS']
    number_of_detectors = 3
    glitch_measurements = ['Van__3x3_High_Resolution_Calibration_2.0',
                           'Van__3x3_High_Resolution_Calibration_3.0',
                           'Van__3x3_High_Resolution_Calibration_4.0',
                           'Van__3x3_High_Resolution_Calibration_8.0',
                           'Van__3x3_High_Resolution_Calibration_18.0',
                           'Van__3x3_High_Resolution_Calibration_20.0',
                           'Van__3x3_High_Resolution_Calibration_70.0',
                           'Van__3x3_High_Resolution_Calibration_120.0',
                           'Van__3x3_High_Resolution_Calibration_160.0',
                           'Van__3x3_High_Resolution_Calibration_200.0',
                           'Van__3x3_High_Resolution_Calibration_250.0',
                           'Van__3x3_High_Flux_Calibration_25.0',
                           'Van__3x3_High_Flux_Calibration_34.0',
                           'Van__3x3_High_Flux_Calibration_48.0',
                           'Van__3x3_High_Flux_Calibration_60.0',
                           'Van__3x3_High_Flux_Calibration_70.0',
                           'Van__3x3_High_Flux_Calibration_140.0',
                           'Van__3x3_High_Flux_Calibration_300.0',
                           'Van__3x3_High_Flux_Calibration_400.0',
                           'Van__3x3_High_Flux_Calibration_450.0',
                           'Van__3x3_High_Flux_Calibration_700.0',
                           'Van__3x3_High_Flux_Calibration_800.0',
                           ]


    # Get all file paths
    dir_name = os.path.dirname(__file__)
    folder_1 = os.path.join(dir_name, '../Archive/V_3x3_HR_and_HF/')
    folder_2 = os.path.join(dir_name, '../Archive/V3x3_remainder_HR_HF/')
    files_1 = np.array([file for file in os.listdir(folder_1) if file[-4:] == '.zip'])
    file_paths_1 = np.core.defchararray.add(np.array(len(files_1)*[folder_1]), files_1)
    files_2 = np.array([file for file in os.listdir(folder_2) if file[-4:] == '.zip'])
    file_paths_2 = np.core.defchararray.add(np.array(len(files_2)*[folder_2]), files_2)
    file_paths = np.concatenate((file_paths_1, file_paths_2), axis=None)
    # Start clustering of all files
    calibration_previous = 'Van__3x3_High_Resolution_Calibration_2.0'
    E_i_previous = 2
    ce = pd.DataFrame()
    e  = pd.DataFrame()
    t  = pd.DataFrame()
    measurement_time = 0
    for i, file_path in enumerate(file_paths):
        print('%d/%d)' % (i, len(file_paths)))
        # Unzip data
        mesytec_file_path = unzip_data(file_path)
        # Import data from mesytec file
        data = import_data(mesytec_file_path)
        # Remove the temporary '.mesytec'-file
        os.remove(mesytec_file_path)
        # Find calibration
        name = file_path.rsplit('/', 1)[-1]
        if 'HR' in name:
            setting = 'High_Resolution'
        else:
            setting = 'High_Flux'
        energy = round(float(name[name.find('V_')+2:name.find('meV')]), 1)
        if energy == 41.0:
            energy = 40.8
        calibration = 'Van__3x3_%s_Calibration_%.1f' % (setting, energy)
        if calibration == calibration_previous:
            ce, e, t, measurement_time = cluster_and_append(data, ce, e, t, measurement_time, calibration, glitch_measurements)
        else:
            print('Saving...')
            # Save current clusters
            path = os.path.join(dir_name, '../Clusters/MG/%s_meV.h5' % calibration_previous)
            save(ce, e, t, measurement_time, calibration_previous, number_of_detectors,
                 module_order, detector_types, calibration_previous, E_i_previous, path)
            # Reset values
            ce = pd.DataFrame()
            e = pd.DataFrame()
            t = pd.DataFrame()
            measurement_time = 0
            # Start clustering the next calibration
            ce, e, t, measurement_time = cluster_and_append(data, ce, e, t, measurement_time, calibration, glitch_measurements)
        calibration_previous = calibration
        E_i_previous = energy






def cluster_and_append(data, ce, e, t, measurement_time, calibration,
                       glitch_measurements):
    if calibration in glitch_measurements:
        discard_glitch_events = True
    else:
        discard_glitch_events = False
    # Declare parameters
    keep_only_coincident_events = False
    strong_glitches = {'Van__3x3_High_Resolution_Calibration_18.0': 0.15,
                       'Van__3x3_High_Resolution_Calibration_20.0': 0.15,
                       'Van__3x3_High_Resolution_Calibration_25.0': 0.15
                       }
    if calibration in strong_glitches.keys():
        glitch_mid = strong_glitches[calibration]
    else:
        glitch_mid = 0.5
    ce_temp, e_temp, t_temp = cluster_data(data, [0, 1, 2])
    ce_red, e_red, t_red, m_t = filter_data(ce_temp, e_temp, t_temp,
                                            discard_glitch_events,
                                            keep_only_coincident_events,
                                            glitch_mid)
    ce = ce.append(ce_red)
    e = e.append(e_red)
    t = t.append(t_red)
    measurement_time += m_t
    return ce, e, t, measurement_time


def save(ce, e, t, measurement_time, calibration, number_of_detectors,
         module_order, detector_types, data_set, E_i, path):
    # Save clusters
    ce.to_hdf(path, 'coincident_events', complevel=9)
    e.to_hdf(path, 'events', complevel=9)
    t.to_hdf(path, 'triggers', complevel=9)
    # Save all parameters
    ## Convert to dataframe
    number_det = pd.DataFrame({'number_of_detectors': [number_of_detectors]})
    mod_or     = pd.DataFrame({'module_order': module_order})
    det_types  = pd.DataFrame({'detector_types': detector_types})
    da_set     = pd.DataFrame({'data_set': [data_set]})
    mt         = pd.DataFrame({'measurement_time': [measurement_time]})
    ca         = pd.DataFrame({'calibration': [calibration]})
    ei         = pd.DataFrame({'E_i': [E_i]})
    ## Save to file
    number_det.to_hdf(path, 'number_of_detectors', complevel=9)
    mod_or.to_hdf(path, 'module_order', complevel=9)
    det_types.to_hdf(path, 'detector_types', complevel=9)
    da_set.to_hdf(path, 'data_set', complevel=9)
    mt.to_hdf(path, 'measurement_time', complevel=9)
    ei.to_hdf(path, 'E_i', complevel=9)
    ca.to_hdf(path, 'calibration', complevel=9)









# =============================================================================
# Helper Functions
# =============================================================================

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def create_dict(size, names):
    clu = {names[0]: np.zeros([size], dtype=int)}

    for name in names[1:len(names)]:
        clu.update({name: np.zeros([size], dtype=int)})

    return clu

def create_ess_channel_to_coordinate_map(theta, offset):
    dirname = os.path.dirname(__file__)
    file_path = os.path.join(dirname,
                             '../Tables/Coordinates_MG_SEQ_ESS.xlsx')
    matrix = pd.read_excel(file_path).values
    coordinates = matrix[1:801]
    ess_ch_to_coord = np.empty((3, 124, 80), dtype='object')
    coordinate = {'x': -1, 'y': -1, 'z': -1}
    axises =  ['x','y','z']

    c_offset = [[-1, 1, -1], [-1, -1, 1], [-1, 1, 1]]
    c_count = 0

    for i, row in enumerate(coordinates):
        grid_ch = i // 20 + 80
        for j, col in enumerate(row):
            module = j // 12
            layer = (j // 3) % 4
            wire_ch = (19 - (i % 20)) + (layer * 20)
            coordinate_count = j % 3
            coordinate[axises[coordinate_count]] = col
            if coordinate_count == 2:
                x = coordinate['x']
                y = coordinate['y']
                z = coordinate['z']
                # Convert from [mm] to [m]
                x = x/1000
                y = y/1000
                z = z/1000
                # Insert corners of vessels
                if wire_ch == 0 and grid_ch == 80 and module == 0:
                    ess_ch_to_coord[0][120][0] = {'x': offset['x'], 'y': offset['y'], 'z': offset['z']}
                if (  (wire_ch == 0 and grid_ch == 119 and module == 0)
                    | (wire_ch == 60 and grid_ch == 80 and module == 2)
                    | (wire_ch == 60 and grid_ch == 119 and module == 2)
                    ):
                    x_temp = x + 46.514/1000 * c_offset[c_count][0] + np.finfo(float).eps
                    y_temp = y + 37.912/1000 * c_offset[c_count][1] + np.finfo(float).eps
                    z_temp = z + 37.95/1000 * c_offset[c_count][2] + np.finfo(float).eps
                    z_temp, x_temp, y_temp = x_temp, y_temp, z_temp
                    x_temp, z_temp = get_new_x(x_temp, z_temp, theta), get_new_y(x_temp, z_temp, theta)
                    # Apply translation
                    x_temp += offset['x']
                    y_temp += offset['y']
                    z_temp += offset['z']
                    ess_ch_to_coord[0][121+c_count][0] = {'x': x_temp,
                                                          'y': y_temp,
                                                          'z': z_temp}
                    c_count += 1

                # Shift to match internal and external coordinate system
                z, x, y = x, y, z
                # Apply rotation
                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                # Apply translation
                x += offset['x']
                y += offset['y']
                z += offset['z']

                ess_ch_to_coord[module, grid_ch, wire_ch] = {'x': x, 'y': y,
                                                             'z': z}
                coordinate = {'x': -1, 'y': -1, 'z': -1}

    return ess_ch_to_coord

def create_ill_channel_to_coordinate_map(theta, offset):

    WireSpacing  = 10     #  [mm]
    LayerSpacing = 23.5   #  [mm]
    GridSpacing  = 23.5   #  [mm]

    x_offset = 46.514     #  [mm]
    y_offset = 37.912     #  [mm]
    z_offset = 37.95      #  [mm]

    corners =   [[0, 80], [0, 119], [60, 80], [60, 119]]
    corner_offset = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1]]

    # Make for longer to include the for corners of the vessel
    ill_ch_to_coord = np.empty((3,124,80),dtype='object')
    for Bus in range(0,3):
        for GridChannel in range(80,120):
            for WireChannel in range(0,80):
                    x = (WireChannel % 20)*WireSpacing + x_offset
                    y = ((WireChannel // 20)*LayerSpacing
                         + (Bus*4*LayerSpacing) + y_offset)
                    z = ((GridChannel-80)*GridSpacing) + z_offset
                    # Convert from [mm] to [m]
                    x = x/1000
                    y = y/1000
                    z = z/1000
                    # Shift to match internal and external coordinate system
                    z, x, y = x, y, z
                    # Apply rotation
                    x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                    # Apply translation
                    x += offset['x']
                    y += offset['y']
                    z += offset['z']

                    ill_ch_to_coord[Bus,GridChannel,WireChannel] = {'x': x,
                                                                    'y': y,
                                                                    'z': z}
        if Bus == 0:
            for i, corner in enumerate(corners[1:2]):
                WireChannel = corner[0]
                GridChannel = corner[1]
                x = (WireChannel % 20)*WireSpacing + x_offset
                y = ((WireChannel // 20)*LayerSpacing + (Bus*4*LayerSpacing) + y_offset)
                z = ((GridChannel-80)*GridSpacing) + z_offset
                x += corner_offset[i+1][0] * x_offset
                y += corner_offset[i+1][1] * y_offset
                z += corner_offset[i+1][2] * z_offset
                x = x/1000 + np.finfo(float).eps
                y = y/1000 + np.finfo(float).eps
                z = z/1000 + np.finfo(float).eps
                z, x, y = x, y, z

                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                x += offset['x']
                y += offset['y']
                z += offset['z']
                ill_ch_to_coord[0, 121+i, 0] = {'x': x, 'y': y, 'z': z}

            ill_ch_to_coord[Bus, 120, 0] = {'x': offset['x'], 'y': offset['y'], 'z': offset['z']}


        if Bus == 2:
            for i, corner in enumerate(corners[2:]):
                WireChannel = corner[0]
                GridChannel = corner[1]
                x = (WireChannel % 20)*WireSpacing + x_offset
                y = ((WireChannel // 20)*LayerSpacing + (Bus*4*LayerSpacing) + y_offset)
                z = ((GridChannel-80)*GridSpacing) + z_offset
                x += corner_offset[i+2][0] * x_offset
                y += corner_offset[i+2][1] * y_offset
                z += corner_offset[i+2][2] * z_offset
                x = x/1000
                y = y/1000
                z = z/1000
                z, x, y = x, y, z
                x, z = get_new_x(x, z, theta), get_new_y(x, z, theta)
                x += offset['x']
                y += offset['y']
                z += offset['z']
                ill_ch_to_coord[0, 122+i, 0] = {'x': x, 'y': y, 'z': z}

    return ill_ch_to_coord

def get_d(Bus, WireChannel, GridChannel, detector_vec):
    coord = None
    d = None
    if 0 <= Bus <= 2:
        coord = detector_vec[0][flip_bus(Bus%3), GridChannel,
                                flip_wire(WireChannel)]
    elif 3 <= Bus <= 5:
        coord = detector_vec[1][flip_bus(Bus%3), GridChannel,
                                flip_wire(WireChannel)]
    elif 6 <= Bus <= 8:
        coord = detector_vec[2][flip_bus(Bus%3), GridChannel,
                                flip_wire(WireChannel)]

    return np.sqrt((coord['x'] ** 2) + (coord['y'] ** 2) + (coord['z'] ** 2))


def flip_bus(bus):
    flip_bus_dict = {0: 2, 1: 1, 2: 0}
    return flip_bus_dict[bus]

def flip_wire(wCh):
        if 0 <= wCh <= 19:
            wCh += 60
        elif 20 <= wCh <= 39:
            wCh += 20
        elif 40 <= wCh <= 59:
            wCh -= 20
        elif 60 <= wCh <= 79:
            wCh -= 60
        return wCh

def get_new_x(x, y, theta):
    return np.cos(np.arctan(y/x)+theta)*np.sqrt(x ** 2 + y ** 2)

def get_new_y(x, y, theta):
    return np.sin(np.arctan(y/x)+theta)*np.sqrt(x ** 2 + y ** 2)
