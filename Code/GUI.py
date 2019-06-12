#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:20:42 2018

@author: alexanderbackis
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from cluster import (import_data, cluster_data, filter_data, unzip_data,
                     load_data, save_data, cluster_and_save_all_MG_data)
from Plotting.PHS import (PHS_1D_plot, PHS_2D_plot, PHS_wires_vs_grids_plot,
                          PHS_wires_and_grids_plot)
from Plotting.Coincidences import (Coincidences_2D_plot, Coincidences_3D_plot,
                                   Coincidences_Front_Top_Side_plot,
                                   Coincidences_2D_plot_all_energies)
from Plotting.ToF import (ToF_histogram, ToF_compare_MG_and_He3,
                          plot_all_energies_ToF)
from Plotting.EnergyTransfer import (energy_transfer_histogram,
                                     energy_transfer_compare_MG_and_He3,
                                     plot_all_energies_dE,
                                     plot_efficiency,
                                     plot_FWHM, C4H2I2S_compare_all_energies)
from Plotting.HelperFunctions import (get_He3_duration,
                                      get_He3_tubes_area_and_solid_angle,
                                      get_multi_grid_area_and_solid_angle)
from Plotting.Miscellaneous import (calculate_depth_variation,
                                    calculate_uncertainty,
                                    calculate_all_uncertainties,
                                    signal_dependence_on_ADC_threshold,
                                    background_dependence_on_ADC_threshold,
                                    Timestamp_plot,
                                    Multiplicity_plot,
                                    compare_corrected_and_uncorrected_He3_data,
                                    plot_ce_multiplicity)

from plot import RRM_plot, ToF_sweep_animation
from plot import plot_He3_data, wires_sweep_animation, grids_sweep_animation
from plot import angular_dependence_plot, angular_animation_plot, figure_of_merit_plot
from plot import different_depths, figure_of_merit_energy_sweep, He3_histogram_3D_plot
from plot import He3_histo_all_energies_animation, He3_histogram_3D_ToF_sweep
from plot import cluster_all_raw_He3, get_count_rate
from plot import find_He3_measurement_calibration
from plot import cluster_raw_He3, import_He3_coordinates_raw, beam_monitor_histogram
from plot import plot_He3_variation, plot_He3_variation_dE
from Plotting.Scattering import (compare_all_shoulders, compare_all_shoulders_5x5,
                                 analyze_lineshape)
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self, app, parent=None):
        super(MainWindow, self).__init__(parent)
        dir_name = os.path.dirname(__file__)
        title_screen_path = os.path.join(dir_name, '../Windows/title_screen.ui')
        self.app = app
        self.ui = uic.loadUi(title_screen_path, self)
        self.measurement_time = 0
        self.data_sets = ''
        self.Coincident_events = pd.DataFrame()
        self.Events = pd.DataFrame()
        self.Triggers = pd.DataFrame()
        self.number_of_detectors = 3
        self.module_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.detector_types = ['ILL', 'ESS', 'ESS']
        self.exceptions = [0, 1, 2]
        self.calibration = None
        self.E_i = None
        self.sample = None
        self.load_progress.close()
        self.save_progress.close()
        #self.export_progress.close()
        #self.iter_progress.close()
        self.tof_sweep_progress.close()
        self.wires_sweep_progress.close()
        #self.grids_sweep_progress.close()
        self.ang_dist_progress.close()
        self.close_all_special_buttons()
        self.special_buttons_visible = False
        self.show()
        self.app.processEvents()
        self.update()
        self.app.processEvents()
        self.update()

    def Cluster_action(self):
        dialog = ClusterDialog(self.app, self)
        dialog.setup_cluster_buttons(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.exec_()

    def Load_action(self):
        clusters_path = QFileDialog.getOpenFileName()[0]
        if clusters_path != '':
            load_data(clusters_path, self)
            self.measurement_time = round(self.measurement_time, 2)
            self.measurement_time_label.setText(str(self.measurement_time))
            self.data_sets_label.setText(str(self.data_sets))
            self.module_order_label.setText(str(self.module_order))
            self.detector_types_label.setText(str(self.detector_types))
            self.measurement_time_label.close()
            self.data_sets_label.close()
            self.module_order_label.close()
            self.detector_types_label.close()
            #self.iter_progress.close()
            self.measurement_time_label.show()
            self.data_sets_label.show()
            self.module_order_label.show()
            self.detector_types_label.show()
            self.tof_sweep_progress.close()
            self.update()
            self.app.processEvents()
            # Create folder which will store text files from export
            dir_name = os.path.dirname(__file__)
            folder_path = os.path.join(dir_name, '../text/%s' % self.data_sets)
            mkdir_p(folder_path)

    def Save_action(self):
        save_path = QFileDialog.getSaveFileName()[0]
        if save_path != '':
            save_data(self.Coincident_events, self.Events, self.Triggers,
                      self.number_of_detectors,
                      self.module_order, self.detector_types, self.data_sets,
                      self.measurement_time,
                      self.E_i, self.calibration, self, save_path)

    def close_all_special_buttons(self):
        self.He3_hist.close()
        self.cluster_all_MG.close()
        self.He3_variation.close()
        self.Cluster_all_He3.close()
        self.He3_variation_dE.close()
        self.angular_animation.close()
        self.He3_ToF_sweep.close()
        self.FOM.close()
        self.wires_sweep.close()
        self.FOM_energy_sweep.close()
        self.signalADC.close()
        self.C4H2I2S.close()
        self.backADC.close()
        #self.angular_dependence.close()
        self.shoulder.close()
        self.uncertainty.close()
        self.depth_variation.close()
        #self.cluster_he3.close()
        self.corrUncorr.close()
        self.RRM.close()

    def open_all_special_buttons_action(self):
        if self.special_buttons_visible:
            self.He3_hist.close()
            self.cluster_all_MG.close()
            self.He3_variation.close()
            self.Cluster_all_He3.close()
            self.He3_variation_dE.close()
            self.angular_animation.close()
            self.He3_ToF_sweep.close()
            self.FOM.close()
            self.wires_sweep.close()
            self.FOM_energy_sweep.close()
            self.signalADC.close()
            self.C4H2I2S.close()
            self.backADC.close()
            #self.angular_dependence.close()
            self.shoulder.close()
            self.uncertainty.close()
            self.depth_variation.close()
            self.corrUncorr.close()
            self.RRM.close()
            #self.cluster_he3.close()
            self.special_buttons_visible = False
            self.update()
            self.app.processEvents()
            self.update()
        else:
            self.RRM.show()
            self.He3_hist.show()
            self.cluster_all_MG.show()
            self.He3_variation.show()
            self.Cluster_all_He3.show()
            self.He3_variation_dE.show()
            self.angular_animation.show()
            self.He3_ToF_sweep.show()
            self.FOM.show()
            self.wires_sweep.show()
            self.FOM_energy_sweep.show()
            self.signalADC.show()
            self.C4H2I2S.show()
            self.backADC.show()
            self.shoulder.show()
            self.uncertainty.show()
            self.depth_variation.show()
            self.corrUncorr.show()
            #self.cluster_he3.show()
            #self.angular_dependence.show()
            self.special_buttons_visible = True
            self.update()
            self.app.processEvents()
            self.update()

    def PHS_1D_action(self):
        if self.iter_all.isChecked():
            plot_all_PHS()
        else:
            if (self.data_sets != ''):
                fig = PHS_wires_and_grids_plot(self.Coincident_events,
                                               self.data_sets,
                                               self)
                fig.show()

    def PHS_2D_action(self):
        if (self.data_sets != '') and (self.Events.shape[0] > 0):
            fig = PHS_2D_plot(self.Events, self.data_sets,
                              self.module_order)

    def Coincidences_2D_action(self):
        if (self.data_sets != '') or self.iter_all.isChecked():
            if self.iter_all.isChecked():
                Coincidences_2D_plot_all_energies(self)
            else:
                fig = Coincidences_2D_plot(self.Coincident_events,
                                           self.data_sets,
                                           self.module_order,
                                           self)
                if self.createPlot.isChecked():
                    fig.show()

    def Coincidences_3D_action(self):
        if (self.data_sets != ''):
            Coincidences_3D_plot(self.Coincident_events, self.data_sets, self)

    def Coincidences_Front_Top_Side_action(self):
        if (self.data_sets != ''):
            fig = Coincidences_Front_Top_Side_plot(self.Coincident_events,
                                                   self.data_sets,
                                                   self.module_order,
                                                   self.number_of_detectors,
                                                   self)
            fig.show()

    def Multiplicity_action(self):
        if (self.data_sets != ''):
            Multiplicity_plot(self.Coincident_events,
                              self.data_sets,
                              self.module_order,
                              self.number_of_detectors,
                              self)

    def PHS_wires_vs_grids_action(self):
        if (self.data_sets != ''):
            fig = PHS_wires_vs_grids_plot(self.Coincident_events,
                                          self.data_sets,
                                          self.module_order,
                                          self)
            fig.show()

    def ToF_action(self):
        if self.iter_all.isChecked():
            plot_all_energies_ToF(self)
        elif self.compare_he3.isChecked() and (self.data_sets != ''):
            He3_time = get_He3_duration(self.calibration)
            He3_area, __ = get_He3_tubes_area_and_solid_angle()
            MG_area, __ = get_multi_grid_area_and_solid_angle(self,
                                                              self.calibration,
                                                              self.E_i)
            print('MG_area: %.2f' % MG_area)
            print('He3_area: %-2f' % He3_area)
            fig = ToF_compare_MG_and_He3(self.Coincident_events,
                                         self.calibration,
                                         self.E_i,
                                         self.measurement_time,
                                         He3_time,
                                         MG_area,
                                         He3_area,
                                         self)
            fig.show()
        elif self.data_sets != '':
            fig = ToF_histogram(self.Coincident_events, self.data_sets, self)
            fig.show()

    def Timestamp_action(self):
        if (self.data_sets != ''):
            Timestamp_plot(self.Coincident_events, self.data_sets)

    def dE_action(self):
        if self.iter_all.isChecked():
            plot_all_energies_dE(self)
        elif self.compare_he3.isChecked() and (self.data_sets != ''):
            p0 = [1.20901528e+04, 5.50978749e-02, 1.59896619e+00]
            He3_area, He3_solid_angle = get_He3_tubes_area_and_solid_angle()
            MG_area, MG_solid_angle = get_multi_grid_area_and_solid_angle(self,
                                                                          self.calibration,
                                                                          self.E_i)
            print('He3_area/MG_area: %.2f' % (He3_area/MG_area))
            print('He3_solid_angle/MG_solid_angle: %.2f' % (He3_solid_angle/MG_solid_angle))
            values = energy_transfer_compare_MG_and_He3(self.Coincident_events,
                                                        self.calibration,
                                                        self.E_i,
                                                        MG_solid_angle,
                                                        He3_solid_angle,
                                                        p0,
                                                        self)
            fig, __, __, __, __ = values
            fig.show()
        elif self.data_sets != '':
            fig = energy_transfer_histogram(self.Coincident_events,
                                            self.calibration,
                                            self.E_i, self)
            fig.show()


    def RRM_action(self):
        if (self.data_sets != ''):
            back_yes = True
            E_i_vec = [float(self.RRM_E_i_1.text()), float(self.RRM_E_i_2.text())]
            RRM_plot(self.filter_ce_clusters(), self.data_sets,
                     float(self.RRM_split.text()), E_i_vec,
                     self.measurement_time, back_yes, self,
                     False)

    def FWHM_action(self):
        plot_FWHM()

    def Efficiency_action(self):
        plot_efficiency()

    def ToF_sweep_action(self):
        if (self.data_sets != ''):
            ToF_sweep_animation(self.filter_ce_clusters(),
                                self.data_sets,
                                int(self.tof_start.text()),
                                int(self.tof_stop.text()),
                                int(self.tof_step.text()),
                                self
                                )

    def wires_sweep_action(self):
        if (self.data_sets != ''):
            wires_sweep_animation(self.filter_ce_clusters(),
                                  self.data_sets, self)

    def grids_sweep_action(self):
        if (self.data_sets != ''):
            grids_sweep_animation(self.filter_ce_clusters(),
                                  self.data_sets, self)

    def angular_dependence_action(self):
        if (self.data_sets != ''):
            fig = angular_dependence_plot(self.filter_ce_clusters(),
                                          self.data_sets,
                                          self.calibration)
            fig.show()

    def angular_animation_action(self):
        if (self.data_sets != ''):
            angular_animation_plot(self)

    def FOM_animation_action(self):
        if (self.data_sets != ''):
            figure_of_merit_plot(self)

    def different_depths_action(self):
        if (self.data_sets != ''):
            different_depths(self)

    def FOM_energies_animation_action(self):
        if (self.data_sets != ''):
            figure_of_merit_energy_sweep(self)

    def He3_histogram_3D_action(self):
        def find_He3_measurement_id(calibration):
            dirname = os.path.dirname(__file__)
            path = os.path.join(dirname, '../Tables/experiment_log.xlsx')
            matrix = pd.read_excel(path).values
            measurement_table = {}
            for row in matrix:
                measurement_table.update({row[1]: row[0]})
            return measurement_table[calibration]
        if self.iter_all.isChecked():
            He3_histo_all_energies_animation()
        else:
            path = QFileDialog.getOpenFileName()[0]
            save_path=None
            mes_id=None
            He3_histogram_3D_plot(mes_id, save_path, self.data_sets, path,
                                  self)

    def He3_ToF_sweep_action(self):
        path = QFileDialog.getOpenFileName()[0]
        if path != '':
            title = path.rsplit('/', 1)[-1]
            He3_histogram_3D_ToF_sweep(self, path, title)

    def cluster_all_He3_action(self):
        cluster_all_raw_He3()

    def C4H2I2_all_energies_action(self):
        C4H2I2S_compare_all_energies(self)

    def get_count_rate_action(self):
        if (self.data_sets != ''):
            get_count_rate(self.filter_ce_clusters(),
                           self.measurement_time,
                           self.data_sets,
                           self)

    def cluster_He3_action(self):
        files_to_cluster = QFileDialog.getOpenFileNames()[0]
        if files_to_cluster != '':
            x, y, z, d, az, pol = import_He3_coordinates_raw()
            for path in files_to_cluster:
                id = int(path.rsplit('/', 1)[-1][4:10])
                calibration = find_He3_measurement_calibration(id)
                cluster = cluster_raw_He3(calibration, x, y, z, d,
                                          az, pol, path)
                # Save clusters
                dir_name = os.path.dirname(__file__)
                output_path = os.path.join(dir_name, '../Clusters/He3/%s.h5'
                                                     % calibration)
                cluster.to_hdf(output_path, calibration, complevel=9)

    def cluster_all_MG_action(self):
        cluster_and_save_all_MG_data()

    def beam_monitor_action(self):
        fig = beam_monitor_histogram()
        fig.show()

    def plot_He3_variation_action(self):
        plot_He3_variation()

    def plot_He3_variation_action_dE(self):
        plot_He3_variation_dE()

    def shoulder_action(self):
        analyze_lineshape(self)

    def depth_variation_action(self):
        fig = calculate_depth_variation(self.Coincident_events, self)
        fig.show()

    def calculate_uncertainty_action(self):
        if self.iter_all.isChecked():
            calculate_all_uncertainties()
        else:
            fig, __, __ = calculate_uncertainty(self.calibration)
            fig.show()

    def calculate_signal_ADC_action(self):
        signal_dependence_on_ADC_threshold(self.Coincident_events,
                                           self.E_i, self.calibration,
                                           self)

    def back_ADC_action(self):
        background_dependence_on_ADC_threshold(self.Coincident_events,
                                               self.data_sets,
                                               self)

    def compare_corrected_and_uncorrected_action(self):
        compare_corrected_and_uncorrected_He3_data(self)

    def ce_multiplicity_action(self):
        if (self.data_sets != ''):
            fig = plot_ce_multiplicity(self.Coincident_events, self.data_sets,
                                       self)
            fig.show()

    def toogle_Plot_Text(self):
        self.createPlot.toggled.connect(
            lambda checked: checked and self.createText.setChecked(False))
        self.createText.toggled.connect(
            lambda checked: checked and self.createPlot.setChecked(False))

    def setup_buttons(self):
        self.Cluster.clicked.connect(self.Cluster_action)
        self.Load.clicked.connect(self.Load_action)
        self.PHS_1D.clicked.connect(self.PHS_1D_action)
        self.PHS_2D.clicked.connect(self.PHS_2D_action)
        self.Coincidences_2D.clicked.connect(self.Coincidences_2D_action)
        self.Coincidences_Front_Top_Side.clicked.connect(self.Coincidences_Front_Top_Side_action)
        self.Multiplicity.clicked.connect(self.Multiplicity_action)
        self.PHS_wires_vs_grids.clicked.connect(self.PHS_wires_vs_grids_action)
        self.ToF.clicked.connect(self.ToF_action)
        self.Timestamp.clicked.connect(self.Timestamp_action)
        self.dE.clicked.connect(self.dE_action)
        self.Save.clicked.connect(self.Save_action)
        self.Coincidences_3D.clicked.connect(self.Coincidences_3D_action)
        self.RRM.clicked.connect(self.RRM_action)
        self.FWHM.clicked.connect(self.FWHM_action)
        self.Efficiency.clicked.connect(self.Efficiency_action)
        self.ToF_sweep.clicked.connect(self.ToF_sweep_action)
        self.wires_sweep.clicked.connect(self.wires_sweep_action)
        #self.grids_sweep.clicked.connect(self.grids_sweep_action)
        #self.angular_dependence.clicked.connect(self.angular_dependence_action)
        self.angular_animation.clicked.connect(self.angular_animation_action)
        self.FOM.clicked.connect(self.FOM_animation_action)
        #self.depth.clicked.connect(self.different_depths_action)
        self.FOM_energy_sweep.clicked.connect(self.FOM_energies_animation_action)
        self.He3_hist.clicked.connect(self.He3_histogram_3D_action)
        self.He3_ToF_sweep.clicked.connect(self.He3_ToF_sweep_action)
        self.Cluster_all_He3.clicked.connect(self.cluster_all_He3_action)
        self.C4H2I2S.clicked.connect(self.C4H2I2_all_energies_action)
        self.CountRate.clicked.connect(self.get_count_rate_action)
        #self.cluster_he3.clicked.connect(self.cluster_He3_action)
        self.cluster_all_MG.clicked.connect(self.cluster_all_MG_action)
        #self.beamMonitor.clicked.connect(self.beam_monitor_action)
        self.He3_variation.clicked.connect(self.plot_He3_variation_action)
        self.He3_variation_dE.clicked.connect(self.plot_He3_variation_action_dE)
        self.shoulder.clicked.connect(self.shoulder_action)
        self.depth_variation.clicked.connect(self.depth_variation_action)
        self.uncertainty.clicked.connect(self.calculate_uncertainty_action)
        self.signalADC.clicked.connect(self.calculate_signal_ADC_action)
        self.backADC.clicked.connect(self.back_ADC_action)
        self.activateSpecialButtons.clicked.connect(self.open_all_special_buttons_action)
        self.corrUncorr.clicked.connect(self.compare_corrected_and_uncorrected_action)
        self.ce_multiplicity.clicked.connect(self.ce_multiplicity_action)
        self.toogle_Plot_Text()

    def get_calibration(self):
        calibrations =  ['High_Resolution', 'High_Flux', 'RRM']
        choices = ['High Resolution (HR)', 'High Flux (HF)', 'Rate Repetition Mode (RRM)']
        calibration_dict = {}
        for choice, calibration in zip(choices, calibrations):
            calibration_dict.update({choice: calibration})
        if self.calibration in choices:
            mode = calibration_dict[self.calibration]
            return 'Van__3x3_' + mode + '_Calibration_' + str(self.E_i)
        else:
            return self.calibration

    def filter_ce_clusters(self):
        ce = self.Coincident_events
        ce_filtered = ce[(ce.wM >= self.wM_min.value()) &
                         (ce.wM <= self.wM_max.value()) &
                         (ce.gM >= self.gM_min.value()) &
                         (ce.gM <= self.gM_max.value()) &  
                         (ce.wADC >= self.wADC_min.value()) &
                         (ce.wADC <= self.wADC_max.value()) &
                         (ce.gADC >= self.gADC_min.value()) &
                         (ce.gADC <= self.gADC_max.value()) &
                         (ce.ToF * 62.5e-9 * 1e6 >= self.ToF_min.value()) &
                         (ce.ToF * 62.5e-9 * 1e6 <= self.ToF_max.value()) &
                         (ce.Bus >= self.module_min.value()) &
                         (ce.Bus <= self.module_max.value()) &
                         (((ce.gCh >= self.grid_min.value() + 80 - 1) &
                          (ce.gCh <= self.lowerStartGrid.value() + 80 - 1)) |
                          ((ce.gCh <= self.grid_max.value() + 80 - 1) &
                           (ce.gCh >= self.upperStartGrid.value() + 80 - 1))) &
                         (((ce.wCh >= self.wire_min.value() - 1) &
                          (ce.wCh <= self.wire_max.value() - 1))
                          |
                          ((ce.wCh >= self.wire_min.value() + 20 - 1) &
                           (ce.wCh <= self.wire_max.value() + 20 - 1))
                          |
                          ((ce.wCh >= self.wire_min.value() + 40 - 1) &
                           (ce.wCh <= self.wire_max.value() + 40 - 1))
                          |
                          ((ce.wCh >= self.wire_min.value() + 60 - 1) &
                           (ce.wCh <= self.wire_max.value() + 60 - 1))
                          )
                         ]
        return ce_filtered


class ClusterDialog(QDialog):
    def __init__(self, app, parent=None):
        QDialog.__init__(self, parent)
        dir_name = os.path.dirname(__file__)
        title_screen_path = os.path.join(dir_name, '../Windows/cluster_window.ui')
        # Initiate attributes
        self.app = app
        self.parent = parent
        self.ui = uic.loadUi(title_screen_path, self)
        self.progressBar.close()
        self.File_text.close()
        self.colon.close()
        self.files_to_import = []
        self.import_full_files = True
        self.discard_glitch_events = True
        self.keep_only_coincident_events = False
        self.setWindowTitle("Select specifications")

    def Browse_action(self):
        self.files_to_import = QFileDialog.getOpenFileNames()[0]
        file_names = ''
        for i, file in enumerate(self.files_to_import):
            file_names += file.rsplit('/', 1)[-1]
            if i < len(self.files_to_import) - 1:
                file_names += '\n'
        self.Files.setText(file_names)
        self.update()

    def setup_cluster_buttons(self, parent):
        self.Browse.clicked.connect(self.Browse_action)
        self.initiate_cluster.clicked.connect(self.initiate_cluster_action)
        self.cancel_cluster.clicked.connect(self.cancel_cluster_action)

    def initiate_cluster_action(self, parent=None):
        if self.parent.write.isChecked():
            self.parent.measurement_time = 0
            self.parent.Coincident_events = pd.DataFrame()
            self.parent.Events = pd.DataFrame()
            self.parent.Triggers = pd.DataFrame()

        if len(self.files_to_import) == 1:
            self.parent.data_sets = str(self.files_to_import[0].rsplit('/', 1)[-1])
        else:
            self.parent.data_sets = (str(self.files_to_import[0].rsplit('/', 1)[-1]) 
                                        + ', ...')
        self.import_full_files = self.import_full.isChecked()
        self.discard_glitch_events = self.discard_glitch.isChecked()
        self.keep_only_coincident_events = self.keep_only_ce.isChecked()
        self.parent.calibration = self.choose_calibration.currentItem().text()
        self.parent.E_i = float(self.energy_selection.text())
        self.parent.sample = self.choose_sample.currentItem().text()
        self.progressBar.show()
        self.colon.show()
        self.File_text.show()
        ILL_buses = [self.ILL_1.value(),
                     self.ILL_2.value(),
                     self.ILL_3.value()]
        for i, zip_file_path in enumerate(self.files_to_import):
            self.progressBar.setValue(0)
            self.file_progression_text.close()
            self.file_progression_text.setText(str(i+1) + '/' + str(len(self.files_to_import)))
            self.file_progression_text.show()
            self.Loading_text.close()
            self.Loading_text.setText('Unzipping...')
            self.Loading_text.show()
            self.app.processEvents()
            file_path = unzip_data(zip_file_path)
            self.Loading_text.close()
            self.Loading_text.setText('Importing...')
            self.Loading_text.show()
            self.update()
            self.app.processEvents()
            self.app.processEvents()
            data = import_data(file_path)
            self.Loading_text.close()
            self.Loading_text.setText('Clustering...')
            self.Loading_text.show()
            self.update()
            self.app.processEvents()
            ce_temp, e_temp, t_temp = cluster_data(data, ILL_buses, self.progressBar, 
                                                   self, self.app)
            ce_red, e_red, t_red, m_t = filter_data(ce_temp, e_temp, t_temp,
                                                    self.discard_glitch_events,
                                                    self.keep_only_coincident_events)
            self.parent.measurement_time += m_t
            self.parent.Coincident_events = self.parent.Coincident_events.append(ce_red)
            self.parent.Events = self.parent.Events.append(e_red)
            self.parent.Triggers = self.parent.Triggers.append(t_red)
            os.remove(file_path)
        self.parent.measurement_time = round(self.parent.measurement_time, 2)
        self.parent.Coincident_events.reset_index(drop=True, inplace=True)
        self.parent.Events.reset_index(drop=True, inplace=True)
        self.parent.Triggers.reset_index(drop=True, inplace=True)

        self.parent.measurement_time_label.setText(str(self.parent.measurement_time))
        self.parent.data_sets_label.setText(str(self.parent.data_sets))
        self.parent.module_order_label.setText(str(self.parent.module_order))
        self.parent.detector_types_label.setText(str(self.parent.detector_types))
        # Create folder which will store text files from export
        dir_name = os.path.dirname(__file__)
        folder_path = os.path.join(dir_name, '../text/%s' % self.parent.data_sets)
        mkdir_p(folder_path)
        self.parent.update()
        self.close()

    def cancel_cluster_action(self):
        self.close()


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





app = QApplication(sys.argv)
main_window = MainWindow(app)
main_window.setAttribute(Qt.WA_DeleteOnClose, True)
main_window.setup_buttons()
sys.exit(app.exec_())

