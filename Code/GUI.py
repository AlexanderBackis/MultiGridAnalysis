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
from cluster import import_data, cluster_data, filter_data, unzip_data, load_data, save_data
from cluster import cluster_and_save_all_MG_data
from Plotting.PHS import PHS_1D_plot, PHS_2D_plot, PHS_wires_vs_grids_plot
from Plotting.Coincidences import Coincidences_2D_plot, Coincidences_3D_plot
from Plotting.Coincidences import Coincidences_Front_Top_Side_plot
from plot import Multiplicity_plot
from plot import ToF_plot, Timestamp_plot
from plot import dE_plot, RRM_plot, plot_all_energies
from plot import plot_FWHM_overview, plot_Efficency_overview, ToF_sweep_animation
from plot import plot_He3_data, wires_sweep_animation, grids_sweep_animation
from plot import angular_dependence_plot, angular_animation_plot, figure_of_merit_plot
from plot import different_depths, figure_of_merit_energy_sweep, He3_histogram_3D_plot
from plot import He3_histo_all_energies_animation, He3_histogram_3D_ToF_sweep
from plot import cluster_all_raw_He3, C4H2I2S_compare_all_energies, get_count_rate
from plot import plot_all_energies_ToF, find_He3_measurement_calibration
from plot import cluster_raw_He3, import_He3_coordinates_raw, beam_monitor_histogram
from plot import plot_He3_variation, plot_He3_variation_dE, plot_all_PHS, plot_all_dE
from Plotting.Scattering import compare_all_shoulders
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
        self.export_progress.close()
        self.iter_progress.close()
        self.tof_sweep_progress.close()
        self.wires_sweep_progress.close()
        self.grids_sweep_progress.close()
        self.ang_dist_progress.close()
        self.show()

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
            self.iter_progress.close()
            self.measurement_time_label.show()
            self.data_sets_label.show()
            self.module_order_label.show()
            self.detector_types_label.show()
            self.tof_sweep_progress.close()
            self.update()
            self.app.processEvents()

    def Save_action(self):
        save_path = QFileDialog.getSaveFileName()[0]
        if save_path != '':
            save_data(self.Coincident_events, self.Events, self.Triggers,
                      self.number_of_detectors,
                      self.module_order, self.detector_types, self.data_sets, 
                      self.measurement_time,
                      self.E_i, self.calibration, self, save_path)

    def PHS_1D_action(self):
        if self.iter_all.isChecked():
            plot_all_PHS()
        else:
            if (self.data_sets != '') and (self.Events.shape[0] > 0):
                fig = PHS_1D_plot(self.Events, self.data_sets, self)
                fig.show()

    def PHS_2D_action(self):
        if (self.data_sets != '') and (self.Events.shape[0] > 0):
            fig = PHS_2D_plot(self.Events, self.data_sets,
                              self.module_order)
            fig.show()


    def Coincidences_2D_action(self):
        if (self.data_sets != ''):
            fig = Coincidences_2D_plot(self.filter_ce_clusters(),
                                       self.data_sets,
                                       self.module_order)
            fig.show()


    def Coincidences_3D_action(self):
        if (self.data_sets != ''):
            Coincidences_3D_plot(self.filter_ce_clusters(), self.data_sets)

    def Coincidences_Front_Top_Side_action(self):
        if (self.data_sets != ''):
            fig = Coincidences_Front_Top_Side_plot(self.filter_ce_clusters(),
                                                   self.data_sets,
                                                   self.module_order,
                                                   self.number_of_detectors
                                                   )
            fig.show()

    def Multiplicity_action(self):
        if (self.data_sets != ''):
            Multiplicity_plot(self.filter_ce_clusters(),
                              self.data_sets,
                              self.module_order,
                              self.number_of_detectors
                              )
    
    def PHS_wires_vs_grids_action(self):
        if (self.data_sets != ''):
            fig = PHS_wires_vs_grids_plot(self.filter_ce_clusters(),
                                          self.data_sets,
                                          self.module_order)
            fig.show()

    def ToF_action(self):
        if self.iter_all.isChecked():
            plot_all_energies_ToF(self, self.is_CLB.isChecked(), self.is_pure_al.isChecked())
        else:
            fig, __, __ = ToF_plot(self.Coincident_events, self.data_sets, self.calibration, self.E_i,
                                   self.measurement_time, self.is_CLB.isChecked(), self.is_pure_al.isChecked(),
                                   self)
            fig.show()

    def Timestamp_action(self):
        if (self.data_sets != ''):
            Timestamp_plot(self.Coincident_events, self.data_sets)

    def dE_action(self):
        if not self.compare_he3.isChecked():
            if self.iter_all.isChecked():
                plot_all_dE(self.is_CLB.isChecked(), self.is_pure_al.isChecked(),
                            self, int(self.dE_bins.text()))
            else:
                print('CALIBRATION: %s' % self.get_calibration())
                fig = dE_plot(self.filter_ce_clusters(), self.E_i, self.get_calibration(),
                              self.measurement_time, self, self.is_CLB.isChecked(),
                              self.is_pure_al.isChecked(), int(self.dE_bins.text()))
                fig.show()
        elif self.iter_all.isChecked():
            plot_all_energies(self.is_pure_al.isChecked(),
                                  self.is_raw.isChecked(),
                                  self.is5by5.isChecked(),
                                  self.is_CLB.isChecked(),
                                  self.is_corrected.isChecked(),
                                  self.useGaussian.isChecked(),
                                  self)
        else:
            calcFWHM = True
            vis_help = False
            p0 = [1.20901528e+04, 5.50978749e-02, 1.59896619e+00] #, -2.35758418, 9.43166002e+01]
            fig, __, __, __, __, __ = plot_He3_data(self.filter_ce_clusters(),
                                                        self.data_sets,
                                                        self.get_calibration(),
                                                        self.measurement_time,
                                                        self.E_i,
                                                        calcFWHM,
                                                        vis_help,
                                                        self.back_yes.isChecked(),
                                                        self,
                                                        self.is_pure_al.isChecked(),
                                                        self.is_raw.isChecked(),
                                                        self.is5by5.isChecked(),
                                                        self.useGaussian.isChecked(),
                                                        p0,
                                                        self.is_CLB.isChecked(),
                                                        self.is_corrected.isChecked()
                                                        )
            fig.show()


    def RRM_action(self):
        if (self.data_sets != ''):
            back_yes = True
            E_i_vec = [float(self.RRM_E_i_1.text()), float(self.RRM_E_i_2.text())]
            RRM_plot(self.filter_ce_clusters(), self.data_sets,
                     float(self.RRM_split.text()), E_i_vec,
                     self.measurement_time, back_yes, self, 
                     self.is_pure_al.isChecked())

    def FWHM_action(self):
        plot_FWHM_overview()

    def Efficiency_action(self):
        plot_Efficency_overview()

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
        	He3_histogram_3D_plot(mes_id, save_path, self.data_sets, path)

    def He3_ToF_sweep_action(self):
        path = QFileDialog.getOpenFileName()[0]
        if path != '':
            title = path.rsplit('/', 1)[-1]
            He3_histogram_3D_ToF_sweep(self, path, title)

    def cluster_all_He3_action(self):
        cluster_all_raw_He3()

    def C4H2I2_all_energies_action(self):
        C4H2I2S_compare_all_energies(self,
                                     self.back_yes.isChecked(),
                                     self.is_pure_al.isChecked()
                                     )
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
    			print(id)
    			calibration = find_He3_measurement_calibration(id)
    			cluster = cluster_raw_He3(calibration, x, y, z, d, az, pol, path)
    			# Save clusters
    			dir_name = os.path.dirname(__file__)
    			output_path = os.path.join(dir_name, '../Clusters/He3/%s.h5' % calibration)
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
        compare_all_shoulders(self)

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
        self.grids_sweep.clicked.connect(self.grids_sweep_action)
        self.angular_dependence.clicked.connect(self.angular_dependence_action)
        self.angular_animation.clicked.connect(self.angular_animation_action)
        self.FOM.clicked.connect(self.FOM_animation_action)
        self.depth.clicked.connect(self.different_depths_action)
        self.FOM_energy_sweep.clicked.connect(self.FOM_energies_animation_action)
        self.He3_hist.clicked.connect(self.He3_histogram_3D_action)
        self.He3_ToF_sweep.clicked.connect(self.He3_ToF_sweep_action)
        self.Cluster_all_He3.clicked.connect(self.cluster_all_He3_action)
        self.C4H2I2S.clicked.connect(self.C4H2I2_all_energies_action)
        self.CountRate.clicked.connect(self.get_count_rate_action)
        self.cluster_he3.clicked.connect(self.cluster_He3_action)
        self.cluster_all_MG.clicked.connect(self.cluster_all_MG_action)
        self.beamMonitor.clicked.connect(self.beam_monitor_action)
        self.He3_variation.clicked.connect(self.plot_He3_variation_action)
        self.He3_variation_dE.clicked.connect(self.plot_He3_variation_action_dE)
        self.shoulder.clicked.connect(self.shoulder_action)



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
        self.parent.measurement_time = 0
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
            ce_temp, e_temp, t_temp = cluster_data(data, [0, 1, 2], self.progressBar, 
                                                   self, self.app)
            ce_red, e_red, t_red, m_t = filter_data(ce_temp, e_temp, t_temp,
                                                    self.discard_glitch_events,
                                                    self.keep_only_coincident_events)
            self.parent.measurement_time += m_t
            self.parent.Coincident_events = self.parent.Coincident_events.append(ce_red)
            self.parent.Events = self.parent.Events.append(e_red)
            self.parent.Triggers = self.parent.Triggers.append(t_red)
            os.remove(file_path)

        print(self.parent.measurement_time)

        self.parent.measurement_time = round(self.parent.measurement_time, 2)
        self.parent.Coincident_events.reset_index(drop=True, inplace=True)
        self.parent.Events.reset_index(drop=True, inplace=True)
        self.parent.Triggers.reset_index(drop=True, inplace=True)

        self.parent.measurement_time_label.setText(str(self.parent.measurement_time))
        self.parent.data_sets_label.setText(str(self.parent.data_sets))
        self.parent.module_order_label.setText(str(self.parent.module_order))
        self.parent.detector_types_label.setText(str(self.parent.detector_types))
        self.parent.update()
        self.close()

    def cancel_cluster_action(self):
        self.close()





app = QApplication(sys.argv)
main_window = MainWindow(app)
main_window.setAttribute(Qt.WA_DeleteOnClose, True)
main_window.setup_buttons()
sys.exit(app.exec_())

