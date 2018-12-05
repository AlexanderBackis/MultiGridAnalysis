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
from plot import PHS_1D_plot, PHS_2D_plot, Coincidences_2D_plot
from plot import Coincidences_Front_Top_Side_plot, Multiplicity_plot
from plot import PHS_wires_vs_grids_plot, ToF_plot, Timestamp_plot
from plot import dE_plot, Coincidences_3D_plot, RRM_plot, plot_all_energies
from plot import plot_FWHM_overview, plot_Efficency_overview, ToF_sweep_animation
from plot import plot_He3_data, wires_sweep_animation, grids_sweep_animation
from plot import angular_dependence_plot
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
        self.show()

    def Cluster_action(self):
        dialog = ClusterDialog(self.app, self)
        dialog.setup_cluster_buttons(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.exec_()
    
    def Load_action(self):
        clusters_path = QFileDialog.getOpenFileName()[0]
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
        print(self.E_i)
        self.update()
        self.app.processEvents()

    def Save_action(self):
        save_path = QFileDialog.getSaveFileName()[0]
        save_data(self.Coincident_events, self.Events, self.Triggers,
                  self.number_of_detectors,
                  self.module_order, self.detector_types, self.data_sets, 
                  self.measurement_time,
                  self.E_i, self.calibration, self, save_path)

    def PHS_1D_action(self):
        e = self.Events
        e = e[(e.Channel == self.Channel.value()) & 
              (e.Bus == self.Module.value())]
        PHS_1D_plot(e, self.data_sets)

    def PHS_2D_action(self):
        PHS_2D_plot(self.Events, self.data_sets,
                    self.module_order,
                    self.number_of_detectors)

    def Coincidences_2D_action(self):
        Coincidences_2D_plot(self.filter_ce_clusters(), self.data_sets,
                             self.module_order, self.number_of_detectors)

    def Coincidences_3D_action(self):
        Coincidences_3D_plot(self.filter_ce_clusters(), self.data_sets)

    def Coincidences_Front_Top_Side_action(self):
        Coincidences_Front_Top_Side_plot(self.filter_ce_clusters(),
                                         self.data_sets,
                                         self.module_order,
                                         self.number_of_detectors)

    def Multiplicity_action(self):
        Multiplicity_plot(self.filter_ce_clusters(),
                          self.data_sets,
                          self.module_order,
                          self.number_of_detectors)
    
    def PHS_wires_vs_grids_action(self):
        PHS_wires_vs_grids_plot(self.filter_ce_clusters(),
                                self.data_sets,
                                self.module_order,
                                self.number_of_detectors)

    def ToF_action(self):
        ToF_plot(self.filter_ce_clusters(), self.data_sets)

    def Timestamp_action(self):
        Timestamp_plot(self.filter_ce_clusters(), self.data_sets)

    def dE_action(self):
        if not self.compare_he3.isChecked():
            fig, __, __, __ = dE_plot(self.filter_ce_clusters(), self.data_sets, self.E_i,
                                      self.get_calibration(), self.measurement_time,
                                      self.back_yes.isChecked(), self)
            fig.show()
        elif self.iter_all.isChecked():
            plot_all_energies(self.is_pure_al.isChecked(),
                              self.is_raw.isChecked(),
                              self)
        else:
            calcFWHM = True
            vis_help = False
            fig, __, __, __, __ = plot_He3_data(self.filter_ce_clusters(), self.data_sets,
                                                self.get_calibration(), self.measurement_time,
                                                self.E_i, calcFWHM,
                                                vis_help, self.back_yes.isChecked(), self,
                                                self.is_pure_al.isChecked(),
                                                self.is_raw.isChecked(), self.is5by5.isChecked()
                                                )
            fig.show()


    def RRM_action(self):
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
        ToF_sweep_animation(self.filter_ce_clusters(),
                            self.data_sets,
                            int(self.tof_start.text()),
                            int(self.tof_stop.text()),
                            int(self.tof_step.text()),
                            self
                            )

    def wires_sweep_action(self):
        wires_sweep_animation(self.filter_ce_clusters(),
                              self.data_sets, self)

    def grids_sweep_action(self):
        grids_sweep_animation(self.filter_ce_clusters(),
                              self.data_sets, self)

    def angular_dependence_action(self):
        angular_dependence_plot(self.Coincident_events,
                                self.data_sets, self)


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
                         (ce.gCh >= self.grid_min.value() + 80 - 1) &
                         (ce.gCh <= self.grid_max.value() + 80 - 1) &
                         
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

