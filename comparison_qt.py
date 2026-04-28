# comparison_qt.py

from __future__ import annotations

import os
import csv
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QRadioButton, QButtonGroup, QCheckBox, QLineEdit,
    QFrame, QSizePolicy
)

from cpp_analysis import extract_cpp
from spectrogram import plot_praat_spectrogram, get_f0_summary
from audio_io import SUPPORTED_AUDIO_FORMATS, convert_to_temp_mono_wav, load_audio_as_mono


ROI_COLOR = "#ff3b3b"
ROI_ALPHA = 0.28


class ComparisonWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("CepstralVox 2.0 - File Comparison")
        self.resize(1500, 860)

        self.file1_original = None
        self.file2_original = None
        self.file1_wav = None
        self.file2_wav = None

        self.audio1 = None
        self.audio2 = None
        self.sr1 = None
        self.sr2 = None

        self.region1 = None
        self.region2 = None

        self.result1 = None
        self.result2 = None

        self.span1 = None
        self.span2 = None
        self.roi_patch1 = None
        self.roi_patch2 = None

        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addLayout(self._build_top_controls())
        root.addLayout(self._build_spectrograms(), stretch=1)
        root.addWidget(self._build_results_bar())

    def _build_top_controls(self):
        layout = QGridLayout()
        layout.setSpacing(12)

        file_card = self._card("Files")
        file_layout = file_card.layout()

        self.file1_btn = QPushButton("Select Audio 1")
        self.file1_btn.setObjectName("primaryButton")
        self.file1_btn.clicked.connect(lambda: self.load_audio(slot=1))

        self.file2_btn = QPushButton("Select Audio 2")
        self.file2_btn.setObjectName("primaryButton")
        self.file2_btn.clicked.connect(lambda: self.load_audio(slot=2))

        self.file1_label = QLabel("Audio 1: not loaded")
        self.file1_label.setObjectName("smallLabel")
        self.file2_label = QLabel("Audio 2: not loaded")
        self.file2_label.setObjectName("smallLabel")

        file_layout.addWidget(self.file1_btn)
        file_layout.addWidget(self.file1_label)
        file_layout.addWidget(self.file2_btn)
        file_layout.addWidget(self.file2_label)

        settings_card = self._card("Analysis Settings")
        settings_layout = settings_card.layout()

        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["Sustained vowel", "Connected speech"])
        settings_layout.addWidget(self.file_type_combo)

        radio_row = QHBoxLayout()
        self.cpp_radio = QRadioButton("CPP")
        self.cpps_radio = QRadioButton("CPPS")
        self.cpp_radio.setChecked(True)

        self.measure_group = QButtonGroup(self)
        self.measure_group.addButton(self.cpp_radio)
        self.measure_group.addButton(self.cpps_radio)

        radio_row.addWidget(self.cpp_radio)
        radio_row.addWidget(self.cpps_radio)
        radio_row.addStretch()
        settings_layout.addLayout(radio_row)

        f0_card = self._card("F0 Range")
        f0_layout = f0_card.layout()

        f0_row = QHBoxLayout()
        self.f0_min_entry = QLineEdit("60")
        self.f0_max_entry = QLineEdit("330")
        self.f0_min_entry.setFixedWidth(70)
        self.f0_max_entry.setFixedWidth(70)

        f0_row.addWidget(QLabel("Min"))
        f0_row.addWidget(self.f0_min_entry)
        f0_row.addWidget(QLabel("Max"))
        f0_row.addWidget(self.f0_max_entry)
        f0_layout.addLayout(f0_row)

        self.refresh_f0_btn = QPushButton("Refresh Spectrograms")
        self.refresh_f0_btn.clicked.connect(self.refresh_spectrograms)
        f0_layout.addWidget(self.refresh_f0_btn)

        proc_card = self._card("Processing")
        proc_layout = proc_card.layout()

        self.vad_check = QCheckBox("Enable voiced-only extraction (VAD)")
        self.vad_check.setChecked(True)

        self.pause_check = QCheckBox("Enable pause removal")
        self.pause_check.setChecked(True)

        proc_layout.addWidget(self.vad_check)
        proc_layout.addWidget(self.pause_check)

        action_card = self._card("Actions")
        action_layout = action_card.layout()

        self.run_btn = QPushButton("Run Comparison")
        self.run_btn.setObjectName("cyanButton")
        self.run_btn.clicked.connect(self.run_comparison)

        self.clear_btn = QPushButton("Clear ROIs")
        self.clear_btn.clicked.connect(self.clear_rois)

        action_layout.addWidget(self.run_btn)
        action_layout.addWidget(self.clear_btn)

        layout.addWidget(file_card, 0, 0)
        layout.addWidget(settings_card, 0, 1)
        layout.addWidget(f0_card, 0, 2)
        layout.addWidget(proc_card, 0, 3)
        layout.addWidget(action_card, 0, 4)

        return layout

    def _build_spectrograms(self):
        layout = QHBoxLayout()
        layout.setSpacing(12)

        card1 = self._card("Audio 1 - Select ROI")
        card2 = self._card("Audio 2 - Select ROI")

        self.fig1, self.ax1 = plt.subplots(figsize=(7, 4.8))
        self.fig1.patch.set_facecolor("#111a28")
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.fig2, self.ax2 = plt.subplots(figsize=(7, 4.8))
        self.fig2.patch.set_facecolor("#111a28")
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.roi1_label = QLabel("ROI 1: not selected")
        self.roi1_label.setObjectName("roiInfo")
        self.roi1_label.setAlignment(Qt.AlignCenter)

        self.roi2_label = QLabel("ROI 2: not selected")
        self.roi2_label.setObjectName("roiInfo")
        self.roi2_label.setAlignment(Qt.AlignCenter)

        audio1_controls = QHBoxLayout()
        self.play1_btn = QPushButton("Play Audio 1")
        self.play1_btn.setObjectName("primaryButton")
        self.play1_btn.clicked.connect(lambda: self.play_audio(slot=1))
        self.play1_btn.setEnabled(False)

        self.stop1_btn = QPushButton("Stop Audio 1")
        self.stop1_btn.clicked.connect(self.stop_audio)
        self.stop1_btn.setEnabled(False)

        audio1_controls.addWidget(self.play1_btn)
        audio1_controls.addWidget(self.stop1_btn)
        audio1_controls.addStretch()

        audio2_controls = QHBoxLayout()
        self.play2_btn = QPushButton("Play Audio 2")
        self.play2_btn.setObjectName("primaryButton")
        self.play2_btn.clicked.connect(lambda: self.play_audio(slot=2))
        self.play2_btn.setEnabled(False)

        self.stop2_btn = QPushButton("Stop Audio 2")
        self.stop2_btn.clicked.connect(self.stop_audio)
        self.stop2_btn.setEnabled(False)

        audio2_controls.addWidget(self.play2_btn)
        audio2_controls.addWidget(self.stop2_btn)
        audio2_controls.addStretch()

        card1.layout().addWidget(self.canvas1, stretch=1)
        card1.layout().addWidget(self.roi1_label)
        card1.layout().addLayout(audio1_controls)

        card2.layout().addWidget(self.canvas2, stretch=1)
        card2.layout().addWidget(self.roi2_label)
        card2.layout().addLayout(audio2_controls)

        layout.addWidget(card1, stretch=1)
        layout.addWidget(card2, stretch=1)

        return layout

    def _build_results_bar(self):
        frame = QFrame()
        frame.setObjectName("bottomBar")

        layout = QGridLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        self.result1_label = self._result_metric("Audio 1", "-")
        self.result2_label = self._result_metric("Audio 2", "-")
        self.diff_label = self._result_metric("Difference", "-")
        self.percent_label = self._result_metric("Percent Change", "-")
        self.status_label = QLabel("Load two audio files, select ROIs, then run comparison.")
        self.status_label.setObjectName("smallLabel")

        layout.addWidget(self.result1_label, 0, 0)
        layout.addWidget(self.result2_label, 0, 1)
        layout.addWidget(self.diff_label, 0, 2)
        layout.addWidget(self.percent_label, 0, 3)
        layout.addWidget(self.status_label, 1, 0, 1, 4)

        return frame

    def _card(self, title):
        frame = QFrame()
        frame.setObjectName("card")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)

        label = QLabel(title)
        label.setObjectName("cardTitle")
        layout.addWidget(label)

        return frame

    def _result_metric(self, title, value):
        label = QLabel(
            f"<span style='color:#9aa6b6;'>{title}</span><br>"
            f"<b>{value}</b>"
        )
        label.setObjectName("bigMetric")
        label.setTextFormat(Qt.RichText)
        label.setAlignment(Qt.AlignCenter)
        return label

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #070c14;
            }

            QWidget {
                font-family: Calibri, Arial;
                font-size: 13px;
                color: #d7dde8;
            }

            #card, #bottomBar {
                background-color: #111a28;
                border: 1px solid #233044;
                border-radius: 10px;
            }

            #cardTitle {
                color: #dce6f5;
                font-size: 14px;
                font-weight: 700;
            }

            #smallLabel {
                color: #a9b4c5;
            }

            #roiInfo {
                color: #7dbdff;
                font-weight: 600;
            }

            QPushButton {
                background-color: #131f31;
                border: 1px solid #2a3950;
                border-radius: 7px;
                padding: 9px 14px;
                color: #d7dde8;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #18283f;
                border: 1px solid #3d5576;
            }

            #primaryButton {
                background-color: #0e345c;
                border: 1px solid #1d7fe8;
                color: #e8f5ff;
            }

            #cyanButton {
                background-color: #07313e;
                border: 1px solid #00c8ff;
                color: #dffaff;
            }

            QComboBox, QLineEdit {
                background-color: #0c1421;
                border: 1px solid #2a3950;
                border-radius: 6px;
                padding: 7px;
                color: #e5ecf8;
            }

            QCheckBox, QRadioButton {
                color: #d7dde8;
                spacing: 7px;
            }

            #bigMetric {
                background-color: #0d1624;
                border: 1px solid #202d40;
                border-radius: 9px;
                padding: 10px;
                font-size: 14px;
            }

            #bigMetric b {
                font-size: 24px;
                color: #00c8ff;
            }
        """)

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def get_method(self):
        return "CPP" if self.cpp_radio.isChecked() else "CPPS"

    def get_f0_range(self):
        try:
            f0_min = float(self.f0_min_entry.text())
            f0_max = float(self.f0_max_entry.text())
        except ValueError:
            raise ValueError("Invalid F0 range. Please enter numeric values.")

        if f0_min <= 0 or f0_max <= f0_min:
            raise ValueError("F0 Max must be greater than F0 Min.")

        return f0_min, f0_max

    def load_audio(self, slot: int):
        filters = "Audio files (" + " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_FORMATS) + ")"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Audio {slot}",
            "",
            filters,
        )

        if not file_path:
            return

        try:
            temp_wav = convert_to_temp_mono_wav(file_path)
            audio, sr = load_audio_as_mono(temp_wav)

            if slot == 1:
                self.file1_original = file_path
                self.file1_wav = temp_wav
                self.audio1 = audio
                self.sr1 = sr
                self.region1 = None
                self.result1 = None
                self.file1_label.setText(f"Audio 1: {Path(file_path).name}")
                self.plot_audio(slot=1)
                self.play1_btn.setEnabled(True)
                self.stop1_btn.setEnabled(True)


            else:
                self.file2_original = file_path
                self.file2_wav = temp_wav
                self.audio2 = audio
                self.sr2 = sr
                self.region2 = None
                self.result2 = None
                self.file2_label.setText(f"Audio 2: {Path(file_path).name}")
                self.plot_audio(slot=2)
                self.play2_btn.setEnabled(True)
                self.stop2_btn.setEnabled(True)

            self.status_label.setText("Audio loaded. Select ROI on each spectrogram.")

        except Exception as e:
            QMessageBox.critical(self, "Audio Error", str(e))

    def plot_audio(self, slot: int):
        f0_min, f0_max = self.get_f0_range()

        if slot == 1:
            ax = self.ax1
            canvas = self.canvas1
            fig = self.fig1
            wav = self.file1_wav

            if self.span1:
                try:
                    self.span1.disconnect_events()
                except Exception:
                    pass

            ax.clear()

            plot_praat_spectrogram(
                ax,
                wav,
                max_freq=5000,
                fmin=f0_min,
                fmax=f0_max,
                cmap="magma",
                dynamic_range_db=70,
            )

            fig.tight_layout()
            canvas.draw()

            self.span1 = SpanSelector(
                ax,
                lambda tmin, tmax: self.on_select(slot=1, tmin=tmin, tmax=tmax),
                "horizontal",
                useblit=True,
                props=dict(alpha=ROI_ALPHA, facecolor=ROI_COLOR),
                interactive=True,
            )

        else:
            ax = self.ax2
            canvas = self.canvas2
            fig = self.fig2
            wav = self.file2_wav

            if self.span2:
                try:
                    self.span2.disconnect_events()
                except Exception:
                    pass

            ax.clear()

            plot_praat_spectrogram(
                ax,
                wav,
                max_freq=5000,
                fmin=f0_min,
                fmax=f0_max,
                cmap="magma",
                dynamic_range_db=85,
            )

            fig.tight_layout()
            canvas.draw()

            self.span2 = SpanSelector(
                ax,
                lambda tmin, tmax: self.on_select(slot=2, tmin=tmin, tmax=tmax),
                "horizontal",
                useblit=True,
                props=dict(alpha=ROI_ALPHA, facecolor=ROI_COLOR),
                interactive=True,
            )

    def on_select(self, slot: int, tmin: float, tmax: float):
        if slot == 1:
            audio = self.audio1
            sr = self.sr1
            ax = self.ax1
            canvas = self.canvas1
        else:
            audio = self.audio2
            sr = self.sr2
            ax = self.ax2
            canvas = self.canvas2

        if audio is None or sr is None:
            return

        duration = len(audio) / sr
        tmin = max(0, min(duration, tmin))
        tmax = max(0, min(duration, tmax))

        if tmax <= tmin:
            return

        if slot == 1:
            if self.roi_patch1:
                try:
                    self.roi_patch1.remove()
                except Exception:
                    pass

            self.roi_patch1 = ax.axvspan(
                tmin, tmax, color=ROI_COLOR, alpha=ROI_ALPHA, zorder=4
            )
            self.region1 = (tmin, tmax)
            self.roi1_label.setText(f"ROI 1: {tmin:.2f} – {tmax:.2f} s ({tmax - tmin:.2f} s)")

        else:
            if self.roi_patch2:
                try:
                    self.roi_patch2.remove()
                except Exception:
                    pass

            self.roi_patch2 = ax.axvspan(
                tmin, tmax, color=ROI_COLOR, alpha=ROI_ALPHA, zorder=4
            )
            self.region2 = (tmin, tmax)
            self.roi2_label.setText(f"ROI 2: {tmin:.2f} – {tmax:.2f} s ({tmax - tmin:.2f} s)")

        canvas.draw()

    def refresh_spectrograms(self):
        try:
            if self.file1_wav:
                self.plot_audio(slot=1)
            if self.file2_wav:
                self.plot_audio(slot=2)

            self.status_label.setText("Spectrograms refreshed with current F0 range.")

        except Exception as e:
            QMessageBox.warning(self, "F0 Range Error", str(e))

    def clear_rois(self):
        self.region1 = None
        self.region2 = None

        if self.roi_patch1:
            try:
                self.roi_patch1.remove()
            except Exception:
                pass
            self.roi_patch1 = None

        if self.roi_patch2:
            try:
                self.roi_patch2.remove()
            except Exception:
                pass
            self.roi_patch2 = None

        self.roi1_label.setText("ROI 1: not selected")
        self.roi2_label.setText("ROI 2: not selected")

        self.canvas1.draw()
        self.canvas2.draw()

    def play_audio(self, slot: int):
        try:
            import sounddevice as sd

            sd.stop()

            if slot == 1:
                if self.audio1 is None or self.sr1 is None:
                    QMessageBox.information(self, "Audio 1", "Audio 1 is not loaded.")
                    return

                sd.play(self.audio1, self.sr1)
                self.status_label.setText("Playing Audio 1...")

            else:
                if self.audio2 is None or self.sr2 is None:
                    QMessageBox.information(self, "Audio 2", "Audio 2 is not loaded.")
                    return

                sd.play(self.audio2, self.sr2)
                self.status_label.setText("Playing Audio 2...")

        except Exception:
            QMessageBox.information(
                self,
                "Audio Playback",
                "Install sounddevice for playback:\n\npip install sounddevice",
            )

    def stop_audio(self):
        try:
            import sounddevice as sd
            sd.stop()
            self.status_label.setText("Playback stopped.")
        except Exception:
            pass

    def run_comparison(self):
        if not self.file1_wav or not self.file2_wav:
            QMessageBox.warning(self, "Missing Files", "Please load Audio 1 and Audio 2.")
            return

        method = self.get_method()
        file_type = self.file_type_combo.currentText()

        try:
            f0_min, f0_max = self.get_f0_range()

            self.status_label.setText("Running comparison...")
            self.repaint()

            self.result1 = extract_cpp(
                self.file1_wav,
                region=self.region1,
                method=method,
                file_type=file_type,
                min_f0=f0_min,
                max_f0=f0_max,
                vad_enabled=self.vad_check.isChecked(),
                pause_removal_enabled=self.pause_check.isChecked(),
            )

            self.result2 = extract_cpp(
                self.file2_wav,
                region=self.region2,
                method=method,
                file_type=file_type,
                min_f0=f0_min,
                max_f0=f0_max,
                vad_enabled=self.vad_check.isChecked(),
                pause_removal_enabled=self.pause_check.isChecked(),
            )

            value1 = self.result1.get("cpp")
            value2 = self.result2.get("cpp")

            if value1 is None or value2 is None:
                QMessageBox.warning(self, "Analysis Error", "One of the files did not return a valid value.")
                return

            diff = value2 - value1

            if value1 != 0:
                percent = (diff / value1) * 100.0
                percent_text = f"{percent:+.2f}%"
            else:
                percent_text = "N/A"

            self.result1_label.setText(
                f"<span style='color:#9aa6b6;'>Audio 1</span><br><b>{value1:.2f} dB</b>"
            )
            self.result2_label.setText(
                f"<span style='color:#9aa6b6;'>Audio 2</span><br><b>{value2:.2f} dB</b>"
            )
            self.diff_label.setText(
                f"<span style='color:#9aa6b6;'>Difference</span><br><b>{diff:+.2f} dB</b>"
            )
            self.percent_label.setText(
                f"<span style='color:#9aa6b6;'>Percent Change</span><br><b>{percent_text}</b>"
            )

            csv_path = self.save_comparison_csv(
                method=method,
                file_type=file_type,
                value1=value1,
                value2=value2,
                diff=diff,
                percent_text=percent_text,
                f0_min=f0_min,
                f0_max=f0_max,
            )

            self.status_label.setText(f"Comparison complete. CSV saved: {csv_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Comparison Error",
                f"{e}\n\n{traceback.format_exc()}",
            )

    def save_comparison_csv(
        self,
        method,
        file_type,
        value1,
        value2,
        diff,
        percent_text,
        f0_min,
        f0_max,
    ):
        base_dir = Path(self.file1_original).parent if self.file1_original else Path.cwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = base_dir / f"cepstralvox_comparison_{method}_{timestamp}.csv"

        region1 = self.region1 or (None, None)
        region2 = self.region2 or (None, None)

        rows = [
            {
                "audio": "Audio 1",
                "filename": Path(self.file1_original).name,
                "file_type": file_type,
                "analysis_method": method,
                "value_db": f"{value1:.3f}",
                "roi_start": f"{region1[0]:.3f}" if region1[0] is not None else "",
                "roi_end": f"{region1[1]:.3f}" if region1[1] is not None else "",
                "f0_min": f"{f0_min:.1f}",
                "f0_max": f"{f0_max:.1f}",
                "voiced_only_extraction": self.vad_check.isChecked(),
                "pause_removal": self.pause_check.isChecked(),
            },
            {
                "audio": "Audio 2",
                "filename": Path(self.file2_original).name,
                "file_type": file_type,
                "analysis_method": method,
                "value_db": f"{value2:.3f}",
                "roi_start": f"{region2[0]:.3f}" if region2[0] is not None else "",
                "roi_end": f"{region2[1]:.3f}" if region2[1] is not None else "",
                "f0_min": f"{f0_min:.1f}",
                "f0_max": f"{f0_max:.1f}",
                "voiced_only_extraction": self.vad_check.isChecked(),
                "pause_removal": self.pause_check.isChecked(),
            },
            {
                "audio": "Comparison",
                "filename": "Audio 2 - Audio 1",
                "file_type": file_type,
                "analysis_method": method,
                "value_db": f"{diff:.3f}",
                "roi_start": "",
                "roi_end": "",
                "f0_min": f"{f0_min:.1f}",
                "f0_max": f"{f0_max:.1f}",
                "voiced_only_extraction": self.vad_check.isChecked(),
                "pause_removal": self.pause_check.isChecked(),
            },
            {
                "audio": "Percent Change",
                "filename": percent_text,
                "file_type": file_type,
                "analysis_method": method,
                "value_db": "",
                "roi_start": "",
                "roi_end": "",
                "f0_min": f"{f0_min:.1f}",
                "f0_max": f"{f0_max:.1f}",
                "voiced_only_extraction": self.vad_check.isChecked(),
                "pause_removal": self.pause_check.isChecked(),
            },
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return str(csv_path)