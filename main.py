from __future__ import annotations

import os
import csv
import shutil
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QRadioButton, QButtonGroup, QCheckBox, QLineEdit,
    QFrame, QSizePolicy, QDialog
)

from cpp_analysis import extract_cpp, batch_extract_cpp
from spectrogram import plot_praat_spectrogram, get_f0_summary
from audio_io import (
    SUPPORTED_AUDIO_FORMATS,
    convert_to_temp_mono_wav,
    load_audio_as_mono,
    cleanup_temp_audio_files,
)

from plot_utils import plot_quefrency

APP_VERSION = "2.0.0"

ROI_COLOR = "#ff3b3b"
ROI_ALPHA = 0.28

def resource_path(filename: str) -> str:
    """
    Finds resources both when running from source and when packaged.
    """
    base_dir = Path(getattr(__import__("sys"), "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_dir / filename)


def plot_quefrency_figure(res, method, save_path=None, show=False):
    q = np.array(res["quefrency"]) * 1000
    s = res["spectrum"]
    trend = res["trend"]
    val = res.get("cpp", None)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#0b111c")
    ax.set_facecolor("#111a28")

    ax.plot(q, s, color="#00c8ff", linewidth=1.8, label=f"{method} Cepstrum")

    if trend is not None:
        ax.plot(q, trend, "--", color="#ff4fb3", linewidth=1.5, label=f"{method} Trend")

    q_peak = y_peak = f0_peak = None
    mask = (q >= 2) & (q <= 12)

    if np.any(mask):
        x_roi = q[mask]
        y_roi = s[mask]
        peak_idx = np.argmax(y_roi)

        q_peak = x_roi[peak_idx]
        y_peak = y_roi[peak_idx]
        f0_peak = 1.0 / (q_peak / 1000)

        ax.scatter([q_peak], [y_peak], color="#ff4b4b", s=48, zorder=5)
        ax.text(
            q_peak,
            y_peak + 4,
            f"{q_peak:.2f} ms",
            color="#ff6868",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Quefrency (ms)", color="#d7dde8")
    ax.set_ylabel("Amplitude (dB)", color="#d7dde8")
    ax.set_title(f"Quefrency Spectrum ({method})", color="#e8edf7", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#b8c0cc")

    for spine in ax.spines.values():
        spine.set_color("#344155")

    leg = ax.legend(frameon=True, facecolor="#101826", edgecolor="#2a3548")
    for text in leg.get_texts():
        text.set_color("#e8edf7")

    if val is not None and q_peak is not None and f0_peak is not None:
        title = f"{method} = {val:.2f} dB | Quefrency: {q_peak / 1000:.3f} s | F0: {f0_peak:.2f} Hz"
        fig.suptitle(title, color="#ffffff", fontsize=12, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())

    if show:
        plt.show()

    plt.close(fig)


class CepstralVoxQt(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"CepstralVox version {APP_VERSION}")
        self.resize(1650, 930)

        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.original_audio_path = None
        self.audio_path = None
        self.temp_audio_path = None
        self.audio_data = None
        self.sr = None
        self.region = None

        self.analysis_result = None
        self.analysis_method = None
        self.results_type = None
        self.batch_results = []

        self.span = None
        self.roi_patch = None

        self._build_ui()
        self._apply_theme()

    # ---------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        self.sidebar = self._build_sidebar()
        main_layout.addWidget(self.sidebar)

        content = QVBoxLayout()
        content.setSpacing(12)
        main_layout.addLayout(content, stretch=1)

        content.addLayout(self._build_top_cards())
        content.addLayout(self._build_middle_area(), stretch=1)
        content.addWidget(self._build_bottom_bar())

    def _build_sidebar(self):
        frame = QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(205)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 24, 18, 18)
        layout.setSpacing(16)

        logo_img = QLabel()
        logo_img.setObjectName("logoImage")
        logo_img.setAlignment(Qt.AlignCenter)

        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(
                165, 165,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            logo_img.setPixmap(pixmap)
        else:
            logo_img.setText("Cepstral<span style='color:#00b7ff;'>Vox</span>")
            logo_img.setTextFormat(Qt.RichText)

        layout.addWidget(logo_img)

        version = QLabel(f"v{APP_VERSION}")
        version.setAlignment(Qt.AlignCenter)
        version.setObjectName("versionLabel")
        layout.addWidget(version)

        layout.addSpacing(24)

        self.nav_analysis = QPushButton("  Analysis")
        self.nav_analysis.setObjectName("navActive")
        layout.addWidget(self.nav_analysis)

        self.nav_compare = QPushButton("  Compare Files")
        self.nav_compare.setObjectName("navButton")
        self.nav_compare.clicked.connect(self.open_comparison_module)
        layout.addWidget(self.nav_compare)

        self.nav_batch = QPushButton("  Batch Process")
        self.nav_batch.setObjectName("navButton")
        layout.addWidget(self.nav_batch)

        self.nav_about = QPushButton("  About")
        self.nav_about.setObjectName("navButton")
        self.nav_about.clicked.connect(self.show_about)
        layout.addWidget(self.nav_about)

        layout.addStretch()

        self.status_dot = QLabel("●  Ready")
        self.status_dot.setObjectName("statusReady")
        layout.addWidget(self.status_dot)

        return frame

    def _build_top_cards(self):
        layout = QGridLayout()
        layout.setSpacing(12)

        file_card = self._card("File & Signal")
        file_layout = file_card.layout()

        self.open_btn = QPushButton("Open Audio File")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.clicked.connect(self.load_audio)
        file_layout.addWidget(self.open_btn)

        self.loaded_file_label = QLabel("No file loaded")
        self.loaded_file_label.setObjectName("mutedLabel")
        file_layout.addWidget(self.loaded_file_label)

        self.signal_info_label = QLabel("Sample Rate: -\nChannels: -")
        self.signal_info_label.setObjectName("smallLabel")
        file_layout.addWidget(self.signal_info_label)

        analysis_card = self._card("Analysis Mode")
        analysis_layout = analysis_card.layout()

        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["Sustained vowel", "Connected speech"])
        analysis_layout.addWidget(self.file_type_combo)

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
        analysis_layout.addLayout(radio_row)

        f0_card = self._card("Frequency Range (F0)")
        f0_layout = f0_card.layout()

        f0_row = QHBoxLayout()
        self.f0_min_entry = QLineEdit("60")
        self.f0_max_entry = QLineEdit("330")
        self.f0_min_entry.setFixedWidth(58)
        self.f0_max_entry.setFixedWidth(58)

        f0_row.addWidget(QLabel("Min"))
        f0_row.addWidget(self.f0_min_entry)
        f0_row.addWidget(QLabel("Max"))
        f0_row.addWidget(self.f0_max_entry)
        f0_layout.addLayout(f0_row)

        self.set_f0_btn = QPushButton("Set F0 Range")
        self.set_f0_btn.clicked.connect(self.set_f0_range)
        f0_layout.addWidget(self.set_f0_btn)

        processing_card = self._card("Processing Options")
        processing_layout = processing_card.layout()

        self.vad_check = QCheckBox("Enable voiced-only extraction (VAD)")
        self.vad_check.setChecked(True)
        self.pause_check = QCheckBox("Enable pause removal")
        self.pause_check.setChecked(True)

        processing_layout.addWidget(self.vad_check)
        processing_layout.addWidget(self.pause_check)

        self.batch_btn = QPushButton("Batch Process")
        self.batch_btn.setObjectName("purpleButton")
        self.batch_btn.clicked.connect(self.batch_process)
        processing_layout.addWidget(self.batch_btn)

        overview_card = self._card("Overview")
        overview_layout = overview_card.layout()

        self.total_duration_label = QLabel("Total Duration\n-")
        self.selected_duration_label = QLabel("Selected Duration\n-")
        self.voiced_roi_label = QLabel("Voiced % (ROI)\n-")

        for w in [self.total_duration_label, self.selected_duration_label, self.voiced_roi_label]:
            w.setObjectName("metricSmall")
            overview_layout.addWidget(w)

        layout.addWidget(file_card, 0, 0)
        layout.addWidget(analysis_card, 0, 1)
        layout.addWidget(f0_card, 0, 2)
        layout.addWidget(processing_card, 0, 3)
        layout.addWidget(overview_card, 0, 4)

        layout.setColumnStretch(3, 2)

        return layout

    def _build_middle_area(self):
        layout = QHBoxLayout()
        layout.setSpacing(12)

        spectro_card = QFrame()
        spectro_card.setObjectName("card")
        spectro_layout = QVBoxLayout(spectro_card)
        spectro_layout.setContentsMargins(16, 14, 16, 12)
        spectro_layout.setSpacing(8)

        title_row = QHBoxLayout()
        self.spectro_title = QLabel("Spectrogram & F0 (autocorrelation)")
        self.spectro_title.setObjectName("sectionTitle")
        title_row.addWidget(self.spectro_title)
        title_row.addStretch()

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        title_row.addWidget(self.reset_zoom_btn)

        spectro_layout.addLayout(title_row)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor("#111a28")

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spectro_layout.addWidget(self.canvas, stretch=1)

        self.roi_info_label = QLabel("ROI: not selected")
        self.roi_info_label.setAlignment(Qt.AlignCenter)
        self.roi_info_label.setObjectName("roiInfo")
        spectro_layout.addWidget(self.roi_info_label)

        layout.addWidget(spectro_card, stretch=1)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        live_card = self._card("Live Measurements")
        live_layout = live_card.layout()

        self.mean_f0_label = self._big_metric("Mean F0", "-")
        self.median_f0_label = self._big_metric("Median F0", "-")
        self.std_f0_label = self._big_metric("F0 Std Dev", "-")
        self.cpp_value_label = self._big_metric("CPP", "-")
        self.cpps_value_label = self._big_metric("CPPS", "-")
        self.voiced_value_label = self._big_metric("Voiced %", "-")

        for w in [
            self.mean_f0_label,
            self.median_f0_label,
            self.std_f0_label,
            self.cpp_value_label,
            self.cpps_value_label,
            self.voiced_value_label,
        ]:
            live_layout.addWidget(w)

        quick_card = self._card("Quick Summary")
        quick_card.setObjectName("quickSummaryCard")
        quick_layout = quick_card.layout()

        self.quick_summary_label = QLabel("Load an audio file to begin.")
        self.quick_summary_label.setObjectName("quickSummaryText")
        self.quick_summary_label.setWordWrap(True)
        self.quick_summary_label.setAlignment(Qt.AlignCenter)

        quick_layout.addWidget(self.quick_summary_label)

        right_panel.addWidget(live_card)
        right_panel.addWidget(quick_card)
        right_panel.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(295)

        layout.addWidget(right_widget)

        return layout

    def _build_bottom_bar(self):
        frame = QFrame()
        frame.setObjectName("bottomBar")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("cyanButton")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)

        self.show_quef_btn = QPushButton("Show Quefrency Plot")
        self.show_quef_btn.setObjectName("purpleButton")
        self.show_quef_btn.clicked.connect(self.show_quefrency_plot)
        self.show_quef_btn.setEnabled(False)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setObjectName("greenButton")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()
        layout.addWidget(self.run_btn)
        layout.addWidget(self.show_quef_btn)
        layout.addWidget(self.export_btn)

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

    def _big_metric(self, name, value):
        label = QLabel(f"<span style='color:#9aa6b6;'>{name}</span><br><b>{value}</b>")
        label.setObjectName("bigMetric")
        label.setTextFormat(Qt.RichText)
        return label

    # ---------------------------------------------------------------------
    # Theme
    # ---------------------------------------------------------------------

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

            #sidebar {
                background-color: #07111d;
                border: 1px solid #1d2a3a;
                border-radius: 10px;
            }

            #logo {
                font-size: 30px;
                font-weight: 700;
                color: #e8edf7;
            }

            #versionLabel {
                color: #8c98a9;
                font-size: 13px;
            }

            #card, #bottomBar {
                background-color: #111a28;
                border: 1px solid #233044;
                border-radius: 10px;
            }

            #cardTitle, #sectionTitle {
                color: #dce6f5;
                font-size: 14px;
                font-weight: 700;
            }

            #mutedLabel {
                color: #8d99aa;
            }

            #smallLabel {
                color: #a9b4c5;
                line-height: 135%;
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

            QPushButton:disabled {
                color: #586476;
                background-color: #101722;
                border: 1px solid #1d2735;
            }

            #primaryButton, #playButton {
                background-color: #0e345c;
                border: 1px solid #1d7fe8;
                color: #e8f5ff;
            }

            #cyanButton {
                background-color: #07313e;
                border: 1px solid #00c8ff;
                color: #dffaff;
            }

            #purpleButton {
                background-color: #25143e;
                border: 1px solid #8b4dff;
                color: #f1e8ff;
            }

            #greenButton {
                background-color: #12371f;
                border: 1px solid #33b96b;
                color: #e7fff1;
            }

            #navButton {
                text-align: left;
                background-color: transparent;
                border: none;
                color: #9ba8ba;
                padding: 12px 14px;
            }

            #navButton:hover {
                background-color: #101b2b;
                color: #e8edf7;
            }

            #navActive {
                text-align: left;
                background-color: #10243a;
                border-left: 3px solid #00b7ff;
                color: #dff6ff;
                padding: 12px 14px;
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
                font-size: 22px;
                color: #00c8ff;
            }

            #metricSmall {
                color: #c5cedb;
                font-size: 13px;
            }

            #statusReady {
                color: #4be07d;
                font-weight: 600;
            }
            
                        #logoImage {
                padding: 6px;
            }

            #quickSummaryCard {
                background-color: #091f2e;
                border: 1px solid #00c8ff;
                border-radius: 10px;
            }

            #quickSummaryText {
                color: #eaffff;
                font-size: 16px;
                font-weight: 700;
                line-height: 150%;
                padding: 10px;
            }
        """)

    # ---------------------------------------------------------------------
    # Logic
    # ---------------------------------------------------------------------

    def get_method(self):
        return "CPP" if self.cpp_radio.isChecked() else "CPPS"

    def get_f0_range(self):
        try:
            return float(self.f0_min_entry.text()), float(self.f0_max_entry.text())
        except ValueError:
            raise ValueError("Invalid F0 range. Please enter numeric values.")

    def load_audio(self):
        filters = "Audio files (" + " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_FORMATS) + ")"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            "",
            filters,
        )

        if not file_path:
            return

        try:
            self.original_audio_path = file_path
            self.audio_path = convert_to_temp_mono_wav(file_path)
            self.temp_audio_path = self.audio_path
            self.audio_data, self.sr = load_audio_as_mono(self.audio_path)

            self.region = None
            self.analysis_result = None
            self.batch_results = []

            self.loaded_file_label.setText(Path(file_path).name)
            duration = len(self.audio_data) / self.sr

            self.signal_info_label.setText(
                f"16-bit PCM  •  Mono\nSample Rate: {self.sr / 1000:.2f} kHz"
            )
            self.total_duration_label.setText(f"Total Duration\n{duration:.2f} s")
            self.selected_duration_label.setText("Selected Duration\n-")
            self.voiced_roi_label.setText("Voiced % (ROI)\n-")

            self.show_spectrogram()

            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.show_quef_btn.setEnabled(False)

            self.status_dot.setText("●  Audio loaded")

        except Exception as e:
            QMessageBox.critical(self, "Audio Error", str(e))

    def open_comparison_module(self):
        try:
            from comparison_qt import ComparisonWindow
            self.comparison_window = ComparisonWindow(parent=self)
            self.comparison_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Comparison Module Error", str(e))

    def show_spectrogram(self):
        if not self.audio_path:
            return

        self.ax.clear()

        f0_min, f0_max = self.get_f0_range()

        plot_praat_spectrogram(
            self.ax,
            self.audio_path,
            max_freq=5000,
            fmin=f0_min,
            fmax=f0_max,
            cmap="magma",
            dynamic_range_db=85,
        )

        self.fig.tight_layout()
        self.canvas.draw()

        if self.span:
            try:
                self.span.disconnect_events()
            except Exception:
                pass

        self.span = SpanSelector(
            self.ax,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=ROI_ALPHA, facecolor=ROI_COLOR),
            interactive=True,
        )

        self.update_f0_metrics(region=None)

    def on_select(self, tmin, tmax):
        if self.audio_data is None or self.sr is None:
            return

        duration = len(self.audio_data) / self.sr
        tmin = max(0, min(duration, tmin))
        tmax = max(0, min(duration, tmax))

        if tmax <= tmin:
            return

        if self.roi_patch:
            try:
                self.roi_patch.remove()
            except Exception:
                pass

        self.roi_patch = self.ax.axvspan(
            tmin,
            tmax,
            color=ROI_COLOR,
            alpha=ROI_ALPHA,
            zorder=4,
        )

        self.region = (tmin, tmax)

        roi_dur = tmax - tmin
        self.roi_info_label.setText(f"ROI: {tmin:.2f} – {tmax:.2f} s   ({roi_dur:.2f} s)")
        self.selected_duration_label.setText(f"Selected Duration\n{roi_dur:.2f} s")

        self.canvas.draw()
        self.update_f0_metrics(region=self.region)

    def update_f0_metrics(self, region=None):
        if not self.audio_path:
            return

        try:
            f0_min, f0_max = self.get_f0_range()

            summary = get_f0_summary(
                self.audio_path,
                fmin=f0_min,
                fmax=f0_max,
                region=region,
            )

            def fmt(value, suffix="Hz"):
                return "-" if value is None else f"{value:.1f} {suffix}"

            self.mean_f0_label.setText(
                f"<span style='color:#9aa6b6;'>Mean F0</span><br><b>{fmt(summary['mean_f0'])}</b>"
            )
            self.median_f0_label.setText(
                f"<span style='color:#9aa6b6;'>Median F0</span><br><b>{fmt(summary['median_f0'])}</b>"
            )
            self.std_f0_label.setText(
                f"<span style='color:#9aa6b6;'>F0 Std Dev</span><br><b>{fmt(summary['std_f0'])}</b>"
            )
            self.voiced_value_label.setText(
                f"<span style='color:#9aa6b6;'>Voiced %</span><br><b>{summary['voiced_percent']:.1f}%</b>"
            )
            self.voiced_roi_label.setText(f"Voiced % (ROI)\n{summary['voiced_percent']:.1f}%")

        except Exception:
            pass

    def reset_zoom(self):
        if self.audio_path:
            self.show_spectrogram()

    def set_f0_range(self):
        try:
            f0_min, f0_max = self.get_f0_range()

            if f0_min <= 0 or f0_max <= f0_min:
                raise ValueError("F0 Max must be greater than F0 Min.")

            QMessageBox.information(
                self,
                "F0 Range",
                f"F0 range set to {f0_min:.1f}–{f0_max:.1f} Hz.",
            )

            if self.audio_path:
                self.show_spectrogram()

        except Exception as e:
            QMessageBox.warning(self, "Invalid F0 Range", str(e))

    def run_analysis(self):
        if self.audio_path is None:
            QMessageBox.warning(self, "Error", "Load an audio file first.")
            return

        method = self.get_method()
        file_type = self.file_type_combo.currentText()

        try:
            f0_min, f0_max = self.get_f0_range()

            result = extract_cpp(
                self.audio_path,
                region=self.region,
                method=method,
                file_type=file_type,
                min_f0=f0_min,
                max_f0=f0_max,
                vad_enabled=self.vad_check.isChecked(),
                pause_removal_enabled=self.pause_check.isChecked(),
            )

            self.analysis_result = result
            self.analysis_method = method
            self.results_type = file_type
            self.batch_results = []

            val = result.get("cpp")

            if val is None:
                QMessageBox.warning(self, "Analysis", "No valid CPP/CPPS value was returned.")
                return

            if method == "CPP":
                self.cpp_value_label.setText(
                    f"<span style='color:#9aa6b6;'>CPP</span><br><b>{val:.2f} dB</b>"
                )
            else:
                self.cpps_value_label.setText(
                    f"<span style='color:#9aa6b6;'>CPPS</span><br><b>{val:.2f} dB</b>"
                )

            roi_text = "-"
            if self.region:
                roi_text = f"{self.region[0]:.2f}–{self.region[1]:.2f} s"

            self.quick_summary_label.setText(
                f"<span style='color:#00c8ff; font-size:17px;'>{method}</span><br>"
                f"<span style='color:#ffffff; font-size:28px;'>{val:.2f} dB</span><br><br>"
                f"<span style='color:#b7c6d8;'>File type:</span> {file_type}<br>"
                f"<span style='color:#b7c6d8;'>ROI:</span> {roi_text}<br>"
                f"<span style='color:#b7c6d8;'>F0 range:</span> {f0_min:.1f}–{f0_max:.1f} Hz"
            )

            outpath = os.path.splitext(self.original_audio_path or self.audio_path)[0] + f"_{method}_quefrency.png"
            plot_quefrency_figure(result, method, save_path=outpath, show=False)

            self.export_btn.setEnabled(True)
            self.show_quef_btn.setEnabled(True)
            self.status_dot.setText("●  Analysis complete")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"{e}\n\n{traceback.format_exc()}")

    def show_quefrency_plot(self):
        if not self.analysis_result:
            QMessageBox.information(self, "No Data", "No analysis result available.")
            return

        if self.analysis_result.get("quefrency") is None or self.analysis_result.get("spectrum") is None:
            QMessageBox.information(self, "No Data", "No quefrency data available.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Quefrency Plot - {self.analysis_method}")
        dialog.resize(900, 520)

        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            dialog.setWindowIcon(QIcon(icon_path))

        dialog.setStyleSheet("""
            QDialog {
                background-color: #07111d;
            }

            QPushButton {
                background-color: #0e345c;
                border: 1px solid #1d7fe8;
                border-radius: 7px;
                padding: 9px 24px;
                color: #ffffff;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #155080;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        fig.patch.set_facecolor("#07111d")

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, stretch=1)

        plot_quefrency(
            ax,
            self.analysis_result["quefrency"],
            self.analysis_result["spectrum"],
            trend=self.analysis_result.get("trend"),
            label=f"{self.analysis_method} Cepstrum",
            method=self.analysis_method,
            value=self.analysis_result.get("cpp"),
        )

        fig.tight_layout()
        canvas.draw()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dialog.exec()

    def batch_process(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with WAV files")

        if not folder:
            return

        method = self.get_method()
        file_type = self.file_type_combo.currentText()

        try:
            f0_min, f0_max = self.get_f0_range()

            results = batch_extract_cpp(
                folder,
                method=method,
                file_type=file_type,
                min_f0=f0_min,
                max_f0=f0_max,
                vad_enabled=self.vad_check.isChecked(),
                pause_removal_enabled=self.pause_check.isChecked(),
            )

            self.batch_results = results
            self.analysis_result = None
            self.analysis_method = method

            ok = len([r for r in results if "cpp" in r and r.get("cpp") is not None])
            err = len([r for r in results if "error" in r])

            self.quick_summary_label.setText(
                f"<span style='color:#00c8ff; font-size:17px;'>Batch complete</span><br>"
                f"<span style='color:#ffffff; font-size:24px;'>{method}</span><br><br>"
                f"<span style='color:#4be07d;'>Processed:</span> {ok}<br>"
                f"<span style='color:#ff6868;'>Errors:</span> {err}"
            )

            plot_dir = os.path.join(folder, "quefrency_plots")
            os.makedirs(plot_dir, exist_ok=True)

            for r in results:
                if r.get("quefrency") is not None and r.get("spectrum") is not None:
                    base = os.path.splitext(r.get("filename", "unnamed"))[0]
                    save_path = os.path.join(plot_dir, f"{base}_{method}_quefrency.png")
                    plot_quefrency_figure(r, method, save_path=save_path, show=False)

            self.export_btn.setEnabled(True)
            self.status_dot.setText("●  Batch complete")

        except Exception as e:
            QMessageBox.critical(self, "Batch Error", str(e))

    def export_csv(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV files (*.csv)",
        )

        if not save_path:
            return

        try:
            f0_min, f0_max = self.get_f0_range()

            if self.analysis_result and not self.batch_results:
                region = self.region or (None, None)

                row = {
                    "filename": os.path.basename(self.original_audio_path or self.audio_path),
                    "internal_wav": os.path.basename(self.audio_path),
                    "file_type": self.results_type,
                    "analysis_method": self.analysis_method,
                    "value_db": f"{self.analysis_result['cpp']:.3f}",
                    "roi_start": f"{region[0]:.3f}" if region[0] is not None else "",
                    "roi_end": f"{region[1]:.3f}" if region[1] is not None else "",
                    "f0_min": f"{f0_min:.1f}",
                    "f0_max": f"{f0_max:.1f}",
                    "voiced_only_extraction": self.vad_check.isChecked(),
                    "pause_removal": self.pause_check.isChecked(),
                }

                with open(save_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)

            elif self.batch_results:
                with open(save_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "filename", "file_type", "analysis_method", "value_db",
                        "roi_start", "roi_end", "f0_min", "f0_max",
                        "voiced_only_extraction", "pause_removal"
                    ])

                    for r in self.batch_results:
                        region = r.get("region", (None, None))
                        writer.writerow([
                            r.get("filename", ""),
                            self.file_type_combo.currentText(),
                            self.analysis_method,
                            f"{r.get('cpp', ''):.3f}" if r.get("cpp") is not None else "",
                            f"{region[0]:.3f}" if region and region[0] is not None else "",
                            f"{region[1]:.3f}" if region and region[1] is not None else "",
                            f"{f0_min:.1f}",
                            f"{f0_max:.1f}",
                            self.vad_check.isChecked(),
                            self.pause_check.isChecked(),
                        ])

            self.status_dot.setText("●  CSV exported")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def play_audio(self):
        if self.audio_data is None or self.sr is None:
            return

        try:
            import sounddevice as sd
            sd.stop()
            sd.play(self.audio_data, self.sr)
            self.status_dot.setText("●  Playing")
        except Exception:
            QMessageBox.information(
                self,
                "Audio",
                "Install sounddevice for playback:\n\npip install sounddevice",
            )

    def stop_audio(self):
        try:
            import sounddevice as sd
            sd.stop()
            self.status_dot.setText("●  Stopped")
        except Exception:
            pass

    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About CepstralVox")
        dialog.setFixedSize(620, 360)

        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            dialog.setWindowIcon(QIcon(icon_path))

        dialog.setStyleSheet("""
            QDialog {
                background-color: #07111d;
            }

            QLabel {
                color: #d7dde8;
                font-family: Calibri, Arial;
                font-size: 14px;
            }

            #aboutTitle {
                color: #00c8ff;
                font-size: 20px;
                font-weight: 700;
            }

            #aboutText {
                color: #d7dde8;
                font-size: 14px;
                line-height: 140%;
            }

            #aboutContact {
                color: #8fdcff;
                font-size: 14px;
            }

            QPushButton {
                background-color: #0e345c;
                border: 1px solid #1d7fe8;
                border-radius: 7px;
                padding: 9px 24px;
                color: #ffffff;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #155080;
            }
        """)

        root = QHBoxLayout(dialog)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(24)

        logo_box = QVBoxLayout()
        logo_box.setAlignment(Qt.AlignCenter)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)

        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(
                190, 190,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("CepstralVox")

        logo_box.addWidget(logo_label)

        right = QVBoxLayout()
        right.setSpacing(12)

        title = QLabel(f"CepstralVox version {APP_VERSION}")
        title.setObjectName("aboutTitle")

        text = QLabel(
            "Cepstral analysis software for CPP and CPPS extraction.\n\n"
            "Version 2.0 introduces a PySide6 interface, modern visual design, "
            "multi-format audio loading, colored spectrogram visualization, "
            "and F0 display based on autocorrelation.\n\n"
            "Citation: Cruz, Tiago Lima Bicalho. CepstralVox: A User-Friendly, "
            "Open-Source Tool for Cepstral Voice Analysis. Journal of Voice (2025).\n\n"
            "Fonotech Academy"
        )
        text.setObjectName("aboutText")
        text.setWordWrap(True)

        contact = QLabel(
            "fonotechacademy@gmail.com\n"
            "www.fonotechacademy.com"
        )
        contact.setObjectName("aboutContact")

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)

        right.addWidget(title)
        right.addWidget(text)
        right.addWidget(contact)
        right.addStretch()
        right.addWidget(ok_btn, alignment=Qt.AlignRight)

        root.addLayout(logo_box, stretch=1)
        root.addLayout(right, stretch=2)

        dialog.exec()

    def closeEvent(self, event):
        try:
            self.stop_audio()
        except Exception:
            pass

        try:
            plt.close("all")
        except Exception:
            pass

        try:
            cleanup_temp_audio_files()
        except Exception:
            pass

        temp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_praat")
        try:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
        except Exception:
            pass

        event.accept()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    win = CepstralVoxQt()
    win.show()
    sys.exit(app.exec())