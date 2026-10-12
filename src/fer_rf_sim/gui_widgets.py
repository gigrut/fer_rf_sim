"""PySide6 desktop shell for the FER simulation and analysis services."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event

import numpy as np
from fer_rf_sim.analysis import build_current_array, constant_current_slice
from fer_rf_sim.legacy_config import REAL_DATA_CSV_PATH, REAL_DATA_RF_MAPPING
from fer_rf_sim.real_data import RealSpectrum, load_real_data
from fer_rf_sim.repository import SimulationData, load_simulation
from fer_rf_sim.service import SimulationConfig, SimulationRunner
from .config import load_config
from .math_reference import MATH_SECTIONS, math_reference_html
from .peak_widgets import PeakAnalysisPanel
from .plotly_host import PlotlyDocument, PlotlyHost
from .quadrature import chebyshev_phases
from .transmission_plotting import requested_transmission_grid, simulation_transmission_source
from dataclasses import replace
from collections import OrderedDict


TAB_NAMES = (
    "Config + Run",
    "Simulated Data",
    "Real Data",
    "Peak Extraction",
    "RF Dependence",
    "Math Reference",
)


from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QGridLayout, QGroupBox, QMainWindow, QPushButton, QScrollArea, QSlider,
    QSpinBox, QDoubleSpinBox, QStyle, QTabWidget, QTextBrowser, QTextEdit, QToolButton,
    QProgressBar,
    QVBoxLayout, QWidget,
)
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None

class SimulationWorker(QObject):
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(str, int, int)
    finished = Signal()

    def __init__(self, config: SimulationConfig, cancel_event: Event):
        super().__init__()
        self.config = config
        self.cancel_event = cancel_event

    @Slot()
    def run(self):
        try:
            artifact = SimulationRunner().run(
                self.config, progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as error:  # pragma: no cover - Qt callback boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(str(artifact))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, *, config=None, web_engine_view=QWebEngineView, auto_load_real_data=True):
        super().__init__()
        self.config = config or load_config()
        self.auto_load_real_data = auto_load_real_data
        self._close_requested = False
        self.plot_document = PlotlyDocument() if web_engine_view is not None else None
        self.simulated_host = None
        self.real_host = None
        self.current_array = None
        self.setWindowTitle("FER RF STM Simulation")
        self.resize(1100, 740)
        self.setMinimumSize(760, 520)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QMainWindow { background: #f4f6f6; color: #242d30; }
            QWidget { color: #242d30; }
            QTabWidget::pane { border: 0; background: #ffffff; }
            QTabBar::tab {
                background: #f4f6f6; color: #647176;
                padding: 13px 20px; border: 0;
                border-bottom: 3px solid transparent;
            }
            QTabBar::tab:selected {
                background: #ffffff; color: #087f78;
                border-bottom-color: #087f78;
            }
            QTabBar::tab:hover:!selected { background: #e9efef; }
            QLabel[role="title"] { font-size: 22px; font-weight: 600; }
            QLabel[role="section"] { font-size: 14px; font-weight: 600; }
            QLabel[role="muted"] { color: #647176; }
            QLabel[role="value"] {
                color: #087f78; font-weight: 600; min-width: 75px;
            }
            QGroupBox {
                background: #ffffff; border: 1px solid #dce4e4;
                border-radius: 6px; margin-top: 14px; padding: 16px 12px 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px;
                padding: 0 6px; color: #39484d;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #ffffff; border: 1px solid #cdd8da;
                border-radius: 4px; padding: 6px 9px;
                min-height: 24px; selection-background-color: #087f78;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #087f78; background: #f6fcfb;
            }
            QComboBox QAbstractItemView {
                background: #ffffff; color: #242d30;
                selection-background-color: #def1ed; selection-color: #075e59;
            }
            QPushButton, QToolButton {
                background: #ffffff; border: 1px solid #cdd8da;
                border-radius: 4px; padding: 7px 13px; min-height: 20px;
                font-weight: 600;
            }
            QToolButton { padding: 0; }
            QPushButton:hover, QToolButton:hover {
                background: #edf5f3; border-color: #94b8b2;
            }
            QPushButton:pressed, QToolButton:pressed { background: #dcece8; }
            QPushButton:focus, QToolButton:focus { border-color: #087f78; }
            QPushButton[role="primary"] {
                background: #087f78; color: #ffffff; border-color: #087f78;
            }
            QPushButton[role="primary"]:hover { background: #066a64; }
            QPushButton[role="primary"]:pressed { background: #055953; }
            QPushButton:disabled, QToolButton:disabled {
                background: #edf0f0; color: #879398; border-color: #dce4e4;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #dce4e4; border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #087f78; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #087f78; border: 2px solid #ffffff;
                width: 14px; height: 14px; margin: -7px 0; border-radius: 8px;
            }
            QSlider::handle:horizontal:hover { background: #055953; }
            QProgressBar {
                min-height: 8px; max-height: 8px;
                border: 0; background: #dce4e4; border-radius: 4px;
            }
            QProgressBar::chunk { background: #087f78; border-radius: 4px; }
            QTextEdit {
                background: #f6f8f8; border: 1px solid #dce4e4;
                border-radius: 4px; padding: 8px; color: #45565b;
            }
            QScrollArea { border: 0; background: #ffffff; }
            QScrollArea > QWidget > QWidget { background: #ffffff; }
            QToolTip {
                background: #242d30; color: #ffffff;
                border: 0; padding: 5px 8px;
            }
        """)
        self.thread = None
        self.worker = None
        self.current_artifact = None
        self.simulated_data: SimulationData | None = None
        self.simulated_plot_view = None
        self.simulated_status = None
        self.simulated_plot_type = None
        self.constant_current_cache = OrderedDict()
        self.constant_current_cache_limit = 16
        self.constant_current_targets = np.array([], dtype=np.float64)
        self.real_data_path: Path | None = None
        self.real_spectra: list[RealSpectrum] = []
        self.real_plot_view = None

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self._build_config_tab()
        self._build_simulated_data_tab(web_engine_view)
        self._build_real_data_tab(web_engine_view)
        self.peak_analysis = PeakAnalysisPanel(self, web_engine_view)
        self.tabs.addTab(self.peak_analysis.peak_page, "Peak Extraction")
        self.tabs.addTab(self.peak_analysis.rf_page, "RF Dependence")
        self._build_math_tab()

    @staticmethod
    def _page(title):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        heading = QLabel(title)
        heading.setProperty("role", "title")
        layout.addWidget(heading)
        return page, layout

    def _build_config_tab(self):
        page, layout = self._page("Simulation setup")

        self.use_lut = QComboBox()
        self.use_lut.setAccessibleName("Transmission evaluation")
        self.use_lut.addItem("Direct", False)
        self.use_lut.addItem("LUT", True)
        self.use_lut.setCurrentIndex(int(self.config.use_lut))
        self.dc_table = QComboBox()
        self.dc_table.setAccessibleName("RF averaging")
        self.dc_table.addItem("Full phase evaluation", False)
        self.dc_table.addItem("Adaptive DC-current table", True)
        self.dc_table.setCurrentIndex(int(self.config.dc_table))
        methods = QGroupBox("Methods")
        methods_form = QFormLayout(methods)
        methods_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        methods_form.addRow("Transmission evaluation", self.use_lut)
        methods_form.addRow("RF averaging", self.dc_table)
        layout.addWidget(methods)

        self.n_E = self._integer_field(self.config.n_E)
        self.n_V = self._integer_field(self.config.n_V)
        self.n_Z = self._integer_field(self.config.n_Z)
        self.n_A = self._integer_field(self.config.n_A)
        self.n_cheb = self._integer_field(self.config.n_cheb)
        self.E_min = self._number_field(self.config.E_min)
        self.E_max = self._number_field(self.config.E_max)
        self.v_min = self._number_field(self.config.v_min)
        self.v_max = self._number_field(self.config.v_max)
        self.z_min = self._number_field(self.config.z_min)
        self.z_max = self._number_field(self.config.z_max)
        self.A_min = self._number_field(self.config.A_min)
        self.A_max = self._number_field(self.config.A_max)
        self.phi_tip = self._number_field(self.config.phi_tip)
        self.phi_samp = self._number_field(self.config.phi_samp)
        self.output = QLineEdit(str(self.config.out))
        self._initial_values = {name: getattr(self, name).value() for name in (
            "n_E", "n_V", "n_Z", "n_A", "n_cheb", "E_min", "E_max", "v_min", "v_max",
            "z_min", "z_max", "A_min", "A_max", "phi_tip", "phi_samp",
        )}
        self._initial_values["out"] = self.output.text()
        self._initial_values.update(use_lut=self.config.use_lut, dc_table=self.config.dc_table)

        groups = QGridLayout()
        groups.setHorizontalSpacing(20)
        groups.setVerticalSpacing(12)
        groups.setColumnStretch(0, 1)
        groups.setColumnStretch(1, 1)
        sections = (
            ("Sampling", 0, 0, 1, (
                ("Energy points", self.n_E),
                ("Voltage points", self.n_V),
                ("Height points", self.n_Z),
                ("RF amplitude points", self.n_A),
                ("RF quadrature points", self.n_cheb),
            )),
            ("Work functions", 1, 0, 1, (
                ("Tip (eV)", self.phi_tip),
                ("Sample (eV)", self.phi_samp),
            )),
            ("Sweep ranges", 0, 1, 2, (
                ("Energy minimum (eV)", self.E_min),
                ("Energy maximum (eV)", self.E_max),
                ("Bias minimum (V)", self.v_min),
                ("Bias maximum (V)", self.v_max),
                ("Tip height minimum (nm)", self.z_min),
                ("Tip height maximum (nm)", self.z_max),
                ("RF amplitude minimum (V)", self.A_min),
                ("RF amplitude maximum (V)", self.A_max),
            )),
        )
        for title, row, column, row_span, fields in sections:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(8)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for label, field in fields:
                form.addRow(label, field)
            groups.addWidget(group, row, column, row_span, 1)
        layout.addLayout(groups)

        self._build_simulation_advanced(layout)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(QLabel("Output"))
        buttons.addWidget(self.output, stretch=1)
        browse = QToolButton()
        browse.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse.setToolTip("Choose output directory")
        browse.setAccessibleName("Choose output directory")
        browse.setFixedSize(36, 36)
        browse.clicked.connect(self._choose_output)
        self.run_button = QPushButton("Run simulation")
        self.run_button.setProperty("role", "primary")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.setMinimumWidth(170)
        self.run_button.clicked.connect(self._run_simulation)
        buttons.addWidget(browse)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)

        log_heading = QLabel("Run log")
        log_heading.setProperty("role", "section")
        layout.addWidget(log_heading)
        self.simulation_progress = QProgressBar()
        self.simulation_progress.setRange(0, 0)
        self.simulation_progress.setAccessibleName("Simulation progress")
        self.simulation_progress.setToolTip("Simulation is running")
        self.simulation_progress.setFormat("Preparing simulation...")
        self.simulation_progress.hide()
        self.statusBar().addPermanentWidget(self.simulation_progress, 1)
        self.statusBar().hide()
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setPlaceholderText("No simulation runs yet.")
        self.status.setMinimumHeight(90)
        layout.addWidget(self.status, stretch=1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "Config + Run")
        for name in self._initial_values:
            field = self.output if name == "out" else getattr(self, name)
            field.ensurePolished()
            field.setMinimumSize(field.minimumSizeHint())

    @staticmethod
    def _integer_field(value):
        field = QSpinBox()
        field.setRange(1, 100000)
        field.setValue(value)
        return field

    def _build_simulation_advanced(self, layout):
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setAccessibleName("Advanced simulation settings")
        self.advanced_toggle.setToolTip("Advanced simulation settings")
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setCheckable(True)
        layout.addWidget(self.advanced_toggle, alignment=Qt.AlignLeft)
        self.advanced_settings = QWidget()
        advanced = QGridLayout(self.advanced_settings)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setHorizontalSpacing(20)
        advanced.setColumnStretch(0, 1)
        advanced.setColumnStretch(1, 1)

        self.upscale = self._integer_field(self.config.upscale)
        for name in ("upscale_z", "upscale_v", "upscale_e"):
            field = self._integer_field(1)
            field.setMinimum(0)
            field.setSpecialValueText("Global")
            field.setValue(getattr(self.config, name) or 0)
            setattr(self, name, field)
        self.force_rebuild_lut = QCheckBox("Force rebuild")
        self.force_rebuild_lut.setChecked(self.config.force_rebuild_lut)
        self.force_rebuild_lut.setToolTip("Rebuild the transmission LUT even if a compatible cache exists")
        self.dc_table_points = self._integer_field(self.config.dc_table_points)
        self.dc_table_points.setMinimum(2)
        self.dc_table_max_points = self._integer_field(self.config.dc_table_max_points)
        self.dc_table_max_points.setMinimum(2)
        self.dc_table_rtol = self._tolerance_field(self.config.dc_table_rtol)
        self.dc_table_rtol.setToolTip("Relative tolerance for sampled DC-current interpolation checks")
        self.dc_table_atol = self._tolerance_field(self.config.dc_table_atol)
        self.dc_table_atol.setToolTip("Absolute tolerance in normalized current units")
        self.threads = self._integer_field(1)
        self.threads.setMinimum(-1)
        self.threads.setSpecialValueText("All CPUs")
        self.threads.setValue(self.config.threads)
        self.threads.setToolTip("All CPUs (-1), or a positive worker count")
        self.amplitude_chunk = self._integer_field(self.config.amplitude_chunk)

        sections = (
            ("lut_options", "LUT", 0, 0, 2, (
                ("Global multiplier", self.upscale),
                ("Height multiplier", self.upscale_z),
                ("Voltage multiplier", self.upscale_v),
                ("Energy multiplier", self.upscale_e),
                ("Cache", self.force_rebuild_lut),
            )),
            ("dc_table_options", "DC-current table", 0, 1, 1, (
                ("Initial table points", self.dc_table_points),
                ("Maximum table points", self.dc_table_max_points),
                ("Relative tolerance", self.dc_table_rtol),
                ("Absolute tolerance", self.dc_table_atol),
            )),
            ("execution_options", "Execution", 1, 1, 1, (
                ("Workers", self.threads),
                ("Amplitude chunk", self.amplitude_chunk),
            )),
        )
        for name, title, row, column, row_span, fields in sections:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(8)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for label, field in fields:
                form.addRow(label, field)
            setattr(self, name, group)
            advanced.addWidget(group, row, column, row_span, 1)
        self._initial_values.update(self._advanced_simulation_values())
        layout.addWidget(self.advanced_settings)
        self.advanced_settings.hide()
        self.advanced_toggle.toggled.connect(self._toggle_simulation_advanced)
        self.use_lut.currentIndexChanged.connect(self._update_simulation_methods)
        self.dc_table.currentIndexChanged.connect(self._update_simulation_methods)
        self._update_simulation_methods()

    @staticmethod
    def _tolerance_field(value):
        field = QLineEdit(str(value))
        validator = QDoubleValidator(field)
        validator.setBottom(0)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        field.setValidator(validator)
        return field

    def _toggle_simulation_advanced(self, expanded):
        self.advanced_settings.setVisible(expanded)
        self.advanced_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _update_simulation_methods(self):
        self.lut_options.setEnabled(self.use_lut.currentData())
        self.dc_table_options.setEnabled(self.dc_table.currentData())

    def _advanced_simulation_values(self):
        return dict(
            upscale=self.upscale.value(),
            upscale_z=self.upscale_z.value() or None,
            upscale_v=self.upscale_v.value() or None,
            upscale_e=self.upscale_e.value() or None,
            force_rebuild_lut=self.force_rebuild_lut.isChecked(),
            dc_table_points=self.dc_table_points.value(),
            dc_table_max_points=self.dc_table_max_points.value(),
            dc_table_rtol=float(self.dc_table_rtol.text()),
            dc_table_atol=float(self.dc_table_atol.text()),
            threads=self.threads.value(),
            amplitude_chunk=self.amplitude_chunk.value(),
        )

    @staticmethod
    def _number_field(value):
        field = QDoubleSpinBox()
        field.setRange(-1e9, 1e9)
        field.setDecimals(4)
        field.setValue(value)
        return field

    def _build_placeholder_tab(self, title, message):
        page, layout = self._page(title)
        status = QLabel(message)
        status.setProperty("role", "muted")
        status.setWordWrap(True)
        layout.addWidget(status)
        select = QPushButton("Select simulation artifact")
        select.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        select.clicked.connect(self._select_artifact)
        layout.addWidget(select, alignment=Qt.AlignLeft)
        layout.addStretch()
        self.tabs.addTab(page, title)

    def _build_math_tab(self):
        page, layout = self._page("Mathematical model")
        topics = QHBoxLayout()
        topics.addWidget(QLabel("Topic"))
        self.math_topics = QComboBox()
        self.math_topics.setAccessibleName("Math reference topic")
        self.math_topics.addItem("Contents", "contents")
        for anchor, title, _ in MATH_SECTIONS:
            self.math_topics.addItem(title, anchor)
        topics.addWidget(self.math_topics, stretch=1)
        layout.addLayout(topics)
        self.math_browser = QTextBrowser()
        self.math_browser.setAccessibleName("Mathematical model reference")
        self.math_browser.setOpenExternalLinks(False)
        self.math_browser.setStyleSheet(
            "QTextBrowser { background: #ffffff; border: 0; padding: 8px; }"
        )
        self.math_browser.setHtml(math_reference_html())
        self.math_topics.currentIndexChanged.connect(self._scroll_math_topic)
        layout.addWidget(self.math_browser, stretch=1)
        self.tabs.addTab(page, "Math Reference")

    def _scroll_math_topic(self, index):
        anchor = self.math_topics.itemData(index)
        if anchor == "contents":
            self.math_browser.verticalScrollBar().setValue(0)
        else:
            self.math_browser.scrollToAnchor(anchor)

    def _build_real_data_tab(self, web_engine_view):
        page, layout = self._page("Real spectra")

        sources = QHBoxLayout()
        select_file = QPushButton("Select real-data CSV")
        select_file.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        select_file.clicked.connect(self._select_real_data_file)
        sources.addWidget(select_file)
        select_directory = QPushButton("Select real-data folder")
        select_directory.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        select_directory.clicked.connect(self._select_real_data_directory)
        sources.addWidget(select_directory)
        sources.addStretch()
        layout.addLayout(sources)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Spectrum"))
        self.real_spectrum_slider = QSlider(Qt.Horizontal)
        self.real_spectrum_slider.setMinimum(0)
        self.real_spectrum_slider.setMaximum(0)
        self.real_spectrum_slider.valueChanged.connect(self._render_real_plot)
        controls.addWidget(self.real_spectrum_slider, stretch=1)
        self.real_spectrum_value = QLabel("No spectrum selected")
        self.real_spectrum_value.setProperty("role", "muted")
        self.real_spectrum_value.setWordWrap(True)
        layout.addLayout(controls)
        layout.addWidget(self.real_spectrum_value)

        self.real_data_status = QLabel("No real-data source selected.")
        self.real_data_status.setProperty("role", "muted")
        self.real_data_status.setWordWrap(True)
        self.real_data_status.setToolTip(
            str(Path(os.environ.get("FER_REAL_DATA_CSV", REAL_DATA_CSV_PATH)))
        )
        layout.addWidget(self.real_data_status)
        if web_engine_view is None:
            self.real_plot_view = QLabel(
                "Qt WebEngine is unavailable. Install the complete PySide6 package "
                "to display real-data plots."
            )
            self.real_plot_view.setWordWrap(True)
        else:
            self.real_plot_view = web_engine_view()
            self.real_host = PlotlyHost(self.real_plot_view, self.plot_document)
        layout.addWidget(self.real_plot_view, stretch=1)
        self.tabs.addTab(page, "Real Data")

        configured_source = Path(os.environ.get("FER_REAL_DATA_CSV", REAL_DATA_CSV_PATH))
        if self.auto_load_real_data and configured_source.exists():
            self._load_real_data(configured_source)

    def _set_real_data_path(self, path):
        self.real_data_path = Path(path)
        self.real_data_status.setText(
            f"Selected real-data source:\n{self.real_data_path}"
        )
        self._load_real_data(self.real_data_path)

    def _load_real_data(self, path):
        try:
            self.real_spectra = load_real_data(path, rf_mapping=REAL_DATA_RF_MAPPING)
        except (FileNotFoundError, ValueError, OSError) as error:
            self.real_spectra = []
            self.real_data_status.setText(f"Could not load real data: {error}")
            if hasattr(self, "peak_analysis"):
                self.peak_analysis.refresh_inputs()
            return
        self.real_data_path = Path(path)
        self.real_spectrum_slider.blockSignals(True)
        self.real_spectrum_slider.setMaximum(len(self.real_spectra) - 1)
        self.real_spectrum_slider.setValue(0)
        self.real_spectrum_slider.blockSignals(False)
        self.real_data_status.setText(
            f"Loaded {len(self.real_spectra)} real spectra from:\n{path}"
        )
        self._render_real_plot()
        if hasattr(self, "peak_analysis"):
            self.peak_analysis.refresh_inputs()

    def _render_real_plot(self, *_):
        if not self.real_spectra:
            return
        import plotly.graph_objects as go

        spectrum = self.real_spectra[self.real_spectrum_slider.value()]
        data = spectrum.data
        rf_label = "unknown" if spectrum.rf_amplitude is None else f"{spectrum.rf_amplitude:.2f} V peak"
        self.real_spectrum_value.setText(f"{spectrum.path.name} | RF {rf_label}")
        if self.real_host is None:
            return
        figure = go.Figure(go.Scatter(
            x=data["bias (mV)"],
            y=data["lockin-x (mV)"],
            mode="lines",
            line={"color": "#d62728"},
        ))
        figure.update_layout(
            title=f"Real FER spectrum: {spectrum.path.name}",
            xaxis_title="Bias (mV)",
            yaxis_title="lock-in X (mV)",
        )
        self._style_plot(figure)
        self.real_host.update(figure)

    def _select_real_data_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select real-data CSV",
            os.environ.get("FER_REAL_DATA_CSV", REAL_DATA_CSV_PATH),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self._set_real_data_path(path)

    def _select_real_data_directory(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing real-data CSV files",
            os.environ.get("FER_REAL_DATA_CSV", REAL_DATA_CSV_PATH),
        )
        if path:
            self._set_real_data_path(path)

    def _build_simulated_data_tab(self, web_engine_view):
        page, layout = self._page("Simulated data")
        controls = QHBoxLayout()

        select = QPushButton("Select HDF5")
        select.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        select.clicked.connect(self._select_simulation_for_plot)
        controls.addWidget(select)

        self.simulated_view_type = QComboBox()
        self.simulated_view_type.addItems([
            "Current surface",
            "Current heatmap",
            "Tunneling probability heatmap",
            "Tunneling probability surface",
            "Fixed-height dI/dV heatmap",
            "Constant-current dI/dV heatmap",
            "Constant-current dI/dV trace",
        ])
        self.simulated_view_type.currentTextChanged.connect(self._render_simulated_plot)
        self.simulated_view_type.setMinimumWidth(285)
        controls.addWidget(self.simulated_view_type)
        controls.addStretch()
        layout.addLayout(controls)

        axis_controls = QHBoxLayout()
        self.simulated_x_axis = QComboBox()
        self.simulated_y_axis = QComboBox()
        for selector in (self.simulated_x_axis, self.simulated_y_axis):
            selector.addItems(["Bias", "Tip height"])
        self.simulated_y_axis.setCurrentText("Tip height")
        self.simulated_x_axis.currentTextChanged.connect(
            lambda _: self._update_simulated_axes(self.simulated_x_axis, self.simulated_y_axis)
        )
        self.simulated_y_axis.currentTextChanged.connect(
            lambda _: self._update_simulated_axes(self.simulated_y_axis, self.simulated_x_axis)
        )
        axis_controls.addWidget(QLabel("X axis"))
        axis_controls.addWidget(self.simulated_x_axis)
        axis_controls.addWidget(QLabel("Y axis"))
        axis_controls.addWidget(self.simulated_y_axis)
        axis_controls.addWidget(QLabel("Color scale"))
        self.simulated_color_scale = QComboBox()
        self.simulated_color_scale.addItems(["Linear", "Log10"])
        self.simulated_color_scale.setCurrentText("Log10")
        self.simulated_color_scale.currentTextChanged.connect(self._render_simulated_plot)
        axis_controls.addWidget(self.simulated_color_scale)
        axis_controls.addStretch()
        layout.addLayout(axis_controls)

        rf_controls = QHBoxLayout()
        rf_label = QLabel("RF amplitude")
        rf_label.setMinimumWidth(120)
        rf_controls.addWidget(rf_label)
        self.simulated_rf_slider = QSlider()
        self.simulated_rf_slider.setOrientation(Qt.Horizontal)
        self.simulated_rf_slider.setMinimum(0)
        self.simulated_rf_slider.setMaximum(0)
        self.simulated_rf_slider.setTracking(False)
        self.simulated_rf_slider.valueChanged.connect(self._render_simulated_plot)
        rf_controls.addWidget(self.simulated_rf_slider, stretch=1)
        self.simulated_rf_value = QLabel("n/a")
        self.simulated_rf_value.setProperty("role", "value")
        self.simulated_rf_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.simulated_rf_value.setFixedWidth(110)
        rf_controls.addWidget(self.simulated_rf_value)
        layout.addLayout(rf_controls)

        self.simulated_energy_controls = QWidget()
        energy_controls = QHBoxLayout(self.simulated_energy_controls)
        energy_controls.setContentsMargins(0, 0, 0, 0)
        energy_label = QLabel("Electron energy")
        energy_label.setMinimumWidth(120)
        energy_controls.addWidget(energy_label)
        self.simulated_energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.simulated_energy_slider.setMaximum(0)
        self.simulated_energy_slider.setTracking(False)
        self.simulated_energy_slider.valueChanged.connect(self._render_simulated_plot)
        energy_controls.addWidget(self.simulated_energy_slider, stretch=1)
        self.simulated_energy_value = QLabel("n/a")
        self.simulated_energy_value.setProperty("role", "value")
        self.simulated_energy_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.simulated_energy_value.setFixedWidth(110)
        energy_controls.addWidget(self.simulated_energy_value)
        self.simulated_energy_controls.hide()
        layout.addWidget(self.simulated_energy_controls)
        self.transmission_source = None
        self.transmission_slice_cache = None

        self.simulated_setpoint_controls = QWidget()
        setpoint_controls = QHBoxLayout(self.simulated_setpoint_controls)
        setpoint_controls.setContentsMargins(0, 0, 0, 0)
        setpoint_label = QLabel("Current setpoint")
        setpoint_label.setMinimumWidth(120)
        setpoint_controls.addWidget(setpoint_label)
        self.simulated_current_slider = QSlider()
        self.simulated_current_slider.setOrientation(Qt.Horizontal)
        self.simulated_current_slider.setMinimum(0)
        self.simulated_current_slider.setMaximum(0)
        self.simulated_current_slider.valueChanged.connect(
            self._render_simulated_plot
        )
        setpoint_controls.addWidget(self.simulated_current_slider, stretch=1)
        self.simulated_current_value = QLabel("n/a")
        self.simulated_current_value.setProperty("role", "value")
        self.simulated_current_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.simulated_current_value.setFixedWidth(110)
        setpoint_controls.addWidget(self.simulated_current_value)
        layout.addWidget(self.simulated_setpoint_controls)

        self.constant_current_status = QLabel("")
        self.constant_current_status.setProperty("role", "muted")
        self.constant_current_status.setWordWrap(True)
        layout.addWidget(self.constant_current_status)

        self.simulated_status = QLabel("Select a simulation HDF5 file to begin.")
        self.simulated_status.setProperty("role", "muted")
        self.simulated_status.setWordWrap(True)
        layout.addWidget(self.simulated_status)

        if web_engine_view is None:
            self.simulated_plot_view = QLabel(
                "Qt WebEngine is unavailable. Install the complete PySide6 package "
                "to display interactive Plotly plots."
            )
            self.simulated_plot_view.setWordWrap(True)
        else:
            self.simulated_plot_view = web_engine_view()
            self.simulated_host = PlotlyHost(self.simulated_plot_view, self.plot_document)
        layout.addWidget(self.simulated_plot_view, stretch=1)
        self.tabs.addTab(page, "Simulated Data")

    def _update_simulated_axes(self, changed, other):
        other.blockSignals(True)
        other.setCurrentText("Tip height" if changed.currentText() == "Bias" else "Bias")
        other.blockSignals(False)
        self._render_simulated_plot()

    def _select_simulation_for_plot(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select simulation artifact", "", "HDF5 files (*.h5)"
        )
        if not path:
            return
        self.load_artifact(path)

    def load_artifact(self, path):
        self.constant_current_cache.clear()
        self.current_array = None
        self.transmission_source = None
        self.transmission_slice_cache = None
        self.constant_current_status.clear()
        self.simulated_energy_value.setText("n/a")
        try:
            self.simulated_data = load_simulation(path)
            self.current_array = build_current_array(self.simulated_data.current, self.simulated_data.voltage)
        except (FileNotFoundError, ValueError, OSError) as error:
            self.simulated_data = None
            self.simulated_status.setText(f"Could not load simulation: {error}")
            self.constant_current_targets = np.array([], dtype=np.float64)
            self.peak_analysis.refresh_inputs()
            return

        self.current_artifact = Path(path)
        self.constant_current_cache.clear()
        positive_currents = self.simulated_data.current[
            np.isfinite(self.simulated_data.current)
            & (self.simulated_data.current > 0)
        ]
        if positive_currents.size:
            lower = np.log10(positive_currents.min())
            upper = np.log10(positive_currents.max())
            self.constant_current_targets = np.logspace(lower, upper, 501)
            self.simulated_current_slider.blockSignals(True)
            self.simulated_current_slider.setMaximum(
                self.constant_current_targets.size - 1
            )
            self.simulated_current_slider.setValue(
                self.constant_current_targets.size // 2
            )
            self.simulated_current_slider.blockSignals(False)
        else:
            self.constant_current_targets = np.array([], dtype=np.float64)
            self.simulated_current_slider.setMaximum(0)
        self.simulated_rf_slider.blockSignals(True)
        self.simulated_rf_slider.setMaximum(
            self.simulated_data.rf_amplitude.size - 1
        )
        self.simulated_rf_slider.setValue(0)
        self.simulated_rf_slider.blockSignals(False)
        self.simulated_energy_slider.blockSignals(True)
        energy = self.simulated_data.energy
        self.simulated_energy_slider.setMaximum(energy.size - 1 if energy is not None else 0)
        self.simulated_energy_slider.setValue(0)
        self.simulated_energy_slider.blockSignals(False)
        self.simulated_status.setText(
            f"Loaded {self.simulated_data.path.name}: "
            f"{self.simulated_data.current.shape}"
        )
        self._render_simulated_plot()
        self.peak_analysis.refresh_inputs()

    def _render_simulated_plot(self, *_):
        plot_type = self.simulated_view_type.currentText()
        probability_view = plot_type.startswith("Tunneling probability")
        constant_current_view = plot_type.startswith("Constant-current")
        grid_heatmap = plot_type in (
            "Current heatmap", "Tunneling probability heatmap", "Fixed-height dI/dV heatmap",
        )
        for selector in (self.simulated_x_axis, self.simulated_y_axis):
            selector.setEnabled(grid_heatmap)
        self.simulated_color_scale.setEnabled(not constant_current_view)
        self.simulated_energy_controls.setVisible(probability_view)
        self.simulated_energy_slider.setEnabled(
            probability_view and self.simulated_data is not None and self.simulated_data.energy is not None
        )
        self.simulated_current_slider.setEnabled(constant_current_view)
        self.simulated_setpoint_controls.setVisible(constant_current_view)
        if not constant_current_view:
            self.constant_current_status.clear()
        if self.simulated_data is None:
            return

        import plotly.graph_objects as go

        data = self.simulated_data
        rf_index = self.simulated_rf_slider.value()
        rf_value = data.rf_amplitude[rf_index]
        self.simulated_rf_value.setText(f"{rf_value:.4g} V")
        if self.simulated_host is None:
            return
        if plot_type == "Constant-current dI/dV heatmap":
            plot_values, title = self._constant_current_heatmap_data()
            if plot_values is None:
                return
        elif plot_type == "Constant-current dI/dV trace":
            heatmap_values, heatmap_title = self._constant_current_heatmap_data()
            if heatmap_values is None:
                return
            plot_values = heatmap_values[rf_index]
            title = f"dI/dV vs bias at constant current {self.simulated_current_value.text()}"
        else:
            if probability_view:
                try:
                    plot_values, energy, source_label = self._simulated_transmission_data(rf_index)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.constant_current_status.setText(f"Tunneling probability unavailable: {error}")
                    figure = go.Figure()
                    figure.update_layout(title="Tunneling probability unavailable")
                    self._style_plot(figure)
                    self.simulated_host.update(figure)
                    return
                self.simulated_energy_value.setText(f"{energy:.4g} eV")
                self.constant_current_status.setText(source_label)
                quantity = "T"
                title = f"RF-averaged T: E={energy:.4g} eV, RF={rf_value:.4g} V"
            elif plot_type == "Fixed-height dI/dV heatmap":
                if self.current_array is None:
                    return
                plot_values = self.current_array[:, :, rf_index, 1]
                quantity = "dI/dV"
                title = f"Fixed-height dI/dV at RF amplitude {rf_value:.4g} V"
            else:
                plot_values = data.current[:, :, rf_index]
                quantity = "I"
                title = f"Current at RF amplitude {rf_value:.4g} V"
            color_label = "T (probability)" if probability_view else quantity
            if self.simulated_color_scale.currentText() == "Log10":
                with np.errstate(divide="ignore", invalid="ignore"):
                    plot_values = np.log10(np.abs(plot_values))
                color_label = f"log10(|{quantity}|)"
            plot_values = np.array(plot_values, dtype=float, copy=True)
            plot_values[~np.isfinite(plot_values)] = np.nan

        if grid_heatmap:
            swapped = self.simulated_x_axis.currentText() == "Tip height"
            figure = go.Figure(go.Heatmap(
                z=plot_values.T if swapped else plot_values,
                x=data.z if swapped else data.voltage,
                y=data.voltage if swapped else data.z,
                colorscale="RdBu_r" if quantity == "dI/dV" else "Viridis",
                colorbar={"title": color_label},
                hoverongaps=False,
            ))
            figure.update_layout(
                xaxis_title="Tip height (nm)" if swapped else "Bias voltage (V)",
                yaxis_title="Bias voltage (V)" if swapped else "Tip height (nm)",
            )
        elif plot_type in ("Current surface", "Tunneling probability surface"):
            figure = go.Figure(go.Surface(
                z=plot_values,
                x=data.voltage,
                y=data.z,
                colorscale="Viridis",
                colorbar={"title": color_label},
            ))
            figure.update_layout(
                scene={
                    "xaxis_title": "Bias voltage (V)",
                    "yaxis_title": "Tip height (nm)",
                    "zaxis_title": color_label,
                }
            )
        elif plot_type == "Constant-current dI/dV heatmap":
            figure = go.Figure(go.Heatmap(
                z=plot_values,
                x=data.voltage,
                y=data.rf_amplitude,
                colorscale="RdBu_r",
                colorbar={"title": "dI/dV"},
            ))
            figure.update_layout(
                xaxis_title="DC bias (V)",
                yaxis_title="RF amplitude (V)",
            )
        else:
            figure = go.Figure(go.Scatter(
                x=data.voltage,
                y=plot_values,
                mode="lines+markers",
                line={"color": "#1f77b4"},
                marker={"size": 4},
            ))
            figure.update_layout(
                xaxis_title="DC bias (V)",
                yaxis_title="dI/dV",
            )
        figure.update_layout(title=title)
        if grid_heatmap:
            figure.update_layout(margin={"l": 56, "r": 30, "t": 40, "b": 40})
        self._style_plot(figure)
        self.simulated_host.update(figure)
        self.simulated_plot_type = plot_type

    def _simulated_transmission_data(self, rf_index):
        data = self.simulated_data
        if data is None:
            raise ValueError("Load a simulation artifact first")
        if data.energy is None:
            raise ValueError("The artifact has no electron-energy dataset (E)")
        energy_index = self.simulated_energy_slider.value()
        energy = float(data.energy[energy_index])
        if self.transmission_source is None:
            self.transmission_source = simulation_transmission_source(data)
        source, source_label = self.transmission_source
        key = (rf_index, energy_index)
        if self.transmission_slice_cache is None or self.transmission_slice_cache[0] != key:
            nodes = chebyshev_phases(data.metadata.get("n_cheb", 32))
            _, _, values = requested_transmission_grid(
                source, (data.z, data.voltage, data.rf_amplitude, data.energy),
                "V", "Z", {"A": data.rf_amplitude[rf_index], "E": energy}, nodes,
            )
            self.transmission_slice_cache = (key, values)
        return self.transmission_slice_cache[1], energy, source_label

    @staticmethod
    def _style_plot(figure):
        figure.update_layout(
            template="plotly_white",
            font={"family": "Segoe UI, sans-serif", "size": 13, "color": "#39484d"},
            title={"font": {"size": 16}, "x": 0.02, "xanchor": "left"},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f6f8f8",
            margin={"l": 56, "r": 30, "t": 54, "b": 50},
            xaxis={"automargin": True, "gridcolor": "#e3e9e9", "zerolinecolor": "#cdd8da"},
            yaxis={"automargin": True, "gridcolor": "#e3e9e9", "zerolinecolor": "#cdd8da"},
        )

    def _constant_current_heatmap_data(self):
        if self.constant_current_targets.size == 0:
            return None, ""
        index = self.simulated_current_slider.value()
        target = float(self.constant_current_targets[index])
        self.simulated_current_value.setText(f"{target:.3e}")
        if index not in self.constant_current_cache:
            result = constant_current_slice(
                target, self.simulated_data.z, self.current_array,
            )
            self.constant_current_cache[index] = (
                result[..., 0].T,
                result[..., 1].T,
            )
            if len(self.constant_current_cache) > self.constant_current_cache_limit:
                self.constant_current_cache.popitem(last=False)
        self.constant_current_cache.move_to_end(index)
        height_values, derivative_values = self.constant_current_cache[index]
        valid = np.isfinite(derivative_values)
        finite_heights = height_values[np.isfinite(height_values)]
        height_status = (
            f" Z range: {finite_heights.min():.3f}-{finite_heights.max():.3f} nm."
            if finite_heights.size else ""
        )
        self.constant_current_status.setText(
            f"Valid constant-current cells: {valid.sum()}/{valid.size}."
            + height_status
        )
        return (
            derivative_values,
            f"dI/dV at constant current {target:.3e}",
        )

    def _choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if directory:
            self.output.setText(directory)

    def _select_artifact(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select simulation artifact", "", "HDF5 files (*.h5)"
        )
        if path:
            self.current_artifact = Path(path)
            self.status.append(f"Selected artifact: {path}")

    def _run_simulation(self):
        if self.thread is not None:
            self._cancel_simulation()
            return
        if self._close_requested:
            return
        values = dict(
            n_E=self.n_E.value(),
            n_V=self.n_V.value(),
            n_Z=self.n_Z.value(),
            n_A=self.n_A.value(),
            n_cheb=self.n_cheb.value(),
            E_min=self.E_min.value(),
            E_max=self.E_max.value(),
            v_min=self.v_min.value(),
            v_max=self.v_max.value(),
            z_min=self.z_min.value(),
            z_max=self.z_max.value(),
            A_min=self.A_min.value(),
            A_max=self.A_max.value(),
            phi_tip=self.phi_tip.value(),
            phi_samp=self.phi_samp.value(),
            use_lut=self.use_lut.currentData(),
            dc_table=self.dc_table.currentData(),
            out=self.output.text(),
        )
        try:
            values.update(self._advanced_simulation_values())
            edits = {name: value for name, value in values.items() if value != self._initial_values[name]}
            config = replace(self.config, **edits)
            config.validate()
        except ValueError as error:
            self.status.append(f"Invalid configuration: {error}")
            return
        self.run_button.setEnabled(False)
        self.run_button.setText("Cancel simulation")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.run_button.setEnabled(True)
        self.statusBar().show()
        self.simulation_progress.setRange(0, 0)
        self.simulation_progress.setFormat("Preparing simulation...")
        self.simulation_progress.show()
        self.status.append("Starting simulation...")
        transmission_method = "LUT" if config.use_lut else "Direct"
        rf_method = "Adaptive DC-current table" if config.dc_table else "Full phase evaluation"
        self.status.append(f"Transmission: {transmission_method}. RF averaging: {rf_method}.")
        if config.use_lut:
            self.status.append("LUT cache: compatibility check and automatic build/reuse.")
        self.thread = QThread(self)
        self.worker = SimulationWorker(config, Event())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._simulation_completed)
        self.worker.failed.connect(self._simulation_failed)
        self.worker.cancelled.connect(self._simulation_cancelled)
        self.worker.progress.connect(self._simulation_progress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _cancel_simulation(self):
        if self.worker is None or self.worker.cancel_event.is_set():
            return
        self.worker.cancel_event.set()
        self.run_button.setText("Stopping...")
        self.run_button.setEnabled(False)
        self.status.append("Stopping simulation after the active calculation checkpoint...")

    def _simulation_progress(self, stage, completed, total):
        self.simulation_progress.setRange(0, total)
        self.simulation_progress.setValue(completed)
        self.simulation_progress.setFormat(f"{stage}: %v/%m")

    def _simulation_completed(self, artifact):
        self.current_artifact = Path(artifact)
        self.status.append(f"Simulation complete: {artifact}")
        if not self._close_requested:
            self.load_artifact(artifact)
            if self.simulated_data is not None:
                metadata = self.simulated_data.metadata
                cache_status = metadata.get("lut_cache_status")
                if cache_status in ("built", "reused"):
                    self.status.append(f"LUT cache: {cache_status} ({metadata.get('lut_file', '')}).")

    def _simulation_failed(self, message):
        self.status.append(f"Simulation failed: {message}")

    def _simulation_cancelled(self):
        self.status.append("Simulation cancelled.")

    def _thread_finished(self):
        self.simulation_progress.hide()
        self.statusBar().hide()
        self.run_button.setText("Run simulation")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.setEnabled(not self._close_requested)
        self.worker = None
        self.thread = None
        if self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self._close_requested = True
            self.run_button.setEnabled(False)
            self.status.append("Waiting for the simulation to finish before closing.")
            event.ignore()
            return
        self.peak_analysis.close()
        for host in (self.simulated_host, self.real_host):
            if host is not None:
                host.close()
        if self.plot_document is not None:
            self.plot_document.close()
            self.plot_document = None
        super().closeEvent(event)

