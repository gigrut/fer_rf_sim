"""PySide6 desktop shell for the FER simulation and analysis services."""

from __future__ import annotations

import sys
import tempfile
import json
import os
from pathlib import Path

import numpy as np
from FER_sim_post_processing import build_current_array, constant_current_slice
from fer_sim_config import REAL_DATA_CSV_PATH
from real_data_service import RealSpectrum, load_real_data
from simulation_repository import SimulationData, load_simulation
from simulation_service import SimulationConfig, SimulationRunner


TAB_NAMES = (
    "Config + Run",
    "Simulated Data",
    "Real Data",
    "Peak Extraction",
    "RF Dependence",
)


def run_gui() -> int:
    """Launch the desktop application."""
    try:
        from PySide6.QtCore import QObject, QThread, QUrl, Qt, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPushButton,
            QSlider,
            QSpinBox,
            QDoubleSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except ImportError:
            QWebEngineView = None
    except ImportError as error:
        raise RuntimeError(
            "PySide6 is required to run the GUI. Install it with:\n"
            f"{sys.executable} -m pip install PySide6"
        ) from error

    class SimulationWorker(QObject):
        completed = Signal(str)
        failed = Signal(str)

        def __init__(self, config: SimulationConfig):
            super().__init__()
            self.config = config

        @Slot()
        def run(self):
            try:
                artifact = SimulationRunner().run(self.config)
            except Exception as error:  # pragma: no cover - Qt callback boundary
                self.failed.emit(str(error))
            else:
                self.completed.emit(str(artifact))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("FER RF STM Simulation")
            self.resize(1100, 700)
            self.thread = None
            self.worker = None
            self.current_artifact = None
            self.simulated_data: SimulationData | None = None
            self.simulated_plot_view = None
            self.simulated_status = None
            self.simulated_plot_file: Path | None = None
            self.simulated_plot_initialized = False
            self.simulated_plot_type = None
            self.constant_current_cache = {}
            self.constant_current_targets = np.array([], dtype=np.float64)
            self.real_data_path: Path | None = None
            self.real_spectra: list[RealSpectrum] = []
            self.real_plot_view = None
            self.real_plot_file: Path | None = None
            self.real_plot_initialized = False

            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)
            self._build_config_tab()
            self._build_simulated_data_tab(QWebEngineView)
            self._build_real_data_tab(QWebEngineView)
            self._build_placeholder_tab("Peak Extraction", "Configure smoothing and peak detection against real or simulated spectra.")
            self._build_placeholder_tab("RF Dependence", "Compare peak position and width against RF amplitude.")

        def _build_config_tab(self):
            page = QWidget()
            layout = QVBoxLayout(page)
            form = QFormLayout()

            self.n_E = self._integer_field(100)
            self.n_V = self._integer_field(100)
            self.n_Z = self._integer_field(100)
            self.n_A = self._integer_field(30)
            self.n_cheb = self._integer_field(32)
            self.E_min = self._number_field(0.01)
            self.E_max = self._number_field(5.5)
            self.v_min = self._number_field(0.0)
            self.v_max = self._number_field(10.0)
            self.z_min = self._number_field(0.3)
            self.z_max = self._number_field(5.0)
            self.A_min = self._number_field(0.0)
            self.A_max = self._number_field(5.0)
            self.phi_tip = self._number_field(4.0)
            self.phi_samp = self._number_field(5.0)
            self.output = QLineEdit("fer_output")

            form.addRow("Energy points", self.n_E)
            form.addRow("Voltage points", self.n_V)
            form.addRow("Height points", self.n_Z)
            form.addRow("RF amplitude points", self.n_A)
            form.addRow("RF quadrature points", self.n_cheb)
            form.addRow("Energy minimum (eV)", self.E_min)
            form.addRow("Energy maximum (eV)", self.E_max)
            form.addRow("Bias minimum (V)", self.v_min)
            form.addRow("Bias maximum (V)", self.v_max)
            form.addRow("Tip height minimum (nm)", self.z_min)
            form.addRow("Tip height maximum (nm)", self.z_max)
            form.addRow("RF amplitude minimum (V)", self.A_min)
            form.addRow("RF amplitude maximum (V)", self.A_max)
            form.addRow("Tip work function", self.phi_tip)
            form.addRow("Sample work function", self.phi_samp)
            form.addRow("Output directory", self.output)
            layout.addLayout(form)

            buttons = QHBoxLayout()
            browse = QPushButton("Choose output")
            browse.clicked.connect(self._choose_output)
            self.run_button = QPushButton("Run simulation")
            self.run_button.clicked.connect(self._run_simulation)
            buttons.addWidget(browse)
            buttons.addWidget(self.run_button)
            layout.addLayout(buttons)

            self.status = QTextEdit()
            self.status.setReadOnly(True)
            layout.addWidget(self.status)
            self.tabs.addTab(page, "Config + Run")

        @staticmethod
        def _integer_field(value):
            field = QSpinBox()
            field.setRange(1, 100000)
            field.setValue(value)
            return field

        @staticmethod
        def _number_field(value):
            field = QDoubleSpinBox()
            field.setRange(0.0, 100.0)
            field.setDecimals(4)
            field.setValue(value)
            return field

        def _build_placeholder_tab(self, title, message):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(message))
            select = QPushButton("Select simulation artifact")
            select.clicked.connect(self._select_artifact)
            layout.addWidget(select)
            self.tabs.addTab(page, title)

        def _build_real_data_tab(self, web_engine_view):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(
                "Real-data files are not included in this repository. Supported sources:"
            ))
            layout.addWidget(QLabel(
                f"Configured source: {REAL_DATA_CSV_PATH}\n"
                "Environment override: FER_REAL_DATA_CSV\n"
                "Use the selectors below to choose a different CSV or folder."
            ))

            select_file = QPushButton("Select real-data CSV")
            select_file.clicked.connect(self._select_real_data_file)
            layout.addWidget(select_file)
            select_directory = QPushButton("Select real-data folder")
            select_directory.clicked.connect(self._select_real_data_directory)
            layout.addWidget(select_directory)

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Spectrum"))
            self.real_spectrum_slider = QSlider(Qt.Horizontal)
            self.real_spectrum_slider.setMinimum(0)
            self.real_spectrum_slider.setMaximum(0)
            self.real_spectrum_slider.valueChanged.connect(self._render_real_plot)
            controls.addWidget(self.real_spectrum_slider, stretch=1)
            self.real_spectrum_value = QLabel("n/a")
            controls.addWidget(self.real_spectrum_value)
            layout.addLayout(controls)

            self.real_data_status = QLabel("No real-data source selected.")
            self.real_data_status.setWordWrap(True)
            layout.addWidget(self.real_data_status)
            if web_engine_view is None:
                self.real_plot_view = QLabel(
                    "Qt WebEngine is unavailable. Install the complete PySide6 package "
                    "to display real-data plots."
                )
                self.real_plot_view.setWordWrap(True)
            else:
                self.real_plot_view = web_engine_view()
            layout.addWidget(self.real_plot_view, stretch=1)
            self.tabs.addTab(page, "Real Data")

            configured_source = Path(os.environ.get("FER_REAL_DATA_CSV", REAL_DATA_CSV_PATH))
            if configured_source.exists():
                self._load_real_data(configured_source)

        def _set_real_data_path(self, path):
            self.real_data_path = Path(path)
            self.real_data_status.setText(
                f"Selected real-data source:\n{self.real_data_path}"
            )
            self._load_real_data(self.real_data_path)

        def _load_real_data(self, path):
            try:
                self.real_spectra = load_real_data(path)
            except (FileNotFoundError, ValueError, OSError) as error:
                self.real_spectra = []
                self.real_data_status.setText(f"Could not load real data: {error}")
                return
            self.real_data_path = Path(path)
            self.real_plot_initialized = False
            self.real_spectrum_slider.blockSignals(True)
            self.real_spectrum_slider.setMaximum(len(self.real_spectra) - 1)
            self.real_spectrum_slider.setValue(0)
            self.real_spectrum_slider.blockSignals(False)
            self.real_data_status.setText(
                f"Loaded {len(self.real_spectra)} real spectra from:\n{path}"
            )
            self._render_real_plot()

        def _render_real_plot(self, *_):
            if not self.real_spectra or not hasattr(self.real_plot_view, "load"):
                return
            import plotly.graph_objects as go
            from plotly.io import to_html

            spectrum = self.real_spectra[self.real_spectrum_slider.value()]
            data = spectrum.data
            self.real_spectrum_value.setText(
                f"{spectrum.path.name} | RF {spectrum.rf_amplitude:.2f} V"
            )
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
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )
            html = to_html(figure, full_html=True, include_plotlyjs=True)
            if self.real_plot_file is not None:
                self.real_plot_file.unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", prefix="fer_real_plot_",
                encoding="utf-8", delete=False,
            ) as plot_file:
                plot_file.write(html)
                self.real_plot_file = Path(plot_file.name)
            self.real_plot_view.load(QUrl.fromLocalFile(str(self.real_plot_file)))

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
            page = QWidget()
            layout = QVBoxLayout(page)
            controls = QHBoxLayout()

            select = QPushButton("Select HDF5")
            select.clicked.connect(self._select_simulation_for_plot)
            controls.addWidget(select)

            self.simulated_view_type = QComboBox()
            self.simulated_view_type.addItems([
                "Current surface",
                "Current heatmap",
                "Constant-current dI/dV heatmap",
                "Constant-current dI/dV trace",
            ])
            self.simulated_view_type.currentTextChanged.connect(self._render_simulated_plot)
            controls.addWidget(self.simulated_view_type)

            controls.addWidget(QLabel("RF amplitude"))
            self.simulated_rf_slider = QSlider()
            self.simulated_rf_slider.setOrientation(Qt.Horizontal)
            self.simulated_rf_slider.setMinimum(0)
            self.simulated_rf_slider.setMaximum(0)
            self.simulated_rf_slider.valueChanged.connect(self._render_simulated_plot)
            controls.addWidget(self.simulated_rf_slider, stretch=1)
            self.simulated_rf_value = QLabel("n/a")
            controls.addWidget(self.simulated_rf_value)
            layout.addLayout(controls)

            setpoint_controls = QHBoxLayout()
            setpoint_controls.addWidget(QLabel("Current setpoint"))
            self.simulated_current_slider = QSlider()
            self.simulated_current_slider.setOrientation(Qt.Horizontal)
            self.simulated_current_slider.setMinimum(0)
            self.simulated_current_slider.setMaximum(0)
            self.simulated_current_slider.valueChanged.connect(
                self._render_simulated_plot
            )
            setpoint_controls.addWidget(self.simulated_current_slider, stretch=1)
            self.simulated_current_value = QLabel("n/a")
            setpoint_controls.addWidget(self.simulated_current_value)
            layout.addLayout(setpoint_controls)

            self.constant_current_status = QLabel("")
            self.constant_current_status.setWordWrap(True)
            layout.addWidget(self.constant_current_status)

            self.simulated_status = QLabel("Select a simulation HDF5 file to begin.")
            layout.addWidget(self.simulated_status)

            if web_engine_view is None:
                self.simulated_plot_view = QLabel(
                    "Qt WebEngine is unavailable. Install the complete PySide6 package "
                    "to display interactive Plotly plots."
                )
                self.simulated_plot_view.setWordWrap(True)
            else:
                self.simulated_plot_view = web_engine_view()
                self.simulated_plot_view.loadFinished.connect(
                    self._simulated_plot_loaded
                )
            layout.addWidget(self.simulated_plot_view, stretch=1)
            self.tabs.addTab(page, "Simulated Data")

        def _select_simulation_for_plot(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select simulation artifact", "", "HDF5 files (*.h5)"
            )
            if not path:
                return
            try:
                self.simulated_data = load_simulation(path)
            except (FileNotFoundError, ValueError, OSError) as error:
                self.simulated_data = None
                self.simulated_status.setText(f"Could not load simulation: {error}")
                return

            self.current_artifact = Path(path)
            self.simulated_plot_initialized = False
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
            self.simulated_status.setText(
                f"Loaded {self.simulated_data.path.name}: "
                f"{self.simulated_data.current.shape}"
            )
            self._render_simulated_plot()

        def _render_simulated_plot(self, *_):
            if self.simulated_data is None:
                return
            if not hasattr(self.simulated_plot_view, "setHtml"):
                return

            import plotly.graph_objects as go
            from plotly.io import to_html

            data = self.simulated_data
            rf_index = self.simulated_rf_slider.value()
            rf_value = data.rf_amplitude[rf_index]
            self.simulated_rf_value.setText(f"{rf_value:.4g} V")
            plot_type = self.simulated_view_type.currentText()
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
                with np.errstate(divide="ignore", invalid="ignore"):
                    plot_values = np.log10(np.abs(data.current[:, :, rf_index]))
                plot_values[~np.isfinite(plot_values)] = np.nan
                title = f"log10(|Current|) at RF amplitude {rf_value:.4g} V"
            if self.simulated_plot_initialized and plot_type == self.simulated_plot_type:
                self._update_simulated_trace(plot_values, title, plot_type)
                return

            if plot_type == "Current heatmap":
                figure = go.Figure(go.Heatmap(
                    z=plot_values,
                    x=data.voltage,
                    y=data.z,
                    colorscale="Viridis",
                    colorbar={"title": "log10(|I|)"},
                ))
            elif plot_type == "Current surface":
                figure = go.Figure(go.Surface(
                    z=plot_values,
                    x=data.voltage,
                    y=data.z,
                    colorscale="Viridis",
                    colorbar={"title": "log10(|I|)"},
                ))
                figure.update_layout(
                    scene={
                        "xaxis_title": "Bias voltage (V)",
                        "yaxis_title": "Tip height (nm)",
                        "zaxis_title": "log10(|Current|)",
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
            figure.update_layout(title=title, margin={"l": 0, "r": 0, "t": 40, "b": 0})
            html = to_html(
                figure,
                full_html=True,
                include_plotlyjs=True,
                div_id="fer-sim-plot",
            )
            if self.simulated_plot_file is not None:
                self.simulated_plot_file.unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", prefix="fer_sim_plot_",
                encoding="utf-8", delete=False,
            ) as plot_file:
                plot_file.write(html)
                self.simulated_plot_file = Path(plot_file.name)
            self.simulated_plot_view.load(
                QUrl.fromLocalFile(str(self.simulated_plot_file))
            )
            self.simulated_plot_initialized = False
            self.simulated_plot_type = plot_type

        def _constant_current_heatmap_data(self):
            if self.constant_current_targets.size == 0:
                return None, ""
            index = self.simulated_current_slider.value()
            target = float(self.constant_current_targets[index])
            self.simulated_current_value.setText(f"{target:.3e}")
            if index not in self.constant_current_cache:
                current_array = build_current_array(
                    self.simulated_data.current,
                    self.simulated_data.voltage,
                )
                result = constant_current_slice(
                    target, self.simulated_data.z, current_array,
                )
                self.constant_current_cache[index] = (
                    result[..., 0].T,
                    result[..., 1].T,
                )
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

        def _simulated_plot_loaded(self, success):
            self.simulated_plot_initialized = bool(success)

        def _update_simulated_trace(self, current_slice, title, plot_type):
            """Update the existing Plotly trace without rebuilding the page."""
            if plot_type == "Constant-current dI/dV trace":
                field = "y"
                values = [
                    None if not np.isfinite(value) else float(value)
                    for value in current_slice
                ]
                payload = json.dumps(
                    [values],
                    separators=(",", ":"),
                )
            else:
                field = "z"
                values = [
                    [None if not np.isfinite(value) else float(value) for value in row]
                    for row in current_slice
                ]
                payload = json.dumps([values], separators=(",", ":"))
            title_json = json.dumps(title)
            script = (
                "Plotly.restyle('fer-sim-plot', {"
                + field
                + ": "
                + payload
                + "});Plotly.relayout('fer-sim-plot', {title: "
                + title_json
                + "});"
            )
            self.simulated_plot_view.page().runJavaScript(script)

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
            config = SimulationConfig(
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
                out=self.output.text(),
            )
            self.run_button.setEnabled(False)
            self.status.append("Starting simulation...")
            self.thread = QThread()
            self.worker = SimulationWorker(config)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.completed.connect(self._simulation_completed)
            self.worker.failed.connect(self._simulation_failed)
            self.worker.completed.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.finished.connect(self._thread_finished)
            self.thread.start()

        def _simulation_completed(self, artifact):
            self.current_artifact = Path(artifact)
            self.status.append(f"Simulation complete: {artifact}")

        def _simulation_failed(self, message):
            self.status.append(f"Simulation failed: {message}")

        def _thread_finished(self):
            self.run_button.setEnabled(True)
            self.worker = None
            self.thread = None

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_gui())
