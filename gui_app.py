"""PySide6 desktop shell for the FER simulation and analysis services."""

from __future__ import annotations

import sys
import tempfile
import json
import os
from pathlib import Path

import numpy as np
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
            self.real_data_path: Path | None = None

            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)
            self._build_config_tab()
            self._build_simulated_data_tab(QWebEngineView)
            self._build_real_data_tab()
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
            self.phi_tip = self._number_field(4.0)
            self.phi_samp = self._number_field(5.0)
            self.output = QLineEdit("fer_output")

            form.addRow("Energy points", self.n_E)
            form.addRow("Voltage points", self.n_V)
            form.addRow("Height points", self.n_Z)
            form.addRow("RF amplitude points", self.n_A)
            form.addRow("RF quadrature points", self.n_cheb)
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

        def _build_real_data_tab(self):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(
                "Real-data files are not included in this repository. Supported sources:"
            ))
            layout.addWidget(QLabel(
                f"Project convention: {Path.cwd() / 'data' / 'raw'}\n"
                "Environment override: FER_REAL_DATA_CSV\n"
                "Legacy scripts may also reference CSV files in Downloads or Desktop."
            ))

            select_file = QPushButton("Select real-data CSV")
            select_file.clicked.connect(self._select_real_data_file)
            layout.addWidget(select_file)
            select_directory = QPushButton("Select real-data folder")
            select_directory.clicked.connect(self._select_real_data_directory)
            layout.addWidget(select_directory)

            self.real_data_status = QLabel("No real-data source selected.")
            self.real_data_status.setWordWrap(True)
            layout.addWidget(self.real_data_status)
            layout.addStretch(1)
            self.tabs.addTab(page, "Real Data")

        def _set_real_data_path(self, path):
            self.real_data_path = Path(path)
            self.real_data_status.setText(
                f"Selected real-data source:\n{self.real_data_path}"
            )

        def _select_real_data_file(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select real-data CSV",
                os.environ.get("FER_REAL_DATA_CSV", str(Path.cwd() / "data" / "raw")),
                "CSV files (*.csv);;All files (*)",
            )
            if path:
                self._set_real_data_path(path)

        def _select_real_data_directory(self):
            path = QFileDialog.getExistingDirectory(
                self,
                "Select folder containing real-data CSV files",
                str(Path.cwd() / "data" / "raw"),
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
            self.simulated_view_type.addItems(["Current surface", "Current heatmap"])
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
            with np.errstate(divide="ignore", invalid="ignore"):
                current_slice = np.log10(np.abs(data.current[:, :, rf_index]))
            current_slice[~np.isfinite(current_slice)] = np.nan
            self.simulated_rf_value.setText(f"{rf_value:.4g} V")
            plot_type = self.simulated_view_type.currentText()
            if self.simulated_plot_initialized and plot_type == self.simulated_plot_type:
                self._update_simulated_trace(current_slice, rf_value)
                return

            title = f"log10(|Current|) at RF amplitude {rf_value:.4g} V"
            if plot_type == "Current heatmap":
                figure = go.Figure(go.Heatmap(
                    z=current_slice,
                    x=data.voltage,
                    y=data.z,
                    colorscale="Viridis",
                    colorbar={"title": "log10(|I|)"},
                ))
            else:
                figure = go.Figure(go.Surface(
                    z=current_slice,
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

        def _simulated_plot_loaded(self, success):
            self.simulated_plot_initialized = bool(success)

        def _update_simulated_trace(self, current_slice, rf_value):
            """Update the existing Plotly trace without rebuilding the page."""
            values = [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in current_slice
            ]
            z_json = json.dumps([values], separators=(",", ":"))
            title_json = json.dumps(
                f"log10(|Current|) at RF amplitude {rf_value:.4g} V"
            )
            script = (
                "Plotly.restyle('fer-sim-plot', {z: "
                + z_json
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
