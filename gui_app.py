"""PySide6 desktop shell for the FER simulation and analysis services."""

from __future__ import annotations

import sys
from pathlib import Path

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
        from PySide6.QtCore import QObject, QThread, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPushButton,
            QSpinBox,
            QDoubleSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise RuntimeError(
            "PySide6 is required to run the GUI. Install dependencies first."
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

            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)
            self._build_config_tab()
            self._build_placeholder_tab("Simulated Data", "Select a simulation artifact to plot current, dI/dV, RF slices, and constant-current views.")
            self._build_placeholder_tab("Real Data", "Load measured spectra or fitted-peak CSV data for inspection and filtering.")
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
