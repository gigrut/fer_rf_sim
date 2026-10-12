import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget

from fer_rf_sim.config import SimulationConfig
from fer_rf_sim.gui_widgets import MainWindow
from fer_rf_sim import peak_widgets


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sources(tmp_path):
    voltage = np.linspace(0, 4, 401)
    amplitudes = np.array([0., .5, 1.])
    signals = np.column_stack([
        np.exp(-((voltage - (1 + .2 * amplitude)) / .1) ** 2) for amplitude in amplitudes
    ])
    current = np.zeros_like(signals)
    current[1:] = np.cumsum((signals[1:] + signals[:-1]) * np.diff(voltage)[:, None] / 2, axis=0)
    artifact = tmp_path / "peaks.h5"
    with h5py.File(artifact, "w") as file:
        for name, values in {"I": current[None, ...], "z": [1.], "V": voltage, "A_rf": amplitudes}.items():
            file[name] = values
    real = tmp_path / "real"
    real.mkdir()
    for index, amplitude in enumerate(amplitudes):
        pd.DataFrame({"bias (mV)": voltage * 1000, "lockin-x (mV)": signals[:, index] * 1000,
                      "RF (V)": amplitude}).to_csv(real / f"{index}.csv", index=False)
    return artifact, real


def make_window(**options):
    return MainWindow(config=SimulationConfig(n_E=8, n_V=3, n_Z=2, n_A=1),
                      web_engine_view=options.get("web_engine_view"), auto_load_real_data=False)


def test_peak_extraction_real_and_simulated(application, sources):
    artifact, real = sources
    window = make_window()
    panel = window.peak_analysis
    try:
        window.load_artifact(artifact)
        panel.peak_mode.setCurrentText("fixed-height dI/dV")
        panel.peak_prominence.setValue(.1)
        panel.extract_button.click()
        assert len(panel.peak_results) == 3
        assert panel.peak_table.rowCount() == 1
        assert panel.peak_export_button.isEnabled()
        assert panel.peak_results[1].peaks.position[0] == pytest.approx(1.1)
        assert panel.rf_table.rowCount() == 3
        assert [peak.shift for peak in panel.rf_peaks] == pytest.approx([0., .1, .2])
        assert panel.rf_export_button.isEnabled()
        assert "fixed-height dI/dV" in panel.peak_spectrum.itemText(0)
        panel.peak_source.setCurrentText("Real data")
        assert not panel.peak_results
        assert not panel.peak_mode.isEnabled()
        window._load_real_data(real)
        panel.extract_button.click()
        assert len(panel.peak_results) == 3
        assert panel.peak_results[2].peaks.position[0] == pytest.approx(1.2)
        assert panel.peak_table.horizontalHeaderItem(1).text() == "Height (V)"
        assert panel.rf_table.rowCount() == 3
        assert panel.rf_source.currentText() == "Real data"
        panel.rf_max_shift.setValue(.05)
        assert [peak.track for peak in panel.rf_peaks] == [1, 2, 3]
        panel.peak_polarity.setCurrentText("negative")
        assert not panel.peak_results
        assert panel.peak_table.rowCount() == panel.rf_table.rowCount() == 0
        assert not panel.peak_export_button.isEnabled()
        assert not panel.rf_export_button.isEnabled()
    finally:
        window.close()


def test_peak_errors_clear_stale_results(application, sources):
    window = make_window()
    panel = window.peak_analysis
    try:
        assert not panel.extract_button.isEnabled()
        window.load_artifact(sources[0])
        panel.peak_target.setText("not a number")
        panel.extract_button.click()
        assert "failed" in panel.peak_status.text()
        panel.peak_target.setText("nan")
        panel.extract_button.click()
        assert "finite" in panel.peak_status.text()
        window.load_artifact("does-not-exist.h5")
        assert not panel.peak_results
        assert panel.peak_height.count() == 0
        assert not panel.extract_button.isEnabled()
    finally:
        window.close()


def test_unknown_rf_is_extractable_and_can_be_assigned(application, tmp_path):
    for index in range(2):
        pd.DataFrame({"bias (mV)": [0., 1000., 2000.], "lockin-x (mV)": [0., 2., 0.]}).to_csv(
            tmp_path / f"{index}.csv", index=False,
        )
    window = make_window()
    panel = window.peak_analysis
    try:
        window._load_real_data(tmp_path)
        panel.peak_source.setCurrentText("Real data")
        panel.extract_button.click()
        assert sum(result.peaks.position.size for result in panel.peak_results) == 2
        assert not panel.rf_peaks
        assert "two distinct" in panel.rf_status.text()
        assert "2 unknown-RF" in panel.rf_status.text()
        assert not panel.rf_export_button.isEnabled()
        panel.real_rf_input.setText("-1")
        panel.real_rf_button.click()
        assert window.real_spectra[0].rf_amplitude is None
        for index in range(2):
            panel.real_spectrum_selector.setCurrentIndex(index)
            panel.real_rf_input.setText(str(index))
            panel.real_rf_button.click()
            assert window.real_spectra[index].rf_source == "manual"
        panel.extract_button.click()
        assert len(panel.rf_peaks) == 2
        assert {peak.track for peak in panel.rf_peaks} == {1}
        assert panel.rf_export_button.isEnabled()
        window._load_real_data(tmp_path)
        assert not panel.peak_results
        assert not panel.rf_peaks
        assert all(spectrum.rf_amplitude is None for spectrum in window.real_spectra)
    finally:
        window.close()


def test_rf_export_preserves_tracks_and_units(application, sources, tmp_path, monkeypatch):
    window = make_window()
    panel = window.peak_analysis
    try:
        window._load_real_data(sources[1])
        panel.peak_source.setCurrentText("Real data")
        panel.extract_button.click()
        destination = tmp_path / "export.csv"
        monkeypatch.setattr(peak_widgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
        panel.rf_export_button.click()
        with destination.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["track"] for row in rows] == ["1", "1", "1"]
        assert [float(row["shift_v"]) for row in rows] == pytest.approx([0., .1, .2])
        assert all(row["signal_unit"] == "V" for row in rows)
    finally:
        window.close()


def test_analysis_hosts_load_lazily_and_clear(application, sources):
    class FakeView(QWidget):
        loadFinished = Signal(bool)

        def __init__(self):
            super().__init__()
            self.loads = []
            self.scripts = []
            self.figures = []

        def load(self, url):
            self.loads.append(url)
            self.loadFinished.emit(True)

        def page(self):
            return self

        def runJavaScript(self, script):
            self.scripts.append(script)
            payload = script.split("JSON.parse(", 1)[1].split(");Plotly.react", 1)[0]
            self.figures.append(json.loads(json.loads(payload)))

        def stop(self):
            pass

        def setUrl(self, url):
            self.url = url

    window = make_window(web_engine_view=FakeView)
    panel = window.peak_analysis
    document = window.plot_document.path
    try:
        assert not panel.peak_host.started
        assert not panel.rf_host.started
        assert not panel.peak_plot_view.loads
        window.load_artifact(sources[0])
        panel.peak_mode.setCurrentText("fixed-height dI/dV")
        panel.extract_button.click()
        assert len(panel.peak_plot_view.loads) == len(panel.rf_plot_view.loads) == 1
        panel.rf_metric.setCurrentText("height")
        assert panel.rf_plot_view.figures[-1]["layout"]["yaxis"]["title"]["text"] == "Median height (a.u./V)"
        panel.invalidate()
        assert not panel.peak_results
        assert not panel.rf_peaks
        assert not panel.peak_plot_view.isVisible()
    finally:
        window.close()
    assert not document.exists()


@pytest.mark.parametrize("size", [(1100, 740), (760, 520)])
def test_peak_pages_fit_narrow_windows(application, size, sources):
    window = make_window()
    panel = window.peak_analysis
    try:
        window.resize(*size)
        window.load_artifact(sources[0])
        panel.peak_mode.setCurrentText("fixed-height dI/dV")
        panel.extract_button.click()
        for index, page in [(3, panel.peak_page), (4, panel.rf_page)]:
            window.tabs.setCurrentIndex(index)
            window.show()
            application.processEvents()
            assert page.horizontalScrollBar().maximum() == 0
            page.ensureWidgetVisible(panel.extract_button if index == 3 else panel.rf_export_button)
            application.processEvents()
    finally:
        window.close()


@pytest.mark.parametrize("source", ["real", "simulated"])
@pytest.mark.parametrize("size", [(1100, 740), (760, 520)])
def test_native_peak_webengine_smoke(sources, tmp_path, source, size):
    from pathlib import Path
    import subprocess
    import sys

    pytest.importorskip("PySide6.QtWebEngineWidgets")
    environment = dict(os.environ)
    if sys.platform == "win32":
        environment["QT_QPA_PLATFORM"] = "windows"
    result = subprocess.run([
        sys.executable, str(Path(__file__).with_name("peak_webengine_smoke.py")),
        str(sources[0]), str(sources[1]), str(tmp_path), source, str(size[0]), str(size[1]),
    ], env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Rendered {source} peak overlay and RF dependence" in result.stdout
    assert (tmp_path / "peaks.png").is_file()
    assert (tmp_path / "rf.png").is_file()