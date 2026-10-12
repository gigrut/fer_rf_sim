import os
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QPoint, QTimer, Signal
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea, QToolButton, QWidget

from fer_rf_sim import gui_widgets
from fer_rf_sim.config import SimulationConfig


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": np.arange(1., 7.).reshape(2, 3, 1),
                             "z": [1, 2], "V": [0, 1, 2], "A_rf": [0]}.items():
            file[name] = values
    return path


def make_window():
    return gui_widgets.MainWindow(config=SimulationConfig(n_E=8, n_V=3, n_Z=2, n_A=1),
                                  web_engine_view=None, auto_load_real_data=False)


@pytest.fixture
def probability_artifact(tmp_path):
    path = tmp_path / "probability.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": np.ones((2, 3, 2)), "z": [.5, 1.],
                             "V": [1., 2., 3.], "A_rf": [0., .4], "E": [.1, .2]}.items():
            file[name] = values
        file.attrs.update(phi_tip=4., phi_samp=5., n_cheb=4)
    return path


def test_current_heatmap_swaps_bias_and_height(application, artifact):
    from types import SimpleNamespace

    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    try:
        window.simulated_view_type.setCurrentText("Current heatmap")
        window.load_artifact(artifact)
        window.simulated_x_axis.setCurrentText("Tip height")
        figure = figures[-1]
        assert window.simulated_y_axis.currentText() == "Bias"
        assert figure.layout.xaxis.title.text == "Tip height (nm)"
        assert figure.layout.yaxis.title.text == "Bias voltage (V)"
        np.testing.assert_array_equal(figure.data[0].x, [1, 2])
        np.testing.assert_array_equal(figure.data[0].y, [0, 1, 2])
        np.testing.assert_allclose(figure.data[0].z, np.log10(np.arange(1., 7.).reshape(2, 3)).T)
        window.simulated_y_axis.setCurrentText("Tip height")
        assert window.simulated_x_axis.currentText() == "Bias"
        assert np.shape(figures[-1].data[0].z) == (2, 3)
    finally:
        window.close()


def test_probability_heatmap_energy_rf_scale_and_cache(application, probability_artifact, monkeypatch):
    from types import SimpleNamespace
    from fer_rf_sim.physics import transmission
    from fer_rf_sim.quadrature import chebyshev_phases

    calls = []
    original = gui_widgets.requested_transmission_grid
    monkeypatch.setattr(gui_widgets, "requested_transmission_grid",
                        lambda *args: (calls.append(1), original(*args))[1])
    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    try:
        window.simulated_view_type.setCurrentText("Tunneling probability heatmap")
        window.simulated_color_scale.setCurrentText("Linear")
        window.load_artifact(probability_artifact)
        assert not window.simulated_energy_controls.isHidden()
        assert not window.simulated_current_slider.isEnabled()
        assert "Recomputed" in window.constant_current_status.text()
        expected = transmission(.1, np.array([1., 2., 3.])[None, :], np.array([.5, 1.])[:, None], 4., 5.)
        np.testing.assert_allclose(figures[-1].data[0].z, expected, rtol=1e-12, atol=0)
        assert figures[-1].data[0].colorbar.title.text == "T (probability)"
        window.simulated_x_axis.setCurrentText("Tip height")
        np.testing.assert_allclose(figures[-1].data[0].z, expected.T, rtol=1e-12, atol=0)
        window.simulated_color_scale.setCurrentText("Log10")
        np.testing.assert_allclose(figures[-1].data[0].z, np.log10(expected.T), rtol=1e-12, atol=0)
        assert len(calls) == 1
        window.simulated_energy_slider.setValue(1)
        window.simulated_rf_slider.setValue(1)
        nodes = chebyshev_phases(4)
        expected = np.array([
            [transmission(.2, bias + .4 * np.cos(nodes), height, 4., 5.).mean()
             for bias in [1., 2., 3.]] for height in [.5, 1.]
        ])
        np.testing.assert_allclose(figures[-1].data[0].z, np.log10(expected.T), rtol=1e-12, atol=0)
        assert window.simulated_energy_value.text() == "0.2 eV"
        window.simulated_view_type.setCurrentText("Tunneling probability surface")
        assert figures[-1].data[0].type == "surface"
        assert len(calls) == 3
        window.load_artifact(probability_artifact)
        assert len(calls) == 4
    finally:
        window.close()


def test_missing_probability_data_keeps_current_views(application, artifact):
    from types import SimpleNamespace

    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    try:
        window.load_artifact(artifact)
        window.simulated_view_type.setCurrentText("Tunneling probability heatmap")
        assert "no electron-energy dataset" in window.constant_current_status.text()
        assert len(figures[-1].data) == 0
        assert not window.simulated_energy_slider.isEnabled()
        window.simulated_view_type.setCurrentText("Fixed-height dI/dV heatmap")
        window.simulated_color_scale.setCurrentText("Linear")
        np.testing.assert_allclose(figures[-1].data[0].z, np.ones((2, 3)))
        window.simulated_view_type.setCurrentText("Current heatmap")
        assert figures[-1].data[0].type == "heatmap"
        assert not window.constant_current_status.text()
    finally:
        window.close()


@pytest.mark.parametrize("size", [(1100, 740), (760, 520)])
def test_probability_webengine_smoke(probability_artifact, tmp_path, size):
    from pathlib import Path
    import subprocess
    import sys

    pytest.importorskip("PySide6.QtWebEngineWidgets")
    environment = dict(os.environ)
    if sys.platform == "win32":
        environment["QT_QPA_PLATFORM"] = "windows"
    result = subprocess.run([
        sys.executable, str(Path(__file__).with_name("gui_webengine_smoke.py")),
        str(probability_artifact), str(tmp_path / "probability.png"), "probability",
        str(size[0]), str(size[1]),
    ], env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rendered one heatmap" in result.stdout


def test_gui_launcher_imports_application():
    from fer_rf_sim.gui import run_gui

    assert callable(run_gui)
    assert gui_widgets.QApplication is QApplication


@pytest.mark.parametrize("size", [(1100, 740), (760, 520)])
def test_math_reference_is_offline_and_navigable(application, size):
    window = make_window()
    try:
        window.resize(*size)
        assert tuple(window.tabs.tabText(index) for index in range(window.tabs.count())) == (
            gui_widgets.TAB_NAMES
        )
        window.tabs.setCurrentIndex(window.tabs.count() - 1)
        window.show()
        application.processEvents()
        assert window.math_browser.isReadOnly()
        assert not window.math_browser.openExternalLinks()
        assert window.plot_document is None
        assert window.thread is None
        assert window.current_artifact is None
        reference = window.math_browser.toPlainText()
        for _, title, _ in gui_widgets.MATH_SECTIONS:
            assert title in reference
        assert "arbitrary units" in reference
        assert "d = Z in nm" in reference
        assert "V/nm" in reference
        assert "complete currents" in reference
        assert "half-prominence width" in reference
        assert "Existing inconsistency" in reference
        window.math_topics.setCurrentIndex(window.math_topics.count() - 1)
        application.processEvents()
        assert window.math_browser.verticalScrollBar().value() > 0
        window.math_topics.setCurrentIndex(0)
        assert window.math_browser.verticalScrollBar().value() == 0
        assert window.math_browser.width() > 500
    finally:
        window.close()


@pytest.mark.parametrize("size", [(1100, 740), (800, 560)])
def test_config_layout_keeps_fields_accessible(application, tmp_path, size):
    window = make_window()
    window.resize(*size)
    window.show()
    application.processEvents()
    scroll = window.tabs.widget(0)
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable()
    assert window.advanced_settings.isHidden()
    assert window.grab().save(str(tmp_path / "config-collapsed.png"))
    window.advanced_toggle.click()
    application.processEvents()
    assert not window.advanced_settings.isHidden()
    assert {group.title() for group in scroll.findChildren(QGroupBox)} == {
        "Methods", "Sampling", "Sweep ranges", "Work functions", "Fermi energies", "LUT",
        "DC-current table", "Execution",
    }
    for name in window._initial_values:
        field = window.output if name == "out" else getattr(window, name)
        origin = field.mapTo(scroll.widget(), QPoint(0, 0))
        assert origin.x() >= 0
        assert origin.x() + field.width() <= scroll.widget().width()
        assert field.width() >= field.minimumSizeHint().width()
        assert field.height() >= field.minimumSizeHint().height()
    scroll.ensureWidgetVisible(window.run_button)
    application.processEvents()
    origin = window.run_button.mapTo(scroll.viewport(), QPoint(0, 0))
    assert 0 <= origin.y() < scroll.viewport().height()
    assert window.run_button.property("role") == "primary"
    assert not window.run_button.icon().isNull()
    browse = next(button for button in scroll.findChildren(QToolButton)
                  if button.accessibleName() == "Choose output directory")
    assert browse.toolTip() == browse.accessibleName() == "Choose output directory"
    assert window.grab().save(str(tmp_path / "config-expanded.png"))
    window.close()


def test_plot_theme_preserves_axis_titles():
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    figure.update_layout(title="Spectrum", xaxis_title="Bias (V)", yaxis_title="Current")
    gui_widgets.MainWindow._style_plot(figure)
    assert figure.layout.title.text == "Spectrum"
    assert figure.layout.xaxis.title.text == "Bias (V)"
    assert figure.layout.yaxis.title.text == "Current"
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.paper_bgcolor == "#ffffff"


@pytest.mark.parametrize("size", [(1100, 740), (760, 520)])
def test_config_native_smoke(artifact, tmp_path, size):
    from pathlib import Path
    import subprocess
    import sys

    environment = dict(os.environ)
    if sys.platform == "win32":
        environment["QT_QPA_PLATFORM"] = "windows"
    result = subprocess.run([
        sys.executable, str(Path(__file__).with_name("gui_webengine_smoke.py")),
        str(artifact), str(tmp_path / "config.png"), "config", str(size[0]), str(size[1]),
    ], env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rendered without horizontal scrolling" in result.stdout


def test_current_heatmap_labels_units(application, artifact):
    from types import SimpleNamespace

    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    window.simulated_view_type.setCurrentText("Current heatmap")
    window.load_artifact(artifact)
    figure = figures[-1]
    assert figure.layout.xaxis.title.text == "Bias voltage (V)"
    assert figure.layout.yaxis.title.text == "Tip height (nm)"
    assert figure.data[0].type == "heatmap"
    window.close()


def wait_for_thread(window):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    window.thread.finished.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    assert window.thread is None


def test_constant_current_dzdv_views_match_peak_extraction(application, artifact):
    from types import SimpleNamespace
    from fer_rf_sim.peaks import simulated_peak_spectra

    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    try:
        window.load_artifact(artifact)
        target = float(window.constant_current_targets[window.simulated_current_slider.value()])
        expected = simulated_peak_spectra(
            window.simulated_data, mode="constant-current dZ/dV", current_target=target,
        )[0].signal
        window.simulated_view_type.setCurrentText("Constant-current dZ/dV trace")
        assert figures[-1].layout.yaxis.title.text == "dZ/dV (nm/V)"
        np.testing.assert_allclose(figures[-1].data[0].y, expected, equal_nan=True)
        window.simulated_view_type.setCurrentText("Constant-current dZ/dV heatmap")
        assert figures[-1].data[0].colorbar.title.text == "dZ/dV (nm/V)"
        np.testing.assert_allclose(figures[-1].data[0].z[0], expected, equal_nan=True)
    finally:
        window.close()


def test_heatmap_color_range_clip_and_manual(application, tmp_path):
    from types import SimpleNamespace

    # Spread the bulk of the values so the 5% clip yields a non-degenerate range.
    current = np.linspace(1.0, 2.0, 100).reshape(10, 10, 1)
    current[0, 0, 0] = 1e6
    path = tmp_path / "outlier.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": current, "z": np.arange(10.), "V": np.arange(10.), "A_rf": [0]}.items():
            file[name] = values
    window = make_window()
    figures = []
    window.simulated_host = SimpleNamespace(update=figures.append, close=lambda: None)
    try:
        window.simulated_view_type.setCurrentText("Current heatmap")
        window.simulated_color_scale.setCurrentText("Linear")
        window.load_artifact(path)
        assert figures[-1].data[0].zmax is None
        assert not window.simulated_color_min.isEnabled()
        window.simulated_color_range.setCurrentText("Clip 5%")
        clipped_max = figures[-1].data[0].zmax
        assert clipped_max is not None and 1.0 < clipped_max < 2.5
        window.simulated_color_range.setCurrentText("Manual")
        assert window.simulated_color_max.isEnabled()
        window.simulated_color_min.setText("0.5")
        window.simulated_color_max.setText("2")
        window.simulated_color_max.editingFinished.emit()
        assert (figures[-1].data[0].zmin, figures[-1].data[0].zmax) == (0.5, 2.0)
    finally:
        window.close()


def test_typed_slider_values_snap_to_grid(application, probability_artifact):
    window = make_window()
    try:
        window.load_artifact(probability_artifact)
        window.simulated_rf_value.setText("0.3 V")
        window.simulated_rf_value.editingFinished.emit()
        assert window.simulated_rf_slider.value() == 1
        assert window.simulated_rf_value.text() == "0.4 V"
        window.simulated_view_type.setCurrentText("Constant-current dI/dV trace")
        window.simulated_current_value.setText("2e-9")
        window.simulated_current_value.editingFinished.emit()
        target = window.constant_current_targets[window.simulated_current_slider.value()]
        assert target == pytest.approx(2e-9, rel=0.02)
        window.simulated_current_value.setText("abc")
        window.simulated_current_value.editingFinished.emit()
        assert window.simulated_current_value.text() == f"{target:.3e} A"
    finally:
        window.close()


def test_gui_derivatives_cached_and_setpoints_bounded(application, artifact, monkeypatch):
    calls = []
    original = gui_widgets.build_current_array
    monkeypatch.setattr(gui_widgets, "build_current_array",
                        lambda *args: (calls.append(1), original(*args))[1])
    window = make_window()
    window.load_artifact(artifact)
    for index in range(30):
        window.simulated_current_slider.setValue(index)
        window._constant_current_heatmap_data()
    assert len(calls) == 1
    assert len(window.constant_current_cache) == 16
    window.load_artifact(artifact)
    assert not window.constant_current_cache
    assert len(calls) == 2
    window.close()


def test_gui_repeated_runs_failure_and_deferred_close(application, artifact, monkeypatch):
    window = make_window()
    assert window.simulation_progress.isHidden()
    monkeypatch.setattr(gui_widgets.SimulationRunner, "run",
                        lambda self, config, **kwargs: artifact)
    window._run_simulation()
    assert not window.simulation_progress.isHidden()
    assert window.run_button.text() == "Cancel simulation"
    first_thread = window.thread
    assert window.thread is first_thread
    wait_for_thread(window)
    assert window.run_button.isEnabled()
    assert window.run_button.text() == "Run simulation"
    assert window.simulation_progress.isHidden()

    def fail(self, config, **kwargs):
        raise ValueError("test failure")

    monkeypatch.setattr(gui_widgets.SimulationRunner, "run", fail)
    window._run_simulation()
    assert not window.simulation_progress.isHidden()
    wait_for_thread(window)
    assert "test failure" in window.status.toPlainText()
    assert window.simulation_progress.isHidden()
    window._run_simulation()
    window.close()
    assert window._close_requested
    wait_for_thread(window)
    assert window.worker is None


def test_gui_progress_updates_and_simulation_can_be_cancelled(application, artifact, monkeypatch):
    window = make_window()
    started = Event()

    def wait_for_cancel(self, config, *, progress_callback, cancel_event):
        started.set()
        cancel_event.wait(3)
        raise InterruptedError("Simulation cancelled by user")

    monkeypatch.setattr(gui_widgets.SimulationRunner, "run", wait_for_cancel)
    window._run_simulation()
    assert started.wait(2)
    window.worker.progress.emit("Current grid", 3, 10)
    application.processEvents()
    assert window.simulation_progress.value() == 3
    assert window.simulation_progress.maximum() == 10
    assert window.simulation_progress.format() == "Current grid: %v/%m"
    window.run_button.click()
    wait_for_thread(window)
    assert "Simulation cancelled." in window.status.toPlainText()
    assert window.run_button.text() == "Run simulation"
    assert window.simulation_progress.isHidden()
    window.close()


def test_plotly_document_shared_and_removed(application):
    class FakeView(QWidget):
        loadFinished = Signal(bool)

        def __init__(self):
            super().__init__()
            self.scripts = []

        def load(self, url):
            self.url = url
            self.loadFinished.emit(True)

        def page(self):
            return self

        def runJavaScript(self, script):
            self.scripts.append(script)

        def stop(self):
            pass

        def setUrl(self, url):
            self.url = url

    window = gui_widgets.MainWindow(config=SimulationConfig(), web_engine_view=FakeView,
                                    auto_load_real_data=False)
    path = window.plot_document.path
    assert window.real_plot_view.url == window.simulated_plot_view.url
    assert path.is_file()
    window.close()
    assert not path.exists()


def test_unchanged_gui_controls_preserve_precise_config(application, artifact, monkeypatch):
    config = SimulationConfig(phi_tip=4.123456789, use_lut=True, threads=1,
                              dc_table_rtol=0.00012345678901234567,
                              dc_table_atol=1.234567891234567e-18)
    window = gui_widgets.MainWindow(config=config, web_engine_view=None, auto_load_real_data=False)
    captured = []

    def run(self, actual, **kwargs):
        captured.append(actual)
        return artifact

    monkeypatch.setattr(gui_widgets.SimulationRunner, "run", run)
    window._run_simulation()
    wait_for_thread(window)
    assert captured[0].phi_tip == config.phi_tip
    assert captured[0].use_lut is True
    assert captured[0].dc_table_rtol == config.dc_table_rtol
    assert captured[0].dc_table_atol == config.dc_table_atol
    assert captured[0].upscale_z is None
    window.close()


@pytest.mark.parametrize("use_lut,dc_table", [(False, False), (True, False), (False, True), (True, True)])
def test_simulation_method_selectors_reach_worker(application, artifact, monkeypatch, use_lut, dc_table):
    config = SimulationConfig(use_lut=not use_lut, dc_table=not dc_table, threads=1)
    window = gui_widgets.MainWindow(config=config, web_engine_view=None, auto_load_real_data=False)
    captured = []

    def run(self, actual, **kwargs):
        captured.append(actual)
        return artifact

    monkeypatch.setattr(gui_widgets.SimulationRunner, "run", run)
    try:
        assert window.use_lut.currentData() == config.use_lut
        assert window.dc_table.currentData() == config.dc_table
        assert window.lut_options.isEnabled() == config.use_lut
        assert window.dc_table_options.isEnabled() == config.dc_table
        window.use_lut.setCurrentIndex(int(use_lut))
        window.dc_table.setCurrentIndex(int(dc_table))
        assert window.lut_options.isEnabled() == use_lut
        assert window.dc_table_options.isEnabled() == dc_table
        window._run_simulation()
        wait_for_thread(window)
        assert captured[0].use_lut is use_lut
        assert captured[0].dc_table is dc_table
        assert f"Transmission: {'LUT' if use_lut else 'Direct'}" in window.status.toPlainText()
    finally:
        window.close()


@pytest.mark.parametrize("field,value", [("threads", 0), ("dc_table_max_points", 2)])
def test_invalid_advanced_settings_do_not_start_worker(application, field, value):
    window = make_window()
    try:
        getattr(window, field).setValue(value)
        window._run_simulation()
        assert window.thread is None
        assert window.run_button.isEnabled()
        assert "Invalid configuration" in window.status.toPlainText()
    finally:
        window.close()


def test_advanced_simulation_controls_reach_worker(application, artifact, monkeypatch):
    window = make_window()
    captured = []

    def run(self, config, **kwargs):
        captured.append(config)
        return artifact

    monkeypatch.setattr(gui_widgets.SimulationRunner, "run", run)
    try:
        window.use_lut.setCurrentIndex(1)
        window.dc_table.setCurrentIndex(1)
        window.upscale.setValue(2)
        window.upscale_z.setValue(0)
        window.upscale_v.setValue(3)
        window.upscale_e.setValue(4)
        window.force_rebuild_lut.setChecked(True)
        window.dc_table_points.setValue(33)
        window.dc_table_max_points.setValue(2049)
        window.dc_table_rtol.setText("2.5e-4")
        window.dc_table_atol.setText("1.25e-18")
        window.threads.setValue(2)
        window.amplitude_chunk.setValue(3)
        window.E_F_tip.setValue(5.0)
        window.E_F_samp.setValue(4.5)
        window._run_simulation()
        wait_for_thread(window)
        actual = captured[0]
        assert (actual.upscale, actual.upscale_z, actual.upscale_v, actual.upscale_e) == (2, None, 3, 4)
        assert actual.force_rebuild_lut
        assert (actual.dc_table_points, actual.dc_table_max_points) == (33, 2049)
        assert actual.dc_table_rtol == 2.5e-4
        assert actual.dc_table_atol == 1.25e-18
        assert (actual.threads, actual.amplitude_chunk) == (2, 3)
        assert (actual.E_F_tip, actual.E_F_samp) == (5.0, 4.5)
    finally:
        window.close()


@pytest.mark.parametrize("invalid", ["", "not-a-number", "nan", "-1"])
def test_invalid_table_tolerance_does_not_start_worker(application, invalid):
    window = make_window()
    try:
        window.dc_table.setCurrentIndex(1)
        window.dc_table_rtol.setText(invalid)
        window._run_simulation()
        assert window.thread is None
        assert window.run_button.isEnabled()
        assert "Invalid configuration" in window.status.toPlainText()
    finally:
        window.close()


@pytest.mark.parametrize("cache_status", ["built", "reused"])
def test_completed_run_logs_actual_lut_cache_status(application, artifact, cache_status):
    with h5py.File(artifact, "a") as file:
        file.attrs["lut_cache_status"] = cache_status
        file.attrs["lut_file"] = "transmission_lut_example.h5"
    window = make_window()
    try:
        window._simulation_completed(str(artifact))
        assert f"LUT cache: {cache_status} (transmission_lut_example.h5)" in window.status.toPlainText()
    finally:
        window.close()


def test_real_webengine_smoke(artifact, tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    environment = dict(os.environ)
    if sys.platform == "win32":
        environment["QT_QPA_PLATFORM"] = "windows"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("gui_webengine_smoke.py")),
                             str(artifact), str(tmp_path / "gui.png")],
                            env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rendered one heatmap" in result.stdout