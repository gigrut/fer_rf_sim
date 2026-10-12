"""Native WebEngine rendering check for measured and simulated peak workflows."""

import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from fer_rf_sim.config import SimulationConfig
from fer_rf_sim.gui_widgets import MainWindow


def main():
    application = QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    artifact, real, destination, source, width, height = sys.argv[1:]
    window = MainWindow(config=SimulationConfig(), auto_load_real_data=False)
    panel = window.peak_analysis
    window.resize(int(width), int(height))
    document = window.plot_document.path
    outcome = {"passed": False, "error": "Timed out waiting for peak plots"}
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(application.quit)
    timer.start(30000)

    def fail(error):
        outcome["error"] = str(error)
        application.quit()

    def screenshot(view, name):
        page = panel.peak_page if view is panel.peak_plot_view else panel.rf_page
        window_image = window.grab()
        window_image.save(str(Path(destination) / f"window-{name}"))
        origin = view.mapTo(window, QPoint(0, 0))
        viewport = page.viewport()
        viewport_origin = viewport.mapTo(window, QPoint(0, 0))
        region = QRect(origin, view.size()).intersected(QRect(viewport_origin, viewport.size()))
        image = window_image.copy(region)
        assert not image.isNull(), "Plot view produced a null image"
        image.save(str(Path(destination) / name))
        pixels = image.toImage()
        colored = 0
        for vertical in range(0, pixels.height(), 3):
            for horizontal in range(0, pixels.width(), 3):
                color = pixels.pixelColor(horizontal, vertical)
                channels = (color.red(), color.green(), color.blue())
                colored += max(channels) - min(channels) > 40
        assert colored > 20, f"{name}: plot pixels are blank: {colored} colored samples"

    def capture_rf():
        try:
            screenshot(panel.rf_plot_view, "rf.png")
            window.grab().save(str(Path(destination) / "rf-window.png"))
            outcome["passed"] = True
            timer.stop()
            application.quit()
        except Exception as error:
            fail(error)

    def inspect_rf(value):
        try:
            result = json.loads(value)
            assert result["traces"] == 1
            assert result["points"] == 3
            assert result["unit"] == "Median shift (V)"
            application.processEvents()
            panel.rf_page.ensureWidgetVisible(panel.rf_plot_view)
            QTimer.singleShot(250, capture_rf)
        except Exception as error:
            fail(error)

    def read_rf():
        script = (
            "JSON.stringify((() => {const plot=document.getElementById('fer-plot');"
            "return {traces:plot.data.length,points:plot.calcdata[0].length,"
            "unit:plot.layout.yaxis.title.text};})())"
        )
        panel.rf_plot_view.page().runJavaScript(script, inspect_rf)

    def rf_ready(success):
        if success:
            panel.rf_plot_view.loadFinished.disconnect(rf_ready)
            QTimer.singleShot(250, read_rf)

    def capture_peaks():
        try:
            screenshot(panel.peak_plot_view, "peaks.png")
            window.tabs.setCurrentIndex(4)
            panel.rf_metric.setCurrentText("shift")
            application.processEvents()
            panel.rf_page.ensureWidgetVisible(panel.rf_plot_view)
            if panel.rf_host.ready:
                QTimer.singleShot(250, read_rf)
            else:
                panel.rf_plot_view.loadFinished.connect(rf_ready)
        except Exception as error:
            fail(error)

    def inspect_peaks(value):
        try:
            result = json.loads(value)
            assert result["traces"] == 2
            assert result["peaks"] == 1
            application.processEvents()
            panel.peak_page.ensureWidgetVisible(panel.peak_plot_view)
            QTimer.singleShot(250, capture_peaks)
        except Exception as error:
            fail(error)

    def read_peaks():
        script = (
            "JSON.stringify((() => {const plot=document.getElementById('fer-plot');"
            "return {traces:plot.data.length,peaks:plot.calcdata[1].length};})())"
        )
        panel.peak_plot_view.page().runJavaScript(script, inspect_peaks)

    def peak_ready(success):
        if success:
            panel.peak_plot_view.loadFinished.disconnect(peak_ready)
            QTimer.singleShot(250, read_peaks)

    if source == "real":
        window._load_real_data(real)
        panel.peak_source.setCurrentText("Real data")
    else:
        window.load_artifact(artifact)
        panel.peak_mode.setCurrentText("fixed-height dI/dV")
    panel.peak_prominence.setValue(.1)
    window.tabs.setCurrentIndex(3)
    window.show()
    panel.peak_plot_view.loadFinished.connect(peak_ready)
    panel.extract_button.click()
    panel.peak_page.ensureWidgetVisible(panel.peak_plot_view)
    application.exec()
    timer.stop()
    window.close()
    assert outcome["passed"], outcome["error"]
    assert not document.exists(), "Shared Plotly document survived window close"
    print(f"Rendered {source} peak overlay and RF dependence at {width}x{height}")


if __name__ == "__main__":
    main()