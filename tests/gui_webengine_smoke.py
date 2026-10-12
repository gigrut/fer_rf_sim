"""Real WebEngine subprocess smoke; invoked by test_gui.py when available."""

import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from fer_rf_sim.config import SimulationConfig
from fer_rf_sim.gui_widgets import MainWindow


def config_smoke(application):
    window = MainWindow(config=SimulationConfig(), web_engine_view=None, auto_load_real_data=False)
    window.resize(int(sys.argv[4]), int(sys.argv[5]))
    screenshot = Path(sys.argv[2])
    outcome = {}
    window.show()

    def capture():
        window.use_lut.setCurrentIndex(1)
        window.dc_table.setCurrentIndex(1)
        application.processEvents()
        outcome["collapsed_capture"] = window.grab().save(str(screenshot))
        window.advanced_toggle.click()
        application.processEvents()
        scroll = window.tabs.widget(0)
        scroll.ensureWidgetVisible(window.force_rebuild_lut)
        application.processEvents()
        outcome["expanded_capture"] = window.grab().save(
            str(screenshot.with_name(screenshot.stem + "-advanced.png")),
        )
        outcome["no_horizontal_scroll"] = scroll.horizontalScrollBar().maximum() == 0
        outcome["lut_enabled"] = window.force_rebuild_lut.isEnabled()
        outcome["table_enabled"] = window.dc_table_atol.isEnabled()
        window.close()
        application.quit()

    QTimer.singleShot(100, capture)
    QTimer.singleShot(5000, application.quit)
    application.exec()
    window.close()
    assert outcome and all(outcome.values()), f"Config layout failed: {outcome}"
    print("Config methods and advanced settings rendered without horizontal scrolling")


def main():
    application = QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    if len(sys.argv) > 3 and sys.argv[3] == "config":
        config_smoke(application)
        return
    window = MainWindow(config=SimulationConfig(), auto_load_real_data=False)
    probability = len(sys.argv) > 3 and sys.argv[3] == "probability"
    if probability:
        window.resize(int(sys.argv[4]), int(sys.argv[5]))
    window.tabs.setCurrentIndex(1)
    window.show()
    artifact = Path(sys.argv[1])
    screenshot = Path(sys.argv[2])
    document = window.plot_document.path
    outcome = {"passed": False, "capture_attempts": 0}
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(application.quit)
    timer.start(30000)

    def capture():
        image = window.grab()
        image.save(str(screenshot))
        pixels = window.simulated_plot_view.grab().toImage()
        colored = 0
        for vertical in range(0, pixels.height(), max(1, pixels.height() // 30)):
            for horizontal in range(pixels.width() // 8, 7 * pixels.width() // 8, max(1, pixels.width() // 40)):
                color = pixels.pixelColor(horizontal, vertical)
                channels = (color.red(), color.green(), color.blue())
                colored += max(channels) - min(channels) > 20
        outcome["colored_pixels"] = colored
        outcome["capture_attempts"] += 1
        if colored <= 10 and outcome["capture_attempts"] < 30:
            QTimer.singleShot(100, capture)
            return
        outcome["passed"] = outcome["passed"] and not image.isNull() and colored > 10
        timer.stop()
        print(f"WebEngine paint check: {outcome}", flush=True)
        window.close()
        QTimer.singleShot(0, application.quit)

    def finish(value):
        result = json.loads(value)
        outcome["passed"] = result["traces"] == 1 and result["type"] == "heatmap"
        QTimer.singleShot(250, capture)

    def loaded(success):
        window.simulated_plot_view.loadFinished.disconnect(loaded)
        if not success:
            application.quit()
            return
        window.simulated_view_type.setCurrentText("Current heatmap")
        if probability:
            window.simulated_view_type.setCurrentText("Tunneling probability heatmap")
            window.simulated_x_axis.setCurrentText("Tip height")
        window.load_artifact(artifact)
        script = ("JSON.stringify({traces: document.getElementById('fer-plot').data.length, "
                  "type: document.getElementById('fer-plot').data[0].type})")
        window.simulated_plot_view.page().runJavaScript(script, finish)

    window.simulated_plot_view.loadFinished.connect(loaded)
    application.exec()
    if not outcome["passed"]:
        window.close()
    assert outcome["passed"], f"Plotly heatmap did not render through WebEngine: {outcome}"
    assert not document.exists(), "Temporary Plotly HTML survived window close"
    print("WebEngine rendered one heatmap and removed its HTML document")


if __name__ == "__main__":
    main()