"""Optional desktop launcher; importing this module does not import Qt."""

import sys

TAB_NAMES = ("Config + Run", "Simulated Data", "Real Data", "Peak Extraction", "RF Dependence")


def run_gui():
    try:
        from .gui_widgets import QApplication, MainWindow
    except ModuleNotFoundError as error:
        if error.name != "PySide6" and not (error.name or "").startswith("PySide6."):
            raise
        raise RuntimeError("Install fer-rf-sim[gui] to run the desktop application") from error
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_gui())
