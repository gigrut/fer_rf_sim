"""Compatibility adapter; use fer_rf_sim.gui for new code."""

from fer_rf_sim.gui import *

if __name__ == "__main__":
    raise SystemExit(run_gui())
