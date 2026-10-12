"""Compatibility adapter; use fer_rf_sim.workflow for new code."""

from fer_rf_sim.workflow import *
from fer_rf_sim.workflow import _artifact, _run_analysis

if __name__ == "__main__":
    raise SystemExit(main())
