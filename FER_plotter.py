"""Compatibility adapter; use fer_rf_sim.plotting for new code."""

from fer_rf_sim.plotting import *
from fer_rf_sim.plotting import _nearest, _interp_from_lut, _cheb_nodes, _T_time_avg, _T_grid, _log_current

if __name__ == "__main__":
    raise SystemExit(main())
