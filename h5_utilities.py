"""Compatibility adapter; use fer_rf_sim.hdf5 for new code."""

from fer_rf_sim.hdf5 import *

if __name__ == "__main__":
    raise SystemExit(main())
