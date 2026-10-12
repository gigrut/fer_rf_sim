"""Compatibility adapter; use fer_rf_sim.constant_current for new code."""

from fer_rf_sim.constant_current import *

if __name__ == "__main__":
    raise SystemExit(main())
