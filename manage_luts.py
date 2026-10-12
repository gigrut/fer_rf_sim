"""Compatibility adapter; use fer_rf_sim.lut_cli for new code."""

from fer_rf_sim.lut_cli import *

if __name__ == "__main__":
    raise SystemExit(main())
