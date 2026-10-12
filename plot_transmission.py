"""Compatibility adapter; use fer_rf_sim.transmission_cli for new code."""

from fer_rf_sim.transmission_cli import *

if __name__ == "__main__":
    raise SystemExit(main())
