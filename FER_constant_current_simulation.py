"""Compatibility adapter; use fer_rf_sim.engine for new code."""

from fer_rf_sim.engine import *
from fer_rf_sim.engine import _cardano_two_smallest, _upscaled_axis, _merge_yaml_into_args

if __name__ == "__main__":
    print(cli())
