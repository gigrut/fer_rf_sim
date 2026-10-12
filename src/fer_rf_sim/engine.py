"""Compatibility exports for the former monolithic engine."""

from .physics import E_F, a, F_SWITCH, F_BLEND
from .physics import _cardano_two_smallest, image_potential, airy_all, transmission, transmission_wkb, transmission_airy, fit_wkb_prefactor
from .quadrature import D_avg, current, current_batch, rf_averaged_current
from .lut_build import _upscaled_axis, build_lut, list_existing_luts, interp_lut
from .simulation import direct_rf_current_grid, simulate
from .simulate_cli import _merge_yaml_into_args, cli
