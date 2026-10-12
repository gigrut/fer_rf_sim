"""RF amplitudes are volts peak; power conversions require an impedance."""

import numpy as np


def amplitude_to_peak(value, *, unit="V", kind="peak"):
    factors = {"V": 1., "mV": 1e-3}
    conventions = {"peak": 1., "rms": np.sqrt(2.), "peak-to-peak": 0.5}
    if unit not in factors or kind not in conventions:
        raise ValueError("Use V or mV and peak, rms, or peak-to-peak amplitudes")
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("RF amplitudes must be finite and non-negative")
    return values * factors[unit] * conventions[kind]


def dbm_to_peak_volts(power_dbm, impedance=50.):
    power = np.asarray(power_dbm, dtype=float)
    if not np.isfinite(impedance) or impedance <= 0 or not np.isfinite(power).all():
        raise ValueError("Power must be finite and impedance must be positive")
    result = np.sqrt(2 * impedance * 10. ** ((power - 30.) / 10.))
    if not np.isfinite(result).all():
        raise ValueError("Converted RF amplitude is not finite")
    return result


def dbm_to_mw(power_dbm):
    return 10. ** (np.asarray(power_dbm, dtype=float) / 10.)


def dbm_to_mv(power_dbm):
    """Legacy helper returns millivolts peak into 50 ohms."""
    return dbm_to_peak_volts(power_dbm) * 1000