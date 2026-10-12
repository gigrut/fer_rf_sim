"""Shared RF broadening with explicit interpolation and edge conventions."""

import numpy as np
from scipy.interpolate import interp1d

from .quadrature import CGQ


def broaden_signal(bias, signal, amplitude, count=100, *, output_bias=None,
                   interpolation="linear", edge_factor=1.):
    bias = np.asarray(bias, dtype=float)
    signal = np.asarray(signal, dtype=float)
    amplitude = float(np.asarray(amplitude).item())
    if (bias.ndim != 1 or signal.shape != bias.shape or bias.size < 2
            or not np.isfinite(bias).all() or not np.isfinite(signal).all()):
        raise ValueError("Broadening requires matching finite 1D bias and signal arrays")
    if not np.isfinite(amplitude) or amplitude < 0 or not np.isfinite(edge_factor) or edge_factor < 1:
        raise ValueError("Amplitude must be non-negative and edge_factor must be at least one")
    unique_bias, indices = np.unique(bias, return_index=True)
    if unique_bias.size < 2:
        raise ValueError("Broadening requires at least two distinct bias values")
    if interpolation not in ("linear", "cubic"):
        raise ValueError("interpolation must be linear or cubic")
    kind = "linear" if unique_bias.size < 4 else interpolation
    interpolator = interp1d(unique_bias, signal[indices], kind=kind, bounds_error=False, fill_value=np.nan)
    targets = bias if output_bias is None else np.asarray(output_bias, dtype=float)
    if targets.ndim != 1 or not np.isfinite(targets).all():
        raise ValueError("Output bias must be a finite 1D axis")
    result = CGQ(interpolator, targets, amplitude, count)
    edge = edge_factor * amplitude
    valid = (targets >= unique_bias[0] + edge) & (targets <= unique_bias[-1] - edge)
    return np.where(valid, result, np.nan)


def broadenCGQ(spec, x_axis_name, y_axis_name, V_RF, n=100):
    """Legacy dataframe convention: cubic interpolation and a 1.5-amplitude edge mask."""
    return broaden_signal(spec[x_axis_name], spec[y_axis_name], V_RF, n,
                          interpolation="cubic", edge_factor=1.5)


def broaden_index_spectrum(channel, bias, v_rf, n=100):
    """Peak-index convention: regular bias axis, linear interpolation, full-cycle edge mask."""
    import pandas as pd
    regular_bias = np.linspace(np.min(bias), np.max(bias), len(bias))
    values = broaden_signal(bias, channel, v_rf, n, output_bias=regular_bias)
    return pd.DataFrame({"bias": regular_bias, "broadened channel": values})