"""Reusable peak detection for measured and simulated spectra."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True)
class PeakResult:
    """Detected peak measurements in the same units as the input spectrum."""

    index: np.ndarray
    position: np.ndarray
    height: np.ndarray
    prominence: np.ndarray
    width: np.ndarray
    left_position: np.ndarray
    right_position: np.ndarray


def detect_peaks(
    x,
    y,
    *,
    prominence=None,
    distance=None,
    height=None,
    polarity="positive",
    width_reference=0.5,
) -> PeakResult:
    """Detect peaks and estimate widths from one finite, ordered spectrum."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be one-dimensional arrays with matching shapes")
    if x.size < 3 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must contain at least three finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be 'positive' or 'negative'")
    if not 0 < width_reference < 1:
        raise ValueError("width_reference must be between zero and one")

    signal = y if polarity == "positive" else -y
    indices, properties = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
        height=height,
    )
    widths, _, left_indices, right_indices = peak_widths(
        signal, indices, rel_height=width_reference,
    )
    positions = x[indices]
    left_positions = np.interp(left_indices, np.arange(x.size), x)
    right_positions = np.interp(right_indices, np.arange(x.size), x)

    return PeakResult(
        index=indices,
        position=positions,
        height=y[indices],
        prominence=properties.get("prominences", np.empty(0)),
        width=right_positions - left_positions,
        left_position=left_positions,
        right_position=right_positions,
    )
