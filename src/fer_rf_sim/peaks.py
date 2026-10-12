"""Reusable peak detection for measured and simulated spectra."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
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


@dataclass(frozen=True)
class PeakSpectrum:
    label: str
    voltage: np.ndarray
    signal: np.ndarray
    rf_amplitude: float | None = None
    signal_unit: str = "a.u."


@dataclass(frozen=True)
class SpectrumPeaks:
    spectrum: PeakSpectrum
    peaks: PeakResult


@dataclass(frozen=True)
class RFPeak:
    track: int
    result_index: int
    peak_index: int
    rf_amplitude: float
    position: float
    shift: float
    height: float
    prominence: float
    width: float


def export_peak_csv(results, path, *, rf_peaks=None):
    """Write peak measurements with units and optional RF track assignments."""
    import csv

    tracked = {(peak.result_index, peak.peak_index): peak for peak in (rf_peaks or [])}
    fields = ["spectrum", "signal_unit", "rf_peak_v", "peak_index", "position_v", "height",
              "prominence", "width_v", "left_position_v", "right_position_v", "track", "shift_v"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result_index, result in enumerate(results):
            for peak_index, position in enumerate(result.peaks.position):
                observation = tracked.get((result_index, peak_index))
                writer.writerow({
                    "spectrum": result.spectrum.label, "signal_unit": result.spectrum.signal_unit,
                    "rf_peak_v": result.spectrum.rf_amplitude,
                    "peak_index": int(result.peaks.index[peak_index]), "position_v": position,
                    "height": result.peaks.height[peak_index],
                    "prominence": result.peaks.prominence[peak_index], "width_v": result.peaks.width[peak_index],
                    "left_position_v": result.peaks.left_position[peak_index],
                    "right_position_v": result.peaks.right_position[peak_index],
                    "track": observation.track if observation is not None else "",
                    "shift_v": observation.shift if observation is not None else "",
                })


def real_peak_spectra(spectra) -> list[PeakSpectrum]:
    """Normalize measured lock-in spectra to volts, preserving explicit RF metadata."""
    results = []
    for spectrum in spectra:
        data = spectrum.data.sort_values("bias (V)")
        results.append(PeakSpectrum(
            str(spectrum.path), data["bias (V)"].to_numpy(dtype=float),
            data["lockin-x (V)"].to_numpy(dtype=float), spectrum.rf_amplitude, "V",
        ))
    return results


def _finite_gradient(voltage, signal):
    derivative = np.full_like(signal, np.nan, dtype=float)
    indices = np.flatnonzero(np.isfinite(signal))
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if segment.size >= 2:
            derivative[segment] = np.gradient(signal[segment], voltage[segment])
    return derivative


def simulated_peak_spectra(
    data, *, mode="constant-current dI/dV", current_target=None, height_index=0, current_array=None,
) -> list[PeakSpectrum]:
    """Build one bias trace per RF amplitude using the shared simulation analysis."""
    from .analysis import build_current_array, constant_current_slice

    if mode not in {"constant-current dI/dV", "constant-current dZ/dV", "fixed-height dI/dV"}:
        raise ValueError(f"Unknown simulated peak mode: {mode}")
    if current_array is None:
        current_array = build_current_array(data.current, data.voltage)
    if mode == "fixed-height dI/dV":
        if not 0 <= height_index < data.z.size:
            raise ValueError("Height index is outside the simulation grid")
        signals = current_array[height_index, :, :, 1]
        context = f"z={data.z[height_index]:.6g} nm"
    else:
        if current_target is None:
            raise ValueError("A current target is required for constant-current peak extraction")
        contour = constant_current_slice(current_target, data.z, current_array)
        signals = contour[..., 1]
        if mode == "constant-current dZ/dV":
            signals = np.column_stack([
                _finite_gradient(data.voltage, contour[:, index, 0])
                for index in range(data.rf_amplitude.size)
            ])
        context = f"I={current_target:.6g} a.u."
    unit = "nm/V" if mode == "constant-current dZ/dV" else "a.u./V"
    return [
        PeakSpectrum(f"{data.path.name}: {mode}, {context}, RF={amplitude:.6g} V peak",
                     data.voltage, signals[:, index], float(amplitude), unit)
        for index, amplitude in enumerate(data.rf_amplitude)
    ]


def track_rf_peaks(results, *, max_shift=0.25) -> list[RFPeak]:
    """Match nearby peaks one-to-one in increasing RF order; retain replicate measurements."""
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError("Maximum peak shift must be finite and positive")
    known = [index for index, result in enumerate(results) if result.spectrum.rf_amplitude is not None]
    amplitudes = sorted({results[index].spectrum.rf_amplitude for index in known})
    if len(amplitudes) < 2:
        raise ValueError("RF dependence requires at least two distinct known RF amplitudes")
    references = {}
    last_positions = {}
    observations = []
    for amplitude in amplitudes:
        positions_at_amplitude = {}
        for result_index in known:
            result = results[result_index]
            if result.spectrum.rf_amplitude != amplitude:
                continue
            peaks = result.peaks
            tracks = list(last_positions)
            assignments = {}
            if tracks and peaks.position.size:
                distances = np.abs(
                    np.array([last_positions[track] for track in tracks])[:, None] - peaks.position
                )
                penalty = max_shift * (len(tracks) + peaks.position.size + 1)
                costs = np.where(distances <= max_shift, distances, penalty)
                rows, columns = linear_sum_assignment(costs)
                assignments = {
                    int(column): tracks[row] for row, column in zip(rows, columns)
                    if distances[row, column] <= max_shift
                }
            for peak_index, position in enumerate(peaks.position):
                track = assignments.get(peak_index)
                if track is None:
                    track = len(references) + 1
                    references[track] = float(position)
                    last_positions[track] = float(position)
                positions_at_amplitude.setdefault(track, []).append(float(position))
                observations.append(RFPeak(
                    track, result_index, peak_index, float(amplitude), float(position),
                    float(position - references[track]), float(peaks.height[peak_index]),
                    float(peaks.prominence[peak_index]), float(peaks.width[peak_index]),
                ))
        for track, positions in positions_at_amplitude.items():
            last_positions[track] = float(np.median(positions))
    return observations


def extract_spectra(spectra, **options) -> list[SpectrumPeaks]:
    """Extract each finite segment without interpolating across missing data."""
    results = []
    for spectrum in spectra:
        voltage = np.asarray(spectrum.voltage, dtype=float)
        signal = np.asarray(spectrum.signal, dtype=float)
        if voltage.ndim != 1 or voltage.shape != signal.shape:
            raise ValueError(f"{spectrum.label}: voltage and signal must be matching vectors")
        if np.any(~np.isfinite(voltage)) or np.any(np.diff(voltage) <= 0):
            raise ValueError(f"{spectrum.label}: voltage must be finite and strictly increasing")
        if spectrum.rf_amplitude is not None and (
            not np.isfinite(spectrum.rf_amplitude) or spectrum.rf_amplitude < 0
        ):
            raise ValueError(f"{spectrum.label}: RF amplitude must be finite and nonnegative")
        finite_indices = np.flatnonzero(np.isfinite(signal))
        segments = np.split(finite_indices, np.flatnonzero(np.diff(finite_indices) > 1) + 1)
        parts = []
        for segment in segments:
            if segment.size < 3:
                continue
            peaks = detect_peaks(voltage[segment], signal[segment], **options)
            parts.append(PeakResult(
                index=segment[peaks.index], position=peaks.position, height=peaks.height,
                prominence=peaks.prominence, width=peaks.width,
                left_position=peaks.left_position, right_position=peaks.right_position,
            ))
        combined = {}
        for name in PeakResult.__dataclass_fields__:
            dtype = int if name == "index" else float
            combined[name] = (
                np.concatenate([getattr(part, name) for part in parts])
                if parts else np.array([], dtype=dtype)
            )
        results.append(SpectrumPeaks(spectrum, PeakResult(**combined)))
    return results


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
        prominence=(None, None) if prominence is None else prominence,
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
