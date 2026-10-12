"""Separate raw spectra, aggregate exports, and fitted-peak CSV contracts."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ALIASES = {
    "V": ("v", "bias", "voltage"), "Current": ("current", "i"), "dIdV": ("didv",),
    "Z": ("z",), "Setpoint": ("setpoint", "isetpoint", "currentsetpoint"),
    "RFFreq": ("rffreq", "rfreq", "rffrequency"), "RFPower": ("rfpower", "rfpow", "rfpwr", "power"),
    "RFVoltage": ("rfvoltage", "rfv"), "FileName": ("filename", "file", "fname"),
    "LineGroup": ("linegroup", "group", "spectrumindex", "spectrumid"),
    "dZdV": ("dzdv",), "RelativeZ": ("relativez", "zrelative", "zrel"),
}
AGGREGATE_REQUIRED = ("V", "Setpoint", "RFFreq", "RFPower", "dZdV", "RelativeZ")
FITTED_PEAK_REQUIRED = ("PeakNumber", "RF Power", "Setpoint", "LineGroup", "FileName",
                        "FitPeakCenter", "FitPeakWidth", "FitPeakCenterUncertainty", "FitPeakHeight")


def canonicalize(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def map_columns(columns):
    result = {}
    for name, aliases in ALIASES.items():
        matches = [column for column in columns if canonicalize(column) in aliases]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous aggregate column {name}: {matches}")
        if matches:
            result[name] = matches[0]
    return result


def read_header(path, delimiter=","):
    return list(pd.read_csv(path, sep=delimiter, nrows=0).columns)


def files_to_process(source):
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(file) for file in sorted(path.glob("*.csv"))]
    return []


def try_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def iter_filtered_spectra(source, *, setpoint, rf_frequency, rf_powers, voltage_unit,
                          chunksize=200_000, delimiter=",", frequency_tolerance=1., power_tolerance=0.):
    """Stream aggregate exports, retaining declared source units and grouping metadata."""
    if voltage_unit not in ("V", "mV"):
        raise ValueError("Aggregate voltage_unit must be declared as V or mV")
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be positive")
    files = files_to_process(source)
    if not files:
        raise FileNotFoundError(f"No aggregate CSV files at {source}")
    for path in files:
        mapping = map_columns(read_header(path, delimiter))
        missing = set(AGGREGATE_REQUIRED) - mapping.keys()
        if missing:
            raise ValueError(f"{path}: aggregate schema is missing {sorted(missing)}")
        with pd.read_csv(path, sep=delimiter, usecols=list(mapping.values()), chunksize=chunksize) as reader:
            for chunk in reader:
                chunk = chunk.rename(columns={value: key for key, value in mapping.items()})
                for name in AGGREGATE_REQUIRED:
                    chunk[name] = try_to_numeric(chunk[name])
                mask = ((chunk["Setpoint"] == setpoint) & np.isfinite(chunk["RFFreq"])
                        & (np.abs(chunk["RFFreq"] - rf_frequency) <= frequency_tolerance)
                        & np.any(np.abs(chunk["RFPower"].to_numpy()[:, None]
                                        - np.atleast_1d(rf_powers)) <= power_tolerance, axis=1))
                selected = chunk.loc[mask].copy()
                selected.attrs["units"] = {"V": voltage_unit, "RFFreq": "Hz", "RFPower": "dBm"}
                if not selected.empty:
                    yield selected


def load_fitted_peaks(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fitted-peak CSV must be a file: {path}")
    columns = read_header(path)
    missing = set(FITTED_PEAK_REQUIRED) - set(columns)
    if missing:
        raise ValueError(f"Fitted-peak schema is missing {sorted(missing)}")
    optional = [name for name in ("RFEnabled",) if name in columns]
    frame = pd.read_csv(path, usecols=[*FITTED_PEAK_REQUIRED, *optional])
    for name in FITTED_PEAK_REQUIRED:
        if name != "FileName":
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def explicit_baseline_mask(frame, filename_mapping=None):
    """True means RF-off reference; never infer it from filename sorting."""
    if filename_mapping is not None:
        unknown = set(frame["FileName"]) - set(filename_mapping)
        if unknown or any(not isinstance(value, bool) for value in filename_mapping.values()):
            raise ValueError("Baseline mapping must assign a boolean to every filename")
        return frame["FileName"].map(filename_mapping).astype(bool)
    if "RFEnabled" not in frame:
        raise ValueError("RF-off baselines require RFEnabled metadata or an explicit filename mapping")
    if not frame["RFEnabled"].isin([True, False, 0, 1]).all():
        raise ValueError("RFEnabled must contain boolean or 0/1 values")
    return ~frame["RFEnabled"].astype(bool)