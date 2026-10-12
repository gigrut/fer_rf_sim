"""LUT inventory and one strict compatibility contract for every caller."""

import json
from pathlib import Path
import warnings

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fer_rf_sim.provenance import numerical_identity
from fer_rf_sim.repository import validate_axis


def list_lut_paths(directory="fer_output"):
    return sorted(Path(directory).glob("transmission_lut*.h5"),
                  key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)


def lut_matches(file, spec):
    try:
        identity = numerical_identity(spec)
        stored = json.loads(file.attrs["lut_spec_json"])
        if file.attrs["lut_identity"] != identity or numerical_identity(stored) != identity:
            return False
        if file["D"].shape != tuple(len(spec["axes"][name]) for name in ("z", "V", "E")):
            return False
        return all(np.array_equal(file[name][...], spec["axes"][name]) for name in ("z", "V", "E"))
    except (KeyError, ValueError, TypeError):
        return False


def load_lut(path, spec=None):
    with h5py.File(path, "r") as file:
        if spec is not None and not lut_matches(file, spec):
            raise ValueError("LUT numerical identity does not match the requested model and axes")
        if "lut_spec_json" not in file.attrs:
            warnings.warn("Legacy LUT: model and calibration provenance are unknown", UserWarning, stacklevel=2)
        elif spec is None and not lut_matches(file, json.loads(file.attrs["lut_spec_json"])):
            raise ValueError("LUT provenance does not match its datasets")
        axes = tuple(validate_axis(file[name][...], name, minimum=2) for name in ("z", "V", "E"))
        values = file["D"][...]
        if values.shape != tuple(len(axis) for axis in axes) or not np.isfinite(values).all():
            raise ValueError("LUT must be finite with shape (z, V, E)")
        return values, axes, dict(file.attrs)


def lut_interpolator(path, spec=None):
    values, axes, metadata = load_lut(path, spec)
    return RegularGridInterpolator(axes, values, bounds_error=True), axes, metadata