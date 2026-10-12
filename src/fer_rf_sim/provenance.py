"""Versioned numerical identities and atomic HDF5 publication."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np


SCHEMA_VERSION = 2
MODEL_VERSION = "trapezoidal-scaled-airy-3"
CALIBRATION_VERSION = "explicit-branches-1"
CALIBRATION_SETTINGS = {"n_D": 100, "n_E": 200, "E_min": 0.01, "z_switch": 8.0}


def canonical_json(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda value: str(value) if isinstance(value, Path) else value.item())


def numerical_identity(values):
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def lut_spec(energy, voltage, heights, phi_tip, phi_samp, fudge, fermi, switch, blend):
    return {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
            "calibration_version": CALIBRATION_VERSION, "calibration_settings": CALIBRATION_SETTINGS,
            "phi_tip": phi_tip, "phi_samp": phi_samp, "fudge_factor": fudge,
            "E_F": fermi, "F_switch": switch, "F_blend": blend,
            "axes": {name: np.asarray(axis, dtype=np.float64).tolist()
                     for name, axis in (("E", energy), ("V", voltage), ("z", heights))},
            "axis_order": ["z", "V", "E"], "dtype": "float64"}


@contextmanager
def atomic_h5(path):
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with h5py.File(temporary, "w") as file:
            yield file
            file.flush()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)