"""Import-safe repository for simulation HDF5 artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def validate_axis(values, name, minimum=1):
    axis = np.asarray(values, dtype=float)
    if (axis.ndim != 1 or axis.size < minimum or not np.all(np.isfinite(axis))
            or np.any(np.diff(axis) <= 0)):
        raise ValueError(f"{name} must be finite, strictly increasing, and contain at least {minimum} points")
    return axis


@dataclass(frozen=True)
class SimulationData:
    """Simulation arrays and metadata loaded from one HDF5 artifact."""

    path: Path
    current: np.ndarray
    z: np.ndarray
    voltage: np.ndarray
    rf_amplitude: np.ndarray
    metadata: dict[str, Any]
    energy: np.ndarray | None = None


class SimulationRepository:
    """Read simulation artifacts without performing work at import time."""

    def load(self, path: str | Path) -> SimulationData:
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(f"Simulation file not found: {artifact}")

        with h5py.File(artifact, "r") as file:
            required = ("I", "z", "V", "A_rf")
            missing = [name for name in required if name not in file]
            if missing:
                names = ", ".join(missing)
                raise ValueError(f"Simulation file is missing datasets: {names}")

            current = np.asarray(file["I"][...])
            z = np.asarray(file["z"][...])
            voltage = np.asarray(file["V"][...])
            rf_amplitude = np.asarray(file["A_rf"][...])
            energy = validate_axis(file["E"][...], "E") if "E" in file else None
            metadata = {key: value for key, value in file.attrs.items()}

        self._validate_shapes(current, z, voltage, rf_amplitude)
        metadata.setdefault("provenance", "versioned" if "model_version" in metadata else "unknown")
        metadata.setdefault("current_units", "unknown")
        for axis, lower, upper, count in ((z, "z_min", "z_max", "n_Z"),
                                         (voltage, "v_min", "v_max", "n_V"),
                                         (rf_amplitude, "A_min", "A_max", "n_A")):
            metadata.setdefault(lower, float(axis[0]))
            metadata.setdefault(upper, float(axis[-1]))
            metadata.setdefault(count, axis.size)
        return SimulationData(
            path=artifact,
            current=current,
            z=z,
            voltage=voltage,
            rf_amplitude=rf_amplitude,
            metadata=metadata,
            energy=energy,
        )

    @staticmethod
    def _validate_shapes(current, z, voltage, rf_amplitude) -> None:
        if current.ndim != 3:
            raise ValueError("Simulation current dataset must have shape (z, V, A)")
        expected = (z.size, voltage.size, rf_amplitude.size)
        if current.shape != expected:
            raise ValueError(
                f"Current shape {current.shape} does not match axes {expected}"
            )
        if any(axis.ndim != 1 for axis in (z, voltage, rf_amplitude)):
            raise ValueError("Simulation axes must be one-dimensional")
        for name, axis in (("z", z), ("V", voltage), ("A_rf", rf_amplitude)):
            validate_axis(axis, name)


def load_simulation(path: str | Path) -> SimulationData:
    """Load one simulation artifact using the default repository."""
    return SimulationRepository().load(path)
