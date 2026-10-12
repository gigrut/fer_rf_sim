"""Load and normalize the z-dependent real FER CSV data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from .units import amplitude_to_peak, dbm_to_peak_volts


LEGACY_RF_AMPLITUDES = (0.0, 0.56, 1.0, 1.33, 1.78)


@dataclass(frozen=True)
class RealSpectrum:
    path: Path
    data: pd.DataFrame
    rf_amplitude: float | None
    rf_source: str = "unknown"

    def require_rf(self) -> float:
        if self.rf_amplitude is None:
            raise ValueError(f"RF metadata is required for {self.path.name}")
        return self.rf_amplitude


class RealDataRepository:
    """Load one CSV or a folder of instrument CSV files."""

    def load(self, source: str | Path, *, rf_mapping=None, impedance=50.) -> list[RealSpectrum]:
        path = Path(source)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.glob("*.csv"))
        else:
            raise FileNotFoundError(f"Real-data source not found: {path}")
        if not files:
            raise FileNotFoundError(f"No CSV files found in real-data source: {path}")

        spectra = []
        for file_path in files:
            data = pd.read_csv(file_path)
            required = {"bias (mV)", "lockin-x (mV)"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(
                    f"{file_path.name} is missing columns: {', '.join(sorted(missing))}"
                )
            for column in required:
                data[column] = pd.to_numeric(data[column], errors="raise")
            data["bias (V)"] = data["bias (mV)"] * 1e-3
            data["lockin-x (V)"] = data["lockin-x (mV)"] * 1e-3
            if "curr (nA)" in data:
                data["current (A)"] = pd.to_numeric(data["curr (nA)"], errors="raise") * 1e-9
            amplitude, origin = self._rf_metadata(file_path, data, rf_mapping or {}, impedance)
            spectra.append(
                RealSpectrum(
                    path=file_path,
                    data=data,
                    rf_amplitude=amplitude,
                    rf_source=origin,
                )
            )
        return spectra

    @staticmethod
    def _rf_metadata(path, data, mapping, impedance):
        for key in (str(path.resolve()), str(path), path.name):
            if key in mapping:
                value = mapping[key]
                return (None if value is None else float(amplitude_to_peak(value))), "mapping"
        columns = (("RF Amplitude (V)", "peak", "V"), ("RF (V)", "peak", "V"),
                   ("RF RMS (V)", "rms", "V"), ("RF peak-to-peak (V)", "peak-to-peak", "V"))
        for column, kind, unit in columns:
            if column in data:
                values = pd.to_numeric(data[column], errors="raise").dropna().unique()
                if len(values) != 1:
                    raise ValueError(f"{path.name}: RF metadata must have one value per spectrum")
                return float(amplitude_to_peak(values[0], unit=unit, kind=kind)), column
        for column in ("RF Power (dBm)", "RF Power"):
            if column in data:
                values = pd.to_numeric(data[column], errors="raise").dropna().unique()
                if len(values) != 1:
                    raise ValueError(f"{path.name}: RF power must have one value per spectrum")
                return float(dbm_to_peak_volts(values[0], impedance)), column
        return None, "unknown"


def load_real_data(source: str | Path, **options) -> list[RealSpectrum]:
    return RealDataRepository().load(source, **options)
