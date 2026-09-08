"""Load and normalize the z-dependent real FER CSV data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LEGACY_RF_AMPLITUDES = (0.0, 0.56, 1.0, 1.33, 1.78)


@dataclass(frozen=True)
class RealSpectrum:
    path: Path
    data: pd.DataFrame
    rf_amplitude: float


class RealDataRepository:
    """Load one CSV or a folder of instrument CSV files."""

    def load(self, source: str | Path) -> list[RealSpectrum]:
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
        for index, file_path in enumerate(files):
            data = pd.read_csv(file_path)
            required = {"bias (mV)", "lockin-x (mV)"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(
                    f"{file_path.name} is missing columns: {', '.join(sorted(missing))}"
                )
            spectra.append(
                RealSpectrum(
                    path=file_path,
                    data=data,
                    rf_amplitude=LEGACY_RF_AMPLITUDES[index % len(LEGACY_RF_AMPLITUDES)],
                )
            )
        return spectra


def load_real_data(source: str | Path) -> list[RealSpectrum]:
    return RealDataRepository().load(source)
