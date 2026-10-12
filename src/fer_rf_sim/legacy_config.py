"""Compatibility configuration for legacy real-data analysis scripts.

The real-data files are external to this repository. Set FER_REAL_DATA_CSV to
an absolute CSV path or folder, or configure real_data.csv_path in fer_config.yaml.
"""

import os
from pathlib import Path

import yaml


_DEFAULT_CONFIG = Path(os.environ.get("FER_CONFIG", "fer_config.yaml")).resolve()
_REPO_ROOT = _DEFAULT_CONFIG.parent


def _load_config():
    if not _DEFAULT_CONFIG.is_file():
        return {}
    with _DEFAULT_CONFIG.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


_config = _load_config().get("real_data", {})
_allowed = {"csv_path", "raw_csv_path", "fitted_peaks_csv_path", "setpoint", "peak_number",
            "baseline_files", "rf_mapping"}
if not isinstance(_config, dict) or set(_config) - _allowed:
    raise ValueError("Unknown or malformed real_data configuration settings")
_configured_path = _config.get(
    "raw_csv_path",
    _config.get("csv_path", "data/raw/z_dependent_fer_shift_csv"),
)
_env_path = os.environ.get("FER_REAL_DATA_CSV")

REAL_DATA_CSV_PATH = str(
    Path(_env_path) if _env_path else _REPO_ROOT / _configured_path
)
REAL_DATA_SETPOINT = _config.get("setpoint", 1000)
REAL_DATA_PEAK_NUMBER = _config.get("peak_number", 1)
if isinstance(REAL_DATA_PEAK_NUMBER, bool) or not isinstance(REAL_DATA_PEAK_NUMBER, int) or REAL_DATA_PEAK_NUMBER < 1:
    raise ValueError("real_data.peak_number must be a positive 1-based integer")
_fitted_path = os.environ.get("FER_FITTED_PEAKS_CSV") or _config.get("fitted_peaks_csv_path")
REAL_DATA_FITTED_PEAKS_PATH = str(_REPO_ROOT / _fitted_path) if _fitted_path else None
REAL_DATA_BASELINE_FILES = _config.get("baseline_files")
REAL_DATA_RF_MAPPING = _config.get("rf_mapping", {})
