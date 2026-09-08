"""Compatibility configuration for legacy real-data analysis scripts.

The real-data files are external to this repository. Set FER_REAL_DATA_CSV to
an absolute CSV path or folder, or configure real_data.csv_path in fer_config.yaml.
"""

import os
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _REPO_ROOT / "fer_config.yaml"


def _load_config():
    if not _DEFAULT_CONFIG.is_file():
        return {}
    with _DEFAULT_CONFIG.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


_config = _load_config().get("real_data", {})
_configured_path = _config.get(
    "csv_path",
    "data/raw/z_dependent_fer_shift_csv",
)
_env_path = os.environ.get("FER_REAL_DATA_CSV")

REAL_DATA_CSV_PATH = str(
    Path(_env_path) if _env_path else _REPO_ROOT / _configured_path
)
REAL_DATA_SETPOINT = _config.get("setpoint", 1000)
REAL_DATA_PEAK_NUMBER = _config.get("peak_number", 1)
