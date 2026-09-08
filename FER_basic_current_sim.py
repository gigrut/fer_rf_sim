"""Shim: implementation lives in src/fer_rf_sim/simulation/basic_current_legacy.py"""

from fer_rf_sim.simulation._legacy_loader import load_module_path
from pathlib import Path

_LEGACY_PATH = Path(__file__).resolve().parent / "src" / "fer_rf_sim" / "simulation" / "basic_current.py"
_mod = load_module_path(_LEGACY_PATH)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(_LEGACY_PATH), run_name="__main__")
