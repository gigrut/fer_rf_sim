"""Run with an installed interpreter from any CWD; no pytest dependency required."""

import argparse
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import fer_rf_sim
from fer_rf_sim.config import SimulationConfig
from fer_rf_sim.repository import load_simulation
from fer_rf_sim.service import SimulationRunner
from FER_constant_current_simulation import transmission
from FER_basic_current_sim import current


def command(arguments, expected=0):
    result = subprocess.run([sys.executable, "-m", "fer_rf_sim", *arguments],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-wheel", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    assert version("fer-rf-sim") == fer_rf_sim.__version__
    if args.expect_wheel:
        assert "site-packages" in Path(fer_rf_sim.__file__).parts
    assert callable(transmission) and callable(current)
    with TemporaryDirectory(prefix="fer_installed_smoke_") as directory:
        previous = Path.cwd()
        try:
            os.chdir(directory)
            config = SimulationConfig(n_E=12, n_V=4, n_Z=2, n_A=2, n_cheb=4, threads=1,
                                      v_min=.01, v_max=.04, A_max=.003, z_min=.3, z_max=.5,
                                      phi_samp=4., out=directory)
            runner = SimulationRunner()
            direct = load_simulation(runner.run(config))
            lut = load_simulation(runner.run(replace(config, use_lut=True, upscale_v=4)))
            assert direct.path != lut.path
            np.testing.assert_allclose(lut.current, direct.current, rtol=.02, atol=0)
            command(["simulate", "--help"])
            command(["simulate", "--n_V", "1"], expected=2)
            command(["lut", "--fer-output", directory, "list"])
            command(["lut", "--fer-output", directory, "check", "--reference-lut", lut.metadata["lut_file"]])
            if not args.core_only:
                command(["check", "--h5-path", str(direct.path), "--no-show"])
                command(["constant-current", "--h5-path", str(direct.path), "--multiple", "1e-9", "2e-9", "--no-show"])
                command(["workflow", "--skip-simulation", "--h5-path", str(direct.path), "--no-show"])
                for name in ("check", "workflow", "constant-current", "transmission", "lut"):
                    command([name, "--help"])
                from FER_plotter import plot_T_slice_3d
                assert callable(plot_T_slice_3d)
        finally:
            os.chdir(previous)
    print(f"Installed {'wheel' if args.expect_wheel else 'editable'} {fer_rf_sim.__version__}: imports, commands, direct/LUT and legacy adapters passed")


if __name__ == "__main__":
    main()