"""Profile a simulation service call without replacing existing profile outputs."""

import argparse
import cProfile
import importlib.util
from pathlib import Path
import pstats
import subprocess
import sys
from uuid import uuid4

from fer_rf_sim.config import add_simulation_arguments, load_config
from fer_rf_sim.service import SimulationRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--snakeviz", action="store_true")
    values = vars(parser.parse_args(argv))
    directory = values.pop("profile_dir")
    snakeviz = values.pop("snakeviz")
    config = load_config(values.pop("config", None), values)
    directory = directory or Path(config.out) / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / f"simulation_{uuid4().hex}.prof"
    profiler = cProfile.Profile()
    artifact = profiler.runcall(SimulationRunner().run, config)
    profiler.dump_stats(str(profile_path))
    pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(30)
    print(f"Artifact: {artifact}\nProfile: {profile_path}")
    if snakeviz:
        if importlib.util.find_spec("snakeviz") is None:
            parser.error("SnakeViz is optional; install snakeviz to request this viewer")
        subprocess.run([sys.executable, "-m", "snakeviz", str(profile_path)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())