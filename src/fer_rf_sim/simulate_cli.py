import argparse
import time
from .simulation import simulate
from .lut_build import list_existing_luts


def _merge_yaml_into_args(yaml_path: str, parser: argparse.ArgumentParser,
                          args: argparse.Namespace) -> argparse.Namespace:
    """Compatibility helper: args must contain only explicitly supplied options."""
    from fer_rf_sim.config import load_config
    values = {key: value for key, value in vars(args).items() if key != 'config'}
    return argparse.Namespace(**vars(load_config(yaml_path, values).as_namespace()))


def cli(argv=None):
    from fer_rf_sim.config import parse_config
    args = parse_config(argv).as_namespace()

    # ---- Act on --list-luts early -------------------------------------------
    if args.list_luts:
        list_existing_luts(args.out)
        return

    # ---- Launch simulation ---------------------------------------------------
    t0 = time.perf_counter()
    current_file = simulate(args)
    print(f"[done] total runtime {time.perf_counter()-t0:.1f} s")
    return current_file
