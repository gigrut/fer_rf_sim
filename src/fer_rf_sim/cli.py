"""Headless command dispatcher. Optional interfaces are imported only on use."""

import argparse
import importlib
import sys


COMMANDS = {"simulate": "simulate_cli", "check": "check", "workflow": "workflow",
            "constant-current": "constant_current", "transmission": "transmission_cli", "lut": "lut_cli"}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="FER simulation and analysis")
    parser.add_argument("command", choices=[*COMMANDS, "gui"])
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(arguments[:1]).command
    try:
        if command == "gui":
            from .gui import run_gui
            return run_gui()
        module = importlib.import_module(f"fer_rf_sim.{COMMANDS[command]}")
        if command == "simulate":
            result = module.cli(arguments[1:])
            if result is not None:
                print(result)
            return 0
        return module.main(arguments[1:])
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1