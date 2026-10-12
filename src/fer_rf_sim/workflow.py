#!/usr/bin/env python3
"""
FER Simulation Workflow Script
Guides users through the complete simulation and analysis process.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import numpy as np

from fer_rf_sim.hdf5 import find_latest_simulation_file
from fer_rf_sim.repository import load_simulation
from fer_rf_sim.service import SimulationConfig, SimulationRunner
from fer_rf_sim.config import add_simulation_arguments, load_config
from dataclasses import fields

def run_simulation(use_lut=None, quick_test=False, config=None):
    """Run the main simulation."""
    print("=" * 60)
    print("STEP 1: Running FER Simulation")
    print("=" * 60)
    
    config = config or load_config()
    if use_lut is not None:
        config.use_lut = use_lut
    if quick_test:
        config.n_E, config.n_V, config.n_Z, config.n_A = 50, 20, 5, 3
    return SimulationRunner().run(config).resolve()

def _artifact(h5_path=None, output_dir="fer_output"):
    path = Path(h5_path) if h5_path is not None else find_latest_simulation_file(output_dir)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _run_analysis(script, h5_path, *arguments):
    module = {"quick_check.py": "check", "constant_current_analysis.py": "constant_current"}[script]
    command = [sys.executable, "-m", f"fer_rf_sim.{module}",
               "--h5-path", str(h5_path), *arguments]
    subprocess.run(command, check=True)


def run_quick_check(h5_path=None, no_show=False):
    """Run quick validation plots."""
    print("\n" + "=" * 60)
    print("STEP 2: Quick Validation")
    print("=" * 60)

    try:
        h5_path = _artifact(h5_path)
    except FileNotFoundError:
        print("Simulation output not found. Please run simulation first.")
        return False

    print("Running quick analysis plots...")
    try:
        _run_analysis("quick_check.py", h5_path, *(["--no-show"] if no_show else []))
        print("Quick validation completed.")
        return True
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Quick validation failed: {e}", file=sys.stderr)
        return False

def run_constant_current_analysis(I_target=None, h5_path=None, no_show=False):
    """Run constant current analysis."""
    print("\n" + "=" * 60)
    print("STEP 3: Constant Current Analysis")
    print("=" * 60)

    try:
        h5_path = _artifact(h5_path)
    except FileNotFoundError:
        print("Simulation output not found. Please run simulation first.")
        return False

    try:
        if I_target is None:
            current = load_simulation(h5_path).current
            positive = current[np.isfinite(current) & (current > 0)]
            if not positive.size:
                raise ValueError("No finite positive currents available for a setpoint")
            I_target = float(np.median(positive))
        if not np.isfinite(I_target):
            raise ValueError("Target current must be finite")
        _run_analysis("constant_current_analysis.py", h5_path, "--I-target", str(I_target),
                  *(["--no-show"] if no_show else []))
        print("Constant current analysis completed.")
        return True
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"Constant-current analysis failed: {e}", file=sys.stderr)
        return False

def run_advanced_plots():
    """Run advanced plotting options."""
    print("\n" + "=" * 60)
    print("STEP 4: Advanced Plotting")
    print("=" * 60)
    
    print("Available advanced plotting options:")
    print("1. FER_plotter.py - Interactive 3D plots and transmission analysis")
    print("2. FER_sim_post_processing.py - Shared constant-current post-processing")
    print()
    print("You can run these manually:")
    print("  fer-sim check --h5-path PATH")
    print("  fer-sim transmission --lut-path PATH --help")
    
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description="FER Simulation Workflow")
    add_simulation_arguments(parser)
    parser.add_argument("--quick-test", action="store_true",
                       help="Run with reduced grid sizes for quick testing")
    parser.add_argument("--skip-simulation", action="store_true",
                       help="Skip simulation step (use existing results)")
    parser.add_argument("--skip-quick-check", action="store_true",
                       help="Skip quick validation step")
    parser.add_argument("--skip-constant-current", action="store_true",
                       help="Skip constant current analysis")
    parser.add_argument("--I-target", type=float,
                       help="Target current level for constant current analysis")
    parser.add_argument("--h5-path", type=Path, help="Existing artifact when skipping simulation")
    parser.add_argument("--no-show", action="store_true", help="Validate analysis without opening a browser")
    parser.add_argument("--steps", nargs='+', 
                       choices=['simulation', 'quick-check', 'constant-current', 'advanced'],
                       help="Run only specific steps")
    
    args = parser.parse_args(argv)
    
    print("FER Simulation Workflow")
    print("=" * 60)
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = ['simulation', 'quick-check', 'constant-current', 'advanced']
    
    success = True
    
    # Step 1: Simulation
    try:
        names = {field.name for field in fields(SimulationConfig)}
        overrides = {key: value for key, value in vars(args).items() if key in names}
        if args.quick_test:
            overrides = {"n_E": 50, "n_V": 20, "n_Z": 5, "n_A": 3, **overrides}
        config = load_config(getattr(args, 'config', None), overrides)
        if config.list_luts:
            from .lut_cli import list_luts
            list_luts(config.out)
            return 0
        if 'simulation' in steps_to_run and not args.skip_simulation:
            h5_path = run_simulation(
                config=config,
            )
        elif any(step in steps_to_run and not getattr(args, "skip_" + step.replace("-", "_"))
                 for step in ("quick-check", "constant-current")):
            h5_path = _artifact(args.h5_path, config.out)
        else:
            h5_path = None
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Workflow failed: {error}", file=sys.stderr)
        return 1
    
    # Step 2: Quick Check
    if 'quick-check' in steps_to_run and not args.skip_quick_check:
        success = run_quick_check(h5_path, no_show=args.no_show) and success
        if not success:
            print("\nQuick check failed; continuing with a nonzero final status.")
    
    # Step 3: Constant Current Analysis
    if 'constant-current' in steps_to_run and not args.skip_constant_current:
        success = run_constant_current_analysis(args.I_target, h5_path, no_show=args.no_show) and success
        if not success:
            print("\nAnalysis failed; continuing with a nonzero final status.")
    
    # Step 4: Advanced Plotting
    if 'advanced' in steps_to_run:
        run_advanced_plots()
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Examine the generated plots")
    print("2. Run advanced plotting scripts if needed")
    print("3. Modify parameters and re-run if necessary")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 