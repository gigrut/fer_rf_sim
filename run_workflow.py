#!/usr/bin/env python3
"""
FER Simulation Workflow Script
Guides users through the complete simulation and analysis process.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_simulation(use_lut=True, quick_test=False):
    """Run the main simulation."""
    print("=" * 60)
    print("STEP 1: Running FER Simulation")
    print("=" * 60)
    
    cmd = ["python", "FER_constant_current_simulation.py"]
    
    if quick_test:
        # Use smaller grids for quick testing
        cmd.extend([
            "--n_E", "50",
            "--n_V", "20", 
            "--n_Z", "5",
            "--n_A", "3"
        ])
        print("Running QUICK TEST with reduced grid sizes...")
    else:
        # Use default grid sizes
        cmd.extend([
            "--n_E", "100",
            "--n_V", "100",
            "--n_Z", "10", 
            "--n_A", "10"
        ])
    
    if use_lut:
        cmd.append("--use-lut")
        print("Using LUT for faster computation...")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Simulation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed with error code {e.returncode}")
        return False

def run_quick_check():
    """Run quick validation plots."""
    print("\n" + "=" * 60)
    print("STEP 2: Quick Validation")
    print("=" * 60)
    
    if not Path("fer_output/current.h5").exists():
        print("✗ Simulation output not found. Please run simulation first.")
        return False
    
    print("Running quick analysis plots...")
    cmd = ["python", "quick_check.py"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Quick validation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Quick validation failed with error code {e.returncode}")
        return False

def run_constant_current_analysis(I_target=None):
    """Run constant current analysis."""
    print("\n" + "=" * 60)
    print("STEP 3: Constant Current Analysis")
    print("=" * 60)
    
    if not Path("fer_output/current.h5").exists():
        print("✗ Simulation output not found. Please run simulation first.")
        return False
    
    if I_target is None:
        # Try to estimate a reasonable current level from the data
        import h5py
        with h5py.File("fer_output/current.h5", "r") as f:
            I_cube = f["I"][...]
            I_target = np.median(I_cube[I_cube > 0])  # Median of positive currents
            print(f"Using estimated current level: I = {I_target:.2e}")
    
    print(f"Running constant current analysis for I = {I_target:.2e}...")
    cmd = ["python", "constant_current_analysis.py", "--I-target", str(I_target)]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Constant current analysis completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Constant current analysis failed with error code {e.returncode}")
        return False

def run_advanced_plots():
    """Run advanced plotting options."""
    print("\n" + "=" * 60)
    print("STEP 4: Advanced Plotting")
    print("=" * 60)
    
    print("Available advanced plotting options:")
    print("1. FER_plotter.py - Interactive 3D plots and transmission analysis")
    print("2. FER_sim_post_processing.py - Advanced post-processing (may need fixes)")
    print()
    print("You can run these manually:")
    print("  python FER_plotter.py")
    print("  python FER_sim_post_processing.py")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="FER Simulation Workflow")
    parser.add_argument("--no-lut", action="store_true", 
                       help="Disable LUT usage (slower but more accurate)")
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
    parser.add_argument("--steps", nargs='+', 
                       choices=['simulation', 'quick-check', 'constant-current', 'advanced'],
                       help="Run only specific steps")
    
    args = parser.parse_args()
    
    print("FER Simulation Workflow")
    print("=" * 60)
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = ['simulation', 'quick-check', 'constant-current', 'advanced']
    
    success = True
    
    # Step 1: Simulation
    if 'simulation' in steps_to_run and not args.skip_simulation:
        success = run_simulation(use_lut=not args.no_lut, quick_test=args.quick_test)
        if not success:
            print("\n✗ Workflow stopped due to simulation failure.")
            return 1
    
    # Step 2: Quick Check
    if 'quick-check' in steps_to_run and not args.skip_quick_check:
        success = run_quick_check()
        if not success:
            print("\n⚠ Quick check failed, but continuing...")
    
    # Step 3: Constant Current Analysis
    if 'constant-current' in steps_to_run and not args.skip_constant_current:
        success = run_constant_current_analysis(args.I_target)
        if not success:
            print("\n⚠ Constant current analysis failed, but continuing...")
    
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
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 