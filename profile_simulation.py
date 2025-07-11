import cProfile
import pstats
import FER_constant_current_simulation
import sys
import subprocess

if __name__ == "__main__":
    # Use small grid sizes for a quick profiling run
    sys.argv = [
        "FER_constant_current_simulation.py",
        "--n_E", "50",
        "--n_V", "50",
        "--n_Z", "50",
        "--n_A", "10"
    ]
    cProfile.run('FER_constant_current_simulation.cli()', 'profile_output.prof')

    # Print a summary of the top bottlenecks
    p = pstats.Stats('profile_output.prof')
    p.strip_dirs().sort_stats('cumtime').print_stats(30)

    # Try to launch SnakeViz for visualization
    try:
        print("\nLaunching SnakeViz for interactive profile visualization...")
        subprocess.run([sys.executable, '-m', 'snakeviz', 'profile_output.prof'])
    except FileNotFoundError:
        print("\nSnakeViz is not installed. To install, run: pip install snakeviz")
        print("Then you can run: python -m snakeviz profile_output.prof") 