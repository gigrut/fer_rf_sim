# FER Simulation Workflow

This project simulates Field Emission Resonance (FER) tunneling with RF modulation. The workflow has been refactored for easier use and faster feedback.

## Quick Start

### 1. Run Complete Workflow
```bash
# Run everything with default settings (recommended)
python run_workflow.py

# Quick test with smaller grids
python run_workflow.py --quick-test

# Skip LUT for more accurate (but slower) calculation
python run_workflow.py --no-lut
```

### 2. Step-by-Step Workflow

#### Step 1: Run Simulation
```bash
# Basic simulation with LUT (fast)
python FER_constant_current_simulation.py --use-lut

# Quick test with smaller grids
python FER_constant_current_simulation.py --use-lut --n_E 50 --n_V 20 --n_Z 5 --n_A 3
```

#### Step 2: Quick Validation
```bash
# Run all quick analysis plots
python quick_check.py

# Specific plots
python quick_check.py --plot-type current
python quick_check.py --plot-type dlog
python quick_check.py --plot-type profiles
```

#### Step 3: Constant Current Analysis
```bash
# Analyze constant current (auto-detects reasonable current level)
python constant_current_analysis.py --I-target 1e-9

# Compare multiple current levels
python constant_current_analysis.py --multiple 1e-10 1e-9 1e-8
```

#### Step 4: Advanced Plotting
```bash
# Interactive 3D plots
python FER_plotter.py

# Advanced post-processing (may need fixes)
python FER_sim_post_processing.py
```

## Script Descriptions

### Core Scripts

- **`FER_constant_current_simulation.py`**: Main simulation engine
  - Calculates transmission coefficients for different barrier geometries
  - Computes tunneling current by integrating over electron energy
  - Optional LUT for speed optimization

- **`quick_check.py`**: Fast validation and basic plots
  - Current vs (Z, V) heatmaps
  - d(log(I))/dV vs (Z, V) heatmaps  
  - Current profiles vs voltage
  - Fast feedback on simulation results

- **`constant_current_analysis.py`**: Post-processing for constant current
  - Finds tip height Z(V) that gives constant current
  - Calculates dI/dV vs V at constant current
  - Supports multiple current levels

- **`run_workflow.py`**: Complete workflow automation
  - Runs all steps in sequence
  - Configurable for different use cases
  - Error handling and progress reporting

### Advanced Scripts

- **`FER_plotter.py`**: Interactive 3D plotting and transmission analysis
- **`FER_sim_post_processing.py`**: Advanced post-processing (needs fixes)

## Key Features

### 1. Transmission Coefficient Calculation
- Two-regime transmission: Simmons-WKB for low fields, Airy-based for high fields
- Handles different tip/sample work functions
- Vectorized for efficiency

### 2. Current Calculation
- Integrates transmission over electron energy
- RF-averaged current calculation
- Supports constant current analysis

### 3. Constant Current Analysis
- Finds Z(V) that gives constant current I_target
- Calculates dI/dV vs V at constant current
- Essential for experimental comparison

### 4. Fast Feedback
- Quick validation plots for immediate feedback
- LUT system for speed optimization
- Modular design for easy testing

## Usage Examples

### Quick Test Run
```bash
# Run quick test with small grids
python run_workflow.py --quick-test

# Check results immediately
python quick_check.py
```

### Production Run
```bash
# Full simulation with LUT
python run_workflow.py

# Analyze specific current level
python constant_current_analysis.py --I-target 1e-9 --a-val 2.0
```

### Custom Analysis
```bash
# Skip simulation, use existing results
python run_workflow.py --skip-simulation --steps quick-check constant-current

# Multiple current levels
python constant_current_analysis.py --multiple 1e-10 1e-9 1e-8 1e-7
```

## Configuration

### Simulation Parameters
- `--n_E`: Energy grid points (default: 100)
- `--n_V`: Voltage grid points (default: 100)  
- `--n_Z`: Tip height grid points (default: 10)
- `--n_A`: RF amplitude grid points (default: 10)
- `--use-lut`: Use lookup table for speed
- `--phi_tip`, `--phi_samp`: Work functions

### Quick Check Options
- `--plot-type`: Type of plot (current, dlog, profiles, all)
- `--a-val`: RF amplitude to analyze

### Constant Current Options
- `--I-target`: Target current level
- `--a-val`: RF amplitude to analyze
- `--multiple`: Multiple current levels

## Output Files

- `fer_output/current.h5`: Main simulation results
- `fer_output/transmission_lut_*.h5`: Lookup tables (if using LUT)

## Troubleshooting

### Common Issues

1. **"No module named pandas"**
   ```bash
   pip install pandas
   ```

2. **Simulation takes too long**
   ```bash
   # Use LUT and smaller grids
   python run_workflow.py --quick-test
   ```

3. **Import errors in post-processing**
   - The post-processing scripts may need fixes for function imports
   - Use `constant_current_analysis.py` instead

4. **Virtual environment issues**
   ```bash
   # Create fresh environment
   python -m venv .venv
   .venv\Scripts\activate.bat  # Windows
   pip install -r requirements.txt
   ```

### Performance Tips

- Use `--use-lut` for faster computation
- Start with `--quick-test` for parameter exploration
- Use smaller grids for initial testing
- The LUT system caches results for reuse

## Next Steps

1. Run a quick test to verify everything works
2. Examine the plots to understand the physics
3. Adjust parameters based on your experimental setup
4. Run full simulation for production results
5. Use constant current analysis for experimental comparison 