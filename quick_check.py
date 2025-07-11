#!/usr/bin/env python3
"""
Quick validation script for FER simulation results.
Provides fast feedback on current vs (Z,V) and d(log(I))/dV vs (Z,V).
"""

import numpy as np
import h5py
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import argparse
import glob

def find_latest_simulation_file(directory=r"C:\Users\willh\OneDrive\Desktop\FER_Simulation\fer_output", pattern="current_phit*.h5"):
    files = glob.glob(str(Path(directory) / pattern))
    if not files:
        raise FileNotFoundError(f"No simulation files matching {pattern} found in {directory}")
    latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"[INFO] Using simulation file: {latest_file}")
    return latest_file


def load_simulation_data(h5_path=None):
    """Load simulation data from HDF5 file."""
    if h5_path is None:
        h5_path = find_latest_simulation_file()
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"Simulation file not found: {h5_path}")
    
    with h5py.File(h5_path, "r") as f:
        I_cube = f["I"][...]           # current (nZ, nV, nA)
        z_values = f["z"][...]         # tip-height axis
        V_values = f["V"][...]         # bias axis
        A_values = f["A_rf"][...]      # RF amplitude axis
        
    return I_cube, z_values, V_values, A_values

def plot_current_vs_z_v(h5_path=None, a_idx=None, a_val=None):
    """Plot log10(current) vs (Z, V) as a 3D surface for a given RF amplitude."""
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    
    # Select RF amplitude
    if a_idx is None:
        if a_val is None:
            a_idx = 0  # Use first RF amplitude
        else:
            a_idx = np.abs(A_values - a_val).argmin()
    a_val = A_values[a_idx]
    
    # Extract 2D slice
    I_2d = I_cube[:, :, a_idx]  # shape (nZ, nV)
    
    # Take log10 of current, handle zeros and negatives
    with np.errstate(divide='ignore', invalid='ignore'):
        logI_2d = np.log10(np.abs(I_2d))
        logI_2d[np.isneginf(logI_2d)] = np.nan
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=logI_2d,
        x=V_values,
        y=z_values,
        colorbar=dict(title='log10(Current)'),
        colorscale='Cividis',
        showscale=True
    )])
    fig.update_layout(
        title=f'log10(Current) vs Tip Height and Bias Voltage (RF Amplitude = {a_val:.2f} V)',
        scene=dict(
            xaxis_title='Bias Voltage (V)',
            yaxis_title='Tip Height (nm)',
            zaxis_title='log10(Current)'
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
    return fig

def plot_dlogI_dV_vs_z_v(h5_path=None, a_idx=None, a_val=None):
    """Plot d(log(I))/dV vs (Z, V) for a given RF amplitude."""
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    
    # Select RF amplitude
    if a_idx is None:
        if a_val is None:
            a_idx = 0
        else:
            a_idx = np.abs(A_values - a_val).argmin()
    a_val = A_values[a_idx]
    
    # Extract 2D slice
    I_2d = I_cube[:, :, a_idx]  # shape (nZ, nV)
    
    # Calculate d(log(I))/dV
    with np.errstate(divide='ignore', invalid='ignore'):
        log_I = np.log(np.abs(I_2d))
    
    # Use gradient for derivative
    dlogI_dV = np.gradient(log_I, V_values, axis=1)
    
    # Create heatmap
    fig = px.imshow(
        dlogI_dV, 
        x=V_values, 
        y=z_values, 
        origin='lower',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        labels={'x': 'Bias Voltage (V)', 'y': 'Tip Height (nm)', 'color': 'd(log(I))/dV'},
        title=f'd(log(I))/dV vs (Z, V) at RF Amplitude = {a_val:.2f} V'
    )
    
    fig.show()
    return fig

def plot_current_profiles(h5_path=None, z_idx=None, z_val=None, a_idx=None, a_val=None):
    """Plot log10(current) profiles vs voltage for specific tip heights."""
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    
    # Select RF amplitude
    if a_idx is None:
        if a_val is None:
            a_idx = 0
        else:
            a_idx = np.abs(A_values - a_val).argmin()
    a_val = A_values[a_idx]
    
    # Select tip heights to plot
    if z_idx is None and z_val is None:
        # Plot a few representative heights
        z_indices = [0, len(z_values)//4, len(z_values)//2, 3*len(z_values)//4, -1]
    elif z_idx is not None:
        z_indices = [z_idx]
    else:
        z_idx = np.abs(z_values - z_val).argmin()
        z_indices = [z_idx]
    
    fig = go.Figure()
    
    for z_idx in z_indices:
        z_val = z_values[z_idx]
        I_profile = I_cube[z_idx, :, a_idx]
        # Take log10 of current, handle zeros and negatives
        with np.errstate(divide='ignore', invalid='ignore'):
            logI_profile = np.log10(np.abs(I_profile))
            logI_profile[np.isneginf(logI_profile)] = np.nan
        
        fig.add_trace(go.Scatter(
            x=V_values,
            y=logI_profile,
            mode='lines+markers',
            name=f'Z = {z_val:.2f} nm',
            hovertemplate='V: %{x:.2f} V<br>log10(I): %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'log10(Current) vs Voltage at RF Amplitude = {a_val:.2f} V',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='log10(Current)',
        hovermode='x unified'
    )
    
    fig.show()
    return fig

def print_simulation_parameters(h5_path=None):
    if h5_path is None:
        h5_path = find_latest_simulation_file()
    import h5py
    keys_of_interest = [
        'phi_tip', 'phi_samp',
        'n_E', 'n_V', 'n_Z', 'n_A', 'n_cheb'
    ]
    print(f"\nKey simulation parameters for {h5_path}:")
    missing = []
    with h5py.File(h5_path, 'r') as f:
        for key in keys_of_interest:
            if key in f.attrs:
                print(f"  {key}: {f.attrs[key]}")
            else:
                missing.append(key)
        if missing:
            print(f"\nWARNING: Missing attributes: {', '.join(missing)}")
            print("All available attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")

def quick_analysis(h5_path=None, a_val=None):
    """Run all quick analysis plots."""
    print_simulation_parameters(h5_path)
    print("Running quick analysis...")
    
    # Plot 1: Current vs (Z, V)
    print("1. Plotting current vs (Z, V)...")
    plot_current_vs_z_v(h5_path, a_val=a_val)
    
    # Plot 2: d(log(I))/dV vs (Z, V)
    print("2. Plotting d(log(I))/dV vs (Z, V)...")
    plot_dlogI_dV_vs_z_v(h5_path, a_val=a_val)
    
    # Plot 3: Current profiles
    print("3. Plotting current profiles vs voltage...")
    plot_current_profiles(h5_path, a_val=a_val)
    
    print("Quick analysis complete!")

def main():
    parser = argparse.ArgumentParser(description="Quick validation of FER simulation results")
    parser.add_argument("--h5-path", default=None, 
                       help="Path to simulation HDF5 file (default: latest current_phit*.h5 in output dir)")
    parser.add_argument("--a-val", type=float, help="RF amplitude value to analyze")
    parser.add_argument("--plot-type", choices=["current", "dlog", "profiles", "all"], 
                       default="all", help="Type of plot to generate")
    
    args = parser.parse_args()
    
    try:
        if args.plot_type == "current":
            print_simulation_parameters(args.h5_path)
            plot_current_vs_z_v(args.h5_path, a_val=args.a_val)
        elif args.plot_type == "dlog":
            print_simulation_parameters(args.h5_path)
            plot_dlogI_dV_vs_z_v(args.h5_path, a_val=args.a_val)
        elif args.plot_type == "profiles":
            print_simulation_parameters(args.h5_path)
            plot_current_profiles(args.h5_path, a_val=args.a_val)
        else:  # all
            quick_analysis(args.h5_path, a_val=args.a_val)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the simulation first using FER_constant_current_simulation.py")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 