#!/usr/bin/env python3
"""
Constant Current Analysis for FER simulation results.
Calculates dI/dV vs V at constant current levels.
"""

import numpy as np
import plotly.graph_objects as go
import argparse
import sys
from fer_rf_sim.repository import load_simulation
from fer_rf_sim.analysis import build_current_array, constant_current_slice

def load_simulation_data(h5_path="fer_output/current.h5"):
    """Load simulation data from HDF5 file."""
    data = load_simulation(h5_path)
    I_cube = data.current
    z_values = data.z
    V_values = data.voltage
    A_values = data.rf_amplitude
        
    return I_cube, z_values, V_values, A_values

def calculate_dIdV(I_cube, V_values):
    """Calculate dI/dV using finite differences."""
    return build_current_array(I_cube, V_values)[..., 1]

def find_constant_current_slice(I_target, I_cube, z_values, V_values, A_values, a_idx=None,
                                current_array=None):
    """
    Find the tip height Z for each (V, A) pair that gives current I_target.
    
    Parameters:
    -----------
    I_target : float
        Target current level
    I_cube : np.ndarray, shape (nZ, nV, nA)
        Current data
    z_values, V_values, A_values : np.ndarray
        Grid values
    a_idx : int, optional
        RF amplitude index to analyze (if None, analyze all)
    
    Returns:
    --------
    Z_slice : np.ndarray, shape (nV, nA) or (nV,)
        Tip heights that give I_target
    dIdV_slice : np.ndarray, shape (nV, nA) or (nV,)
        dI/dV at those tip heights
    """
    if np.shape(I_cube) != (len(z_values), len(V_values), len(A_values)):
        raise ValueError("I_cube must match (z, V, A) axes")
    if current_array is None:
        current_array = build_current_array(I_cube, V_values)
    if a_idx is not None:
        if not isinstance(a_idx, (int, np.integer)) or not 0 <= a_idx < len(A_values):
            raise ValueError("RF amplitude index is out of range")
        current_array = current_array[:, :, a_idx:a_idx + 1]
    result = constant_current_slice(I_target, z_values, current_array)
    if a_idx is not None:
        result = result[:, 0]
    return result[..., 0], result[..., 1]

def plot_constant_current_analysis(I_target, h5_path="fer_output/current.h5", a_idx=None, a_val=None, *, show=True):
    """
    Plot constant current analysis: Z(V) and dI/dV(V) at constant I.
    """
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    if a_val is not None and not np.isfinite(a_val):
        raise ValueError("RF amplitude selection must be finite")
    
    # Select RF amplitude
    if a_idx is None and a_val is not None:
        a_idx = np.abs(A_values - a_val).argmin()
    print(f"Analyzing constant current I = {I_target:.2e}")
    
    # Calculate constant current slices
    Z_slice, dIdV_slice = find_constant_current_slice(I_target, I_cube, z_values, V_values, A_values, a_idx)
    
    # Create subplots
    fig = go.Figure()
    
    if a_idx is not None:
        # Single RF amplitude
        a_val = A_values[a_idx]
        
        # Plot Z vs V
        fig.add_trace(go.Scatter(
            x=V_values,
            y=Z_slice,
            mode='lines+markers',
            name=f'Z(V) at I={I_target:.2e}',
            yaxis='y',
            line=dict(color='blue'),
            hovertemplate='V: %{x:.2f} V<br>Z: %{y:.3f} nm<extra></extra>'
        ))
        
        # Plot dI/dV vs V on secondary y-axis
        fig.add_trace(go.Scatter(
            x=V_values,
            y=dIdV_slice,
            mode='lines+markers',
            name=f'dI/dV(V) at I={I_target:.2e}',
            yaxis='y2',
            line=dict(color='red'),
            hovertemplate='V: %{x:.2f} V<br>dI/dV: %{y:.2e}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Constant Current Analysis: I = {I_target:.2e} at RF Amplitude = {a_val:.2f} V',
            xaxis_title='Bias Voltage (V)',
            yaxis=dict(title='Tip Height (nm)', side='left'),
            yaxis2=dict(title='dI/dV', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
    else:
        # Multiple RF amplitudes - create heatmaps
        fig = go.Figure()
        
        # Z heatmap
        fig.add_trace(go.Heatmap(
            z=Z_slice,
            x=A_values,
            y=V_values,
            colorscale='Viridis',
            name='Z(V,A)',
            colorbar=dict(title='Tip Height (nm)')
        ))
        
        fig.update_layout(
            title=f'Constant Current Tip Height: I = {I_target:.2e}',
            xaxis_title='RF Amplitude (V)',
            yaxis_title='Bias Voltage (V)'
        )
    
    if show:
        fig.show()
    return fig

def analyze_multiple_currents(I_targets, h5_path="fer_output/current.h5", a_idx=None, a_val=None, *, show=True):
    """
    Analyze multiple current levels and plot comparison.
    """
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    if a_val is not None and not np.isfinite(a_val):
        raise ValueError("RF amplitude selection must be finite")
    if len(I_targets) == 0:
        raise ValueError("At least one target current is required")
    
    # Select RF amplitude
    if a_idx is None and a_val is not None:
        a_idx = np.abs(A_values - a_val).argmin()
    
    if a_idx is None:
        a_idx = 0
    current_array = build_current_array(I_cube, V_values)
    fig = go.Figure()

    for I_target in I_targets:
        Z_slice, dIdV_slice = find_constant_current_slice(
            I_target, I_cube, z_values, V_values, A_values, a_idx, current_array,
        )
        
        if a_idx is not None:
            # Plot dI/dV vs V for each current level
            fig.add_trace(go.Scatter(
                x=V_values,
                y=dIdV_slice,
                mode='lines+markers',
                name=f'I = {I_target:.2e}',
                hovertemplate='V: %{x:.2f} V<br>dI/dV: %{y:.2e}<extra></extra>'
            ))
    
    if a_idx is not None:
        a_val = A_values[a_idx]
        fig.update_layout(
            title=f'dI/dV vs V at Constant Current Levels (RF Amplitude = {a_val:.2f} V)',
            xaxis_title='Bias Voltage (V)',
            yaxis_title='dI/dV',
            yaxis_type='log'
        )
    
    if show:
        fig.show()
    return fig

def main(argv=None):
    parser = argparse.ArgumentParser(description="Constant Current Analysis for FER simulation")
    parser.add_argument("--h5-path", default="fer_output/current.h5", 
                       help="Path to simulation HDF5 file")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--I-target", type=float,
                       help="Target current level for constant current analysis")
    parser.add_argument("--a-val", type=float, help="RF amplitude value to analyze")
    parser.add_argument("--a-idx", type=int, help="RF amplitude index to analyze")
    parser.add_argument("--no-show", action="store_true", help="Validate figures without opening a browser")
    targets.add_argument("--multiple", nargs='+', type=float,
                       help="Analyze multiple current levels (space-separated)")
    
    args = parser.parse_args(argv)
    
    try:
        if args.multiple:
            print(f"Analyzing multiple current levels: {args.multiple}")
            analyze_multiple_currents(args.multiple, args.h5_path, args.a_idx, args.a_val, show=not args.no_show)
        else:
            plot_constant_current_analysis(args.I_target, args.h5_path, args.a_idx, args.a_val, show=not args.no_show)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the simulation first using FER_constant_current_simulation.py")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())