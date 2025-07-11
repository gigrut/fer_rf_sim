#!/usr/bin/env python3
"""
Constant Current Analysis for FER simulation results.
Calculates dI/dV vs V at constant current levels.
"""

import numpy as np
import h5py
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d
from pathlib import Path
import argparse

def load_simulation_data(h5_path="fer_output/current.h5"):
    """Load simulation data from HDF5 file."""
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"Simulation file not found: {h5_path}")
    
    with h5py.File(h5_path, "r") as f:
        I_cube = f["I"][...]           # current (nZ, nV, nA)
        z_values = f["z"][...]         # tip-height axis
        V_values = f["V"][...]         # bias axis
        A_values = f["A_rf"][...]      # RF amplitude axis
        
    return I_cube, z_values, V_values, A_values

def calculate_dIdV(I_cube, V_values):
    """Calculate dI/dV using finite differences."""
    return np.gradient(I_cube, V_values, axis=1)

def find_constant_current_slice(I_target, I_cube, z_values, V_values, A_values, a_idx=None):
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
    if a_idx is not None:
        # Analyze single RF amplitude
        I_2d = I_cube[:, :, a_idx]  # shape (nZ, nV)
        dIdV_2d = calculate_dIdV(I_cube[:, :, a_idx:a_idx+1], V_values)[:, :, 0]
        
        Z_slice = np.empty(len(V_values))
        dIdV_slice = np.empty(len(V_values))
        
        for v_idx in range(len(V_values)):
            I_vs_z = I_2d[:, v_idx]
            dIdV_vs_z = dIdV_2d[:, v_idx]
            
            # Ensure monotonic behavior
            if I_vs_z[0] > I_vs_z[-1]:
                I_vs_z = I_vs_z[::-1]
                dIdV_vs_z = dIdV_vs_z[::-1]
                z_axis = z_values[::-1]
            else:
                z_axis = z_values
            
            # Interpolate to find Z for I_target
            try:
                f_inv = interp1d(I_vs_z, z_axis, bounds_error=False, fill_value="extrapolate")
                Z_slice[v_idx] = f_inv(I_target)
                
                # Interpolate dI/dV at that Z
                f_didv = interp1d(z_axis, dIdV_vs_z, bounds_error=False, fill_value="extrapolate")
                dIdV_slice[v_idx] = f_didv(Z_slice[v_idx])
            except:
                Z_slice[v_idx] = np.nan
                dIdV_slice[v_idx] = np.nan
        
        return Z_slice, dIdV_slice
    
    else:
        # Analyze all RF amplitudes
        Z_slice = np.empty((len(V_values), len(A_values)))
        dIdV_slice = np.empty((len(V_values), len(A_values)))
        
        for a_idx in range(len(A_values)):
            Z_a, dIdV_a = find_constant_current_slice(I_target, I_cube, z_values, V_values, A_values, a_idx)
            Z_slice[:, a_idx] = Z_a
            dIdV_slice[:, a_idx] = dIdV_a
        
        return Z_slice, dIdV_slice

def plot_constant_current_analysis(I_target, h5_path="fer_output/current.h5", a_idx=None, a_val=None):
    """
    Plot constant current analysis: Z(V) and dI/dV(V) at constant I.
    """
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    
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
    
    fig.show()
    return fig

def analyze_multiple_currents(I_targets, h5_path="fer_output/current.h5", a_idx=None, a_val=None):
    """
    Analyze multiple current levels and plot comparison.
    """
    I_cube, z_values, V_values, A_values = load_simulation_data(h5_path)
    
    # Select RF amplitude
    if a_idx is None and a_val is not None:
        a_idx = np.abs(A_values - a_val).argmin()
    
    fig = go.Figure()
    
    for I_target in I_targets:
        Z_slice, dIdV_slice = find_constant_current_slice(I_target, I_cube, z_values, V_values, A_values, a_idx)
        
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
    
    fig.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Constant Current Analysis for FER simulation")
    parser.add_argument("--h5-path", default="fer_output/current.h5", 
                       help="Path to simulation HDF5 file")
    parser.add_argument("--I-target", type=float, required=True,
                       help="Target current level for constant current analysis")
    parser.add_argument("--a-val", type=float, help="RF amplitude value to analyze")
    parser.add_argument("--a-idx", type=int, help="RF amplitude index to analyze")
    parser.add_argument("--multiple", nargs='+', type=float,
                       help="Analyze multiple current levels (space-separated)")
    
    args = parser.parse_args()
    
    try:
        if args.multiple:
            print(f"Analyzing multiple current levels: {args.multiple}")
            analyze_multiple_currents(args.multiple, args.h5_path, args.a_idx, args.a_val)
        else:
            plot_constant_current_analysis(args.I_target, args.h5_path, args.a_idx, args.a_val)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the simulation first using FER_constant_current_simulation.py")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 