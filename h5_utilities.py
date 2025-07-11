#!/usr/bin/env python3
"""
HDF5 File Utilities for FER Simulation
Demonstrates how to read, write, and explore HDF5 files.
"""

import h5py
import numpy as np
from pathlib import Path

def explore_h5_file(file_path):
    """Explore the structure and contents of an HDF5 file."""
    print(f"Exploring: {file_path}")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        # Show all datasets
        print("Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if dataset.size < 10:  # Show small datasets
                print(f"    data: {dataset[...]}")
        
        # Show all attributes
        print("\nAttributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Show file size
        file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
        print(f"\nFile size: {file_size:.1f} MB")

def read_simulation_data(file_path):
    """Read simulation data and return as a dictionary."""
    with h5py.File(file_path, 'r') as f:
        data = {
            'I': f['I'][...],           # Current data (nZ, nV, nA)
            'z': f['z'][...],           # Tip height values
            'V': f['V'][...],           # Voltage values
            'A_rf': f['A_rf'][...],     # RF amplitude values
        }
        
        # Read all attributes
        for key, value in f.attrs.items():
            data[key] = value
            
    return data

def print_simulation_summary(file_path):
    """Print a summary of the simulation parameters and data."""
    data = read_simulation_data(file_path)
    
    print(f"Simulation Summary: {file_path}")
    print("=" * 60)
    
    # Parameters
    print("Parameters:")
    print(f"  Work functions: φ_tip = {data['phi_tip']:.1f} eV, φ_samp = {data['phi_samp']:.1f} eV")
    print(f"  Grid sizes: n_E = {data['n_E']}, n_V = {data['n_V']}, n_Z = {data['n_Z']}, n_A = {data['n_A']}")
    print(f"  Voltage range: {data['v_min']:.1f} to {data['v_max']:.1f} V")
    print(f"  Tip height range: {data['z_min']:.1f} to {data['z_max']:.1f} nm")
    print(f"  RF amplitude range: {data['A_min']:.1f} to {data['A_max']:.1f} V")
    print(f"  Fudge factor: {data['fudge_factor']:.3f}")
    print(f"  Used LUT: {data['use_lut']}")
    
    # Data statistics
    print("\nData Statistics:")
    I = data['I']
    print(f"  Current range: {I.min():.2e} to {I.max():.2e}")
    print(f"  Current mean: {I.mean():.2e}")
    print(f"  Non-zero currents: {np.count_nonzero(I)} / {I.size}")
    
    # Memory usage
    memory_mb = I.nbytes / (1024*1024)
    print(f"  Memory usage: {memory_mb:.1f} MB")

def extract_2d_slice(file_path, a_idx=0, save_path=None):
    """Extract a 2D slice (I vs Z,V) for a specific RF amplitude."""
    data = read_simulation_data(file_path)
    
    I_2d = data['I'][:, :, a_idx]  # shape (nZ, nV)
    z_values = data['z']
    V_values = data['V']
    A_value = data['A_rf'][a_idx]
    
    print(f"Extracted 2D slice for RF amplitude = {A_value:.2f} V")
    print(f"Shape: {I_2d.shape}")
    
    if save_path:
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('I', data=I_2d)
            f.create_dataset('z', data=z_values)
            f.create_dataset('V', data=V_values)
            f.attrs['A_rf'] = A_value
            f.attrs['source_file'] = str(file_path)
        print(f"Saved to: {save_path}")
    
    return I_2d, z_values, V_values, A_value

def compare_simulations(file1, file2):
    """Compare two simulation files."""
    print(f"Comparing simulations:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print("=" * 60)
    
    data1 = read_simulation_data(file1)
    data2 = read_simulation_data(file2)
    
    # Compare parameters
    param_keys = ['phi_tip', 'phi_samp', 'n_E', 'n_V', 'n_Z', 'n_A', 
                  'v_min', 'v_max', 'z_min', 'z_max', 'A_min', 'A_max']
    
    print("Parameter differences:")
    for key in param_keys:
        if key in data1 and key in data2:
            val1, val2 = data1[key], data2[key]
            if val1 != val2:
                print(f"  {key}: {val1} vs {val2}")
    
    # Compare data if shapes match
    if data1['I'].shape == data2['I'].shape:
        diff = np.abs(data1['I'] - data2['I'])
        print(f"\nData differences:")
        print(f"  Max difference: {diff.max():.2e}")
        print(f"  Mean difference: {diff.mean():.2e}")
        print(f"  RMS difference: {np.sqrt((diff**2).mean()):.2e}")
    else:
        print(f"\nData shapes don't match:")
        print(f"  File 1: {data1['I'].shape}")
        print(f"  File 2: {data2['I'].shape}")

def main():
    """Example usage of HDF5 utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HDF5 File Utilities")
    parser.add_argument("file", help="HDF5 file to analyze")
    parser.add_argument("--explore", action="store_true", help="Explore file structure")
    parser.add_argument("--summary", action="store_true", help="Print simulation summary")
    parser.add_argument("--extract", type=int, help="Extract 2D slice for RF amplitude index")
    parser.add_argument("--compare", help="Compare with another file")
    parser.add_argument("--save-slice", help="Save extracted slice to this file")
    
    args = parser.parse_args()
    
    if args.explore:
        explore_h5_file(args.file)
    
    if args.summary:
        print_simulation_summary(args.file)
    
    if args.extract is not None:
        extract_2d_slice(args.file, args.extract, args.save_slice)
    
    if args.compare:
        compare_simulations(args.file, args.compare)
    
    # Default: show summary
    if not any([args.explore, args.summary, args.extract is not None, args.compare]):
        print_simulation_summary(args.file)

if __name__ == "__main__":
    main() 