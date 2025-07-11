#!/usr/bin/env python3
"""
Benchmark LUT interpolation vs direct transmission calculation
"""

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from FER_constant_current_simulation import transmission, build_lut

def benchmark_transmission_vs_interpolation():
    """Compare performance of direct transmission vs LUT interpolation."""
    
    # Test parameters
    n_E = 50
    n_V = 50  
    n_Z = 50
    n_cheb = 32
    
    # Create test grids
    E_F = 5.5
    Eg = np.linspace(0.01, E_F, n_E)
    Vg = np.linspace(1.0, 10.0, n_V)
    zg = np.linspace(0.3, 5.0, n_Z)
    
    phi_t = 4.0
    phi_s = 4.0
    
    print(f"Benchmarking with grids: n_E={n_E}, n_V={n_V}, n_Z={n_Z}")
    print(f"Chebyshev nodes: {n_cheb}")
    print()
    
    # Build LUT
    print("Building LUT...")
    t0 = time.perf_counter()
    lut = build_lut(Eg, Vg, zg, phi_t, phi_s, upscale=1, fudge=3.69)
    lut_time = time.perf_counter() - t0
    print(f"LUT build time: {lut_time:.2f} s")
    
    # Create interpolator
    interp = RegularGridInterpolator((zg, Vg, Eg), lut, bounds_error=False, fill_value=0.0)
    
    # Test case: single (z, V, A) point
    z_test = 1.0
    Vdc_test = 5.0
    Arf_test = 2.0
    nodes = np.pi*(2*np.arange(1, n_cheb+1)-1)/(2*n_cheb)
    
    # Method 1: Direct transmission calculation
    print("\nTesting direct transmission calculation...")
    t0 = time.perf_counter()
    for _ in range(10):  # Run multiple times for better timing
        D_direct = np.zeros(n_E)
        for i, E in enumerate(Eg):
            V_inst = Vdc_test + Arf_test * np.cos(nodes)
            # Ensure arrays for transmission function
            E_arr = np.full_like(V_inst, E)
            z_arr = np.full_like(V_inst, z_test)
            T_vals = transmission(E_arr, V_inst, z_arr, phi_t, phi_s)
            D_direct[i] = T_vals.mean()
    direct_time = (time.perf_counter() - t0) / 10
    print(f"Direct calculation time: {direct_time:.4f} s")
    
    # Method 2: LUT interpolation
    print("Testing LUT interpolation...")
    t0 = time.perf_counter()
    for _ in range(1000):  # Run many more times since it's faster
        V_inst = Vdc_test + Arf_test * np.cos(nodes)
        nE, nN = Eg.size, V_inst.size
        E_rep = np.repeat(Eg, nN)
        V_rep = np.tile(V_inst, nE)
        z_rep = np.full(E_rep.shape, z_test)
        pts = np.stack([z_rep, V_rep, E_rep], axis=1)
        vals = interp(pts).reshape(nE, nN)
        D_lut = vals.mean(axis=1)
    lut_time = (time.perf_counter() - t0) / 1000
    print(f"LUT interpolation time: {lut_time:.6f} s")
    
    # Compare results
    print(f"\nSpeedup: {direct_time/lut_time:.1f}x")
    print(f"Relative error: {np.max(np.abs(D_direct - D_lut) / (np.abs(D_direct) + 1e-12)):.2e}")
    
    # Calculate total operations for full simulation
    n_A = 10  # Default from your change
    total_direct_ops = n_Z * n_V * n_A * n_E * n_cheb
    total_lut_ops = n_Z * n_V * n_A * n_E * n_cheb
    
    print(f"\nFull simulation operations:")
    print(f"  Direct: {total_direct_ops:,} transmission calculations")
    print(f"  LUT: {n_Z * n_V * n_E:,} transmission calculations + {total_lut_ops:,} interpolations")
    
    estimated_direct_time = total_direct_ops * direct_time / (n_E * n_cheb)
    estimated_lut_time = (n_Z * n_V * n_E * direct_time / (n_E * n_cheb)) + (total_lut_ops * lut_time)
    
    print(f"\nEstimated full simulation times:")
    print(f"  Direct: {estimated_direct_time/60:.1f} minutes")
    print(f"  LUT: {estimated_lut_time/60:.1f} minutes")
    print(f"  LUT advantage: {estimated_direct_time/estimated_lut_time:.1f}x faster")

if __name__ == '__main__':
    benchmark_transmission_vs_interpolation() 