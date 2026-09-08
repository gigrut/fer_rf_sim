#!/usr/bin/env python3
"""
Test script to verify mathematical fixes in FER_constant_current_simulation.py
"""

import numpy as np
import pytest
import warnings
import subprocess
from pathlib import Path

from FER_constant_current_simulation import (
    _cardano_two_smallest, 
    image_potential, 
    airy_all, 
    transmission, 
    calibrate_fudge,
    current
)

import run_workflow


def test_run_workflow_accepts_parameterized_sim_output(tmp_path, monkeypatch):
    """Workflow should detect the newest current_phit*.h5 result."""
    output_dir = tmp_path / "fer_output"
    output_dir.mkdir()
    output_file = output_dir / "current_phit_42.h5"
    output_file.touch()

    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_workflow.run_quick_check() is True
    assert calls == [["python", "quick_check.py"]]


def test_run_workflow_requires_no_legacy_current_file(tmp_path, monkeypatch):
    """Workflow should not fail solely because fer_output/current.h5 is missing."""
    output_dir = tmp_path / "fer_output"
    output_dir.mkdir()
    (output_dir / "current_phit_99.h5").touch()

    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_workflow.run_constant_current_analysis(1e-9) is True

def test_cardano_solver():
    """Test the Cardano solver with edge cases."""
    print("Testing Cardano solver...")
    
    # Test with normal coefficients
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([0.0, -1.0, -2.0])
    c = np.array([0.0, 0.0, -1.0])
    d = np.array([0.0, 0.0, 0.0])
    
    roots = _cardano_two_smallest(a, b, c, d)
    print(f"Normal case roots: {roots}")
    assert np.all(np.isfinite(roots)), "Roots should be finite"
    
    # A zero cubic coefficient should fall back to the quadratic equation.
    roots_quadratic = _cardano_two_smallest(
        np.array([0.0]),
        np.array([1.0]),
        np.array([-3.0]),
        np.array([2.0]),
    )
    print(f"Quadratic roots: {roots_quadratic}")
    assert np.allclose(roots_quadratic, [[1.0, 2.0]])
    
    print("✓ Cardano solver tests passed")

def test_image_potential():
    """Test image potential calculation with edge cases."""
    print("Testing image potential...")
    
    # Test with normal parameters
    D = 1.0
    V = np.array([0.0, 0.1, 1.0])
    phi_t = 4.0
    
    phi_im = image_potential(D, V, phi_t)
    print(f"Normal case phi_im: {phi_im}")
    assert np.all(np.isfinite(phi_im)), "Image potential should be finite"
    
    # Test with zero voltage
    phi_im_zero = image_potential(D, 0.0, phi_t)
    print(f"Zero voltage phi_im: {phi_im_zero}")
    assert np.isfinite(phi_im_zero), "Image potential should be finite for zero voltage"
    
    # Test with negative voltage
    phi_im_neg = image_potential(D, -1.0, phi_t)
    print(f"Negative voltage phi_im: {phi_im_neg}")
    assert np.isfinite(phi_im_neg), "Image potential should be finite for negative voltage"
    
    print("✓ Image potential tests passed")

def test_airy_functions():
    """Test Airy function evaluation with edge cases."""
    print("Testing Airy functions...")
    
    # Test with normal values
    z_normal = np.array([-5.0, 0.0, 5.0, 10.0])
    airy_vals = airy_all(z_normal)
    print(f"Normal case Airy values shape: {airy_vals.shape}")
    assert np.all(np.isfinite(airy_vals)), "Airy values should be finite"
    
    # Test with very large values (should handle overflow)
    z_large = np.array([100.0, 1000.0])
    airy_large = airy_all(z_large)
    print(f"Large z Airy values: {airy_large}")
    assert np.all(np.isfinite(airy_large)), "Airy values should be finite for large z"
    
    # Test with very small values
    z_small = np.array([1e-10, 1e-8])
    airy_small = airy_all(z_small)
    print(f"Small z Airy values: {airy_small}")
    assert np.all(np.isfinite(airy_small)), "Airy values should be finite for small z"
    
    print("✓ Airy function tests passed")

def test_transmission():
    """Test transmission calculation with edge cases."""
    print("Testing transmission...")
    
    # Test with normal parameters
    E = np.array([0.1, 1.0, 2.0])[:, None]
    V = np.array([0.5, 1.0])[None, :]
    d = 1.0
    phi_t = 4.0
    phi_s = 4.0
    
    T = transmission(E, V, d, phi_t, phi_s)
    print(f"Normal case transmission shape: {T.shape}")
    assert T.shape == (3, 2)
    assert np.all(np.isfinite(T)), "Transmission should be finite"
    assert np.all((T >= 0) & (T <= 1)), "Transmission must be a probability"
    
    # Test with zero field (should use WKB)
    T_wkb = transmission(E, 0.0, d, phi_t, phi_s)
    print(f"Zero field transmission: {T_wkb}")
    assert np.all(np.isfinite(T_wkb)), "WKB transmission should be finite"
    expected_wkb = 3.69 * np.exp(-2 * 5.12 * d * np.sqrt(phi_t + 5.5 - E))
    assert np.allclose(T_wkb, expected_wkb)
    
    # Test with very small barrier thickness
    T_thin = transmission(E, V, 0.1, phi_t, phi_s)
    print(f"Thin barrier transmission shape: {T_thin.shape}")
    assert np.all(np.isfinite(T_thin)), "Thin barrier transmission should be finite"
    assert np.all(T_thin >= T), "A thinner barrier should not reduce transmission"

    with pytest.raises(ValueError, match="must be positive"):
        transmission(1.0, 1.0, 0.0, phi_t, phi_s)
    
    print("✓ Transmission tests passed")

def test_current_calculation():
    """Test current calculation with edge cases."""
    print("Testing current calculation...")
    
    # Test with normal parameters
    Eg = np.array([0.1, 1.0, 2.0, 3.0, 4.0])
    D_E = np.array([0.1, 0.5, 0.3, 0.1, 0.01])
    Vdc = 1.0
    
    I = current(D_E, Eg, Vdc)
    print(f"Normal case current: {I}")
    assert np.isfinite(I), "Current should be finite"
    assert I >= 0, "Current should be non-negative"
    
    # Test with zero transmission
    I_zero = current(np.zeros_like(D_E), Eg, Vdc)
    print(f"Zero transmission current: {I_zero}")
    assert I_zero == 0.0, "Zero transmission should give zero current"
    
    # Non-finite transmission must be rejected instead of silently integrated.
    D_E_nan = D_E.copy()
    D_E_nan[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        current(D_E_nan, Eg, Vdc)
    
    print("✓ Current calculation tests passed")

def test_fudge_calibration():
    """Test fudge factor calibration with edge cases."""
    print("Testing fudge calibration...")
    
    fudge, samples = calibrate_fudge(
        phi_t=4.0, phi_s=4.0, F1=0.2,
        D_min=0.3, D_max=5.0, n_D=10, n_E=50
    )
    print(f"Normal case fudge: {fudge}, samples: {samples}")
    assert np.isfinite(fudge), "Fudge factor should be finite"
    assert fudge > 0, "Fudge factor should be positive"
    assert samples.size > 0

    with pytest.raises(ValueError, match="0 < D_min < D_max"):
        calibrate_fudge(4.0, 4.0, 0.2, D_min=0.0, D_max=5.0)
    
    print("✓ Fudge calibration tests passed")

def test_edge_cases():
    """Test various edge cases that could cause problems."""
    print("Testing edge cases...")

    T_small = transmission(1e-12, 1e-12, 1e-12, 1e-12, 1e-12)
    print(f"Very small parameters transmission: {T_small}")
    assert np.all(np.isfinite(T_small)), "Should handle very small positive parameters"

    T_large = transmission(100.0, 100.0, 100.0, 100.0, 100.0)
    print(f"Very large parameters transmission: {T_large}")
    assert np.all(np.isfinite(T_large)), "Should handle very large parameters"
    assert np.all((T_large >= 0) & (T_large <= 1))
    
    print("✓ Edge case tests passed")

def main():
    """Run all tests."""
    print("Running mathematical fixes tests...\n")
    
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        test_cardano_solver()
        print()
        
        test_image_potential()
        print()
        
        test_airy_functions()
        print()
        
        test_transmission()
        print()
        
        test_current_calculation()
        print()
        
        test_fudge_calibration()
        print()
        
        test_edge_cases()
        print()
    
    print("🎉 All tests completed successfully!")
    print("The mathematical fixes appear to be working correctly.")

if __name__ == "__main__":
    main() 