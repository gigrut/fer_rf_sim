import numpy as np
import pandas as pd
import h5py
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator, interp1d
import argparse
# Note: constant_current_slice and build_constant_current_4D are defined in this file
# The plotting functions from FER_plotter are not available in the current version

# ------------------------------------------------------------------------------
# Load the simulation output (must run FER_constant_current_simulation.py first)
# ------------------------------------------------------------------------------

H5_PATH = "fer_output/current.h5"

with h5py.File(H5_PATH, "r") as f:
    # I has shape (nD, nV, nA)
    I_cube     = f["I"][...]           # current
    D_values   = f["z"][...]           # tip-height axis
    V_values   = f["V"][...]           # bias axis
    A_values   = f["A_rf"][...]        # RF amplitude axis

# Compute dI/dV by finite‐difference along the V axis
# shape will be exactly (nD, nV, nA)
dI_dV_cube = np.gradient(I_cube, V_values, axis=1)

# Build the full 4D array (nD, nV, nA, 2)
#   [:,:,:,0] = I, [:,:,:,1] = dI/dV
current_array = np.stack([I_cube, dI_dV_cube], axis=-1)


def upsample_current_array(current_array, D_values, V_values, A_values,
                           factor_D=1, factor_V=1, factor_A=1):
    """
    Upsample a 4D array of current (I) and its derivative (dI/dV) along the
    D (tip height), V (bias voltage), and A (RF amplitude) dimensions.

    Parameters
    ----------
    current_array : np.ndarray, shape (nD, nV, nA, 2)
        Array containing I in channel 0 and dI/dV in channel 1.
    D_values : np.ndarray, shape (nD,)
        Original tip-height grid values.
    V_values : np.ndarray, shape (nV,)
        Original bias-voltage grid values.
    A_values : np.ndarray, shape (nA,)
        Original RF-amplitude grid values.
    factor_D : int
        Upsampling factor along the D axis.
    factor_V : int
        Upsampling factor along the V axis.
    factor_A : int
        Upsampling factor along the A axis.

    Returns
    -------
    new_D : np.ndarray, shape (nD*factor_D,)
        Upsampled tip-height grid.
    new_V : np.ndarray, shape (nV*factor_V,)
        Upsampled bias-voltage grid.
    new_A : np.ndarray, shape (nA*factor_A,)
        Upsampled RF-amplitude grid.
    up : np.ndarray, shape (nD*factor_D, nV*factor_V, nA*factor_A, 2)
        Interpolated array with I and dI/dV on the finer grid.
    """
    nD, nV, nA, _ = current_array.shape
    new_nD = nD * factor_D
    new_nV = nV * factor_V
    new_nA = nA * factor_A

    new_D = np.linspace(D_values.min(), D_values.max(), new_nD)
    new_V = np.linspace(V_values.min(), V_values.max(), new_nV)
    new_A = np.linspace(A_values.min(), A_values.max(), new_nA)

    # Create meshgrid of target points
    Dm, Vm, Am = np.meshgrid(new_D, new_V, new_A, indexing="ij")
    pts = np.stack([Dm, Vm, Am], axis=-1)   # shape (new_nD,new_nV,new_nA,3)

    up = np.empty((new_nD, new_nV, new_nA, 2), float)
    for ch in (0,1):
        slice_ch = current_array[..., ch]
        interp = RegularGridInterpolator(
            (D_values, V_values, A_values),
            slice_ch,
            bounds_error=False,
            fill_value=None
        )
        up[..., ch] = interp(pts)
    return new_D, new_V, new_A, up


def invert_current_for_constant_I_4D_vectorized(I_target, D_vals, M):
    """
    Find, for each (V, A) pair, the tip height D such that I(D, V, A) == I_target,
    then return both that D and the corresponding dI/dV at that height.

    Parameters
    ----------
    I_target : float
        Desired current level to slice at.
    D_vals : np.ndarray, shape (nD,)
        Tip-height grid over which I is defined.
    M : np.ndarray, shape (nD, nV, nA, 2)
        4D array where M[...,0] is I and M[...,1] is dI/dV.

    Returns
    -------
    Dt : np.ndarray, shape (nV, nA)
        Interpolated tip height for each (V, A) that yields I_target.
    dIdVt : np.ndarray, shape (nV, nA)
        Interpolated dI/dV at those tip heights.
    """
    nD,nV,nA,_ = M.shape
    Dt = np.empty((nV,nA))
    dIdVt = np.empty((nV,nA))
    for j in range(nV):
        for k in range(nA):
            I_vs_D    = M[:,j,k,0]
            dIdV_vs_D = M[:,j,k,1]
            D_axis    = D_vals

            # ensure monotonic increasing I(D)
            if I_vs_D[0] > I_vs_D[-1]:
                I_vs_D    = I_vs_D[::-1]
                dIdV_vs_D = dIdV_vs_D[::-1]
                D_axis    = D_axis[::-1]

            # invert current to find D
            f_inv = interp1d(I_vs_D, D_axis,
                             bounds_error=False,
                             fill_value="extrapolate")
            Di = f_inv(I_target)
            Dt[j,k] = Di

            # interpolate dI/dV at that D
            f_didv = interp1d(D_axis, dIdV_vs_D,
                              bounds_error=False,
                              fill_value="extrapolate")
            dIdVt[j,k] = f_didv(Di)
    return Dt, dIdVt


def constant_current_slice(I_target, D_vals, M):
    """
    Compute a constant-current slice at I_target, returning both the tip height
    and dI/dV as functions of (V, A).

    Parameters
    ----------
    I_target : float
        Desired current level.
    D_vals : np.ndarray, shape (nD,)
        Tip-height grid.
    M : np.ndarray, shape (nD, nV, nA, 2)
        4D array with I and dI/dV.

    Returns
    -------
    slice_4D : np.ndarray, shape (nV, nA, 2)
        Where slice_4D[...,0] is D(V,A) and slice_4D[...,1] is dI/dV(V,A).
    """
    Dt, dIdVt = invert_current_for_constant_I_4D_vectorized(I_target, D_vals, M)
    return np.stack((Dt, dIdVt), axis=-1)


def build_constant_current_4D(I_target_list, D_vals, M):
    """
    Build a 4D stack of constant-current slices for multiple target currents.

    Parameters
    ----------
    I_target_list : sequence of float
        Currents at which to compute constant-current slices.
    D_vals : np.ndarray, shape (nD,)
        Tip-height grid.
    M : np.ndarray, shape (nD, nV, nA, 2)
        4D array with I and dI/dV.

    Returns
    -------
    const_curr_4D : np.ndarray, shape (nI, nV, nA, 2)
        Stack of constant-current slices for each I in I_target_list.
    """
    return np.stack([
        constant_current_slice(I_t, D_vals, M)
        for I_t in I_target_list
    ], axis=0)

###################################################
# HELPER FUNCTIONS FOR PLOTTING
###################################################


if __name__ == "__main__":
    # first upsample if you like:
    upsample = False
    if upsample:
        print('Begin upsampling')
        D, V, A, curr = upsample_current_array(
            current_array, D_values, V_values, A_values,
            factor_D=1, factor_V=1, factor_A=1
        )
        print('Done upsampling')
    else:
        D, V, A, curr = D_values, V_values, A_values, current_array

    # pick some target currents and A‐slice
    I_targets = np.logspace(-10, -6, 50)
    I_small_list = np.array([1E-10, 1E-9, 1E-8])
    Z_small_list = D[::20]
    
    cc4 = build_constant_current_4D(I_targets, D_values, current_array)


    print('Plotting...')
    Z_vs_V=1
    if Z_vs_V:
        plot_Z_vs_V_at_constant_I_and_A(curr, D, V, A, I_targets, x_bounds=[1,10])
    
    vol=0
    if vol:
        plot_current_and_dIdV_volume(curr, D, V, A, plot_curr=False, plot_dIdV=True,
                                    opacity=0.5, opacityscale='uniform', 
                                    iso_range=[5, 99] ,surface_count=5, log=True, caps=False)
    dIdV_vs_V=1
    if dIdV_vs_V:
        plot_dIdV_vs_V_at_constant_I_and_A(curr, D, V, A, I_small_list, y_bounds=[1,10])

    dIdV_CC_heatmap=1
    if dIdV_CC_heatmap:
        print('Plotting CC heatmap')
        plot_dIdV_heatmap_CC_slider(
            current_array, D_values, V_values, A_values,
            I_targets=I_targets, x_bounds=[1,10])
    
    dIdV_CH_heatmap=0
    if dIdV_CH_heatmap:  
        print('Plotting CH heatmap')
        plot_dIdV_heatmap_CH_slider(curr, D, V, A,
                                    D, x_bounds=[1,10], log_scale=True)
        
        
    isosurface=0
    if isosurface:
        plot_constant_current_isosurface(cc4, curr, D, V, A,
                                        I_targets=[1e-10])

    print('Done')