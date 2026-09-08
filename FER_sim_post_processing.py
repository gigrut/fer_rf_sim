import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path

from h5_utilities import find_latest_simulation_file
from simulation_repository import load_simulation

# ------------------------------------------------------------------------------
# Load simulation data only when explicitly requested. Importing this module must
# remain safe for GUI and test callers.
# ------------------------------------------------------------------------------

def build_current_array(I_cube, V_values):
    """Return channels ``[..., 0] = I`` and ``[..., 1] = dI/dV``."""
    I_cube = np.asarray(I_cube, dtype=np.float64)
    V_values = np.asarray(V_values, dtype=np.float64)
    if I_cube.ndim != 3 or V_values.ndim != 1:
        raise ValueError("I_cube must be 3D and V_values must be 1D")
    if I_cube.shape[1] != V_values.size:
        raise ValueError("V_values must match the voltage axis of I_cube")
    dI_dV_cube = np.gradient(I_cube, V_values, axis=1)
    return np.stack([I_cube, dI_dV_cube], axis=-1)


def load_post_processing_data(h5_path=None):
    """Load one artifact and derive the current/dI/dV analysis array."""
    if h5_path is None:
        h5_path = find_latest_simulation_file()
    data = load_simulation(h5_path)
    current_array = build_current_array(data.current, data.voltage)
    return data.path, data.z, data.voltage, data.rf_amplitude, current_array


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
    factors = (factor_D, factor_V, factor_A)
    if any(not isinstance(factor, (int, np.integer)) or factor < 1 for factor in factors):
        raise ValueError("upsampling factors must be positive integers")
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
    if D_vals.ndim != 1 or D_vals.size != nD:
        raise ValueError("D_vals must match the first dimension of M")
    if np.any(np.diff(D_vals) <= 0):
        raise ValueError("D_vals must be strictly increasing")
    Dt = np.full((nV,nA), np.nan)
    dIdVt = np.full((nV,nA), np.nan)
    for j in range(nV):
        for k in range(nA):
            I_vs_D = M[:,j,k,0]
            dIdV_vs_D = M[:,j,k,1]

            # Ensure a strictly monotonic current curve before inversion.
            if I_vs_D[0] > I_vs_D[-1]:
                current_axis = I_vs_D[::-1]
                height_axis = D_vals[::-1]
            else:
                current_axis = I_vs_D
                height_axis = D_vals
            if np.any(np.diff(current_axis) <= 0):
                continue
            if not current_axis[0] <= I_target <= current_axis[-1]:
                continue

            # invert current to find D
            Di = np.interp(I_target, current_axis, height_axis)
            Dt[j,k] = Di

            # D_vals remains increasing even when current decreases with height.
            dIdVt[j,k] = np.interp(Di, D_vals, dIdV_vs_D)
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


def main():
    parser = argparse.ArgumentParser(description="Simulation post-processing")
    parser.add_argument(
        "--h5-path",
        help="Simulation artifact to analyze (defaults to the newest output)",
    )
    parser.add_argument(
        "--I-target",
        type=float,
        help="Optional normalized current target for constant-current analysis",
    )
    parser.add_argument("--factor-D", type=int, default=1)
    parser.add_argument("--factor-V", type=int, default=1)
    parser.add_argument("--factor-A", type=int, default=1)
    args = parser.parse_args()

    path, D, V, A, current_array = load_post_processing_data(args.h5_path)
    if max(args.factor_D, args.factor_V, args.factor_A) > 1:
        D, V, A, current_array = upsample_current_array(
            current_array, D, V, A,
            factor_D=args.factor_D,
            factor_V=args.factor_V,
            factor_A=args.factor_A,
        )

    print(f"Loaded: {path}")
    print(f"Current shape: {current_array[..., 0].shape}")
    print(f"Axes: z={D.size}, V={V.size}, A={A.size}")
    if args.I_target is not None:
        result = constant_current_slice(args.I_target, D, current_array)
        print(f"Constant-current slice shape: {result.shape}")
        print(f"Valid heights: {np.count_nonzero(np.isfinite(result[..., 0]))}")


if __name__ == "__main__":
    main()