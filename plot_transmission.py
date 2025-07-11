import numpy as np
from FER_plotter import load_param_arrays_from_config, plot_T_slice_3d, find_latest_lut
from scipy.interpolate import RegularGridInterpolator

# --- User-editable section ---
# Choose axes and constant for the plot
x_axis = 'V'              # X axis: 'Z', 'V', 'A', or 'x'
y_axis = 'A'              # Y axis: 'Z', 'V', 'A', or 'x'
const_axis = 'Z'          # Constant axis: 'Z', 'V', 'A', or 'x'
const_value = 2.0         # Value for the constant axis (nm, V, or A)
component = 'magnitude'   # 'magnitude', 'real', 'imaginary', 'phase'

# --- Load parameter arrays from config ---
Zg, Vg, Ag, Eg = load_param_arrays_from_config()

# --- Load LUT and build interpolator ---
lut_path = find_latest_lut()
import h5py
with h5py.File(lut_path, 'r') as f:
    LUT = f['D'][...]
    Z_values = f['z'][...]
    V_values = f['V'][...]
    E_values = f['E'][...]
# Chebyshev nodes for RF averaging
n_cheb = 16
nodes = np.pi * (2 * np.arange(1, n_cheb + 1) - 1) / (2 * n_cheb)
interp = RegularGridInterpolator((Z_values, V_values, E_values), LUT, bounds_error=False, fill_value=0.0)

# --- Build T_array (nZ, nV, nA, nE) ---
def _T_grid(interp, Egrid, Vgrid, Agrid, Z_val, nodes):
    nA, nE, nV, nN = len(Agrid), len(Egrid), len(Vgrid), len(nodes)
    Vdc_grid = Vgrid[None, None, :, None]
    Arf_grid = Agrid[:, None, None, None]
    phase    = np.cos(nodes)[None, None, None, :]
    V_inst   = Vdc_grid + Arf_grid * phase
    E_grid   = Egrid[None, :, None, None]
    V_b      = np.broadcast_to(V_inst, (nA, nE, nV, nN))
    E_b      = np.broadcast_to(E_grid, (nA, nE, nV, nN))
    pts = np.column_stack([
        np.full(V_b.size, Z_val),
        V_b.ravel(),
        E_b.ravel()
    ])
    T = interp(pts).reshape(nA, nE, nV, nN).mean(axis=3)
    return T

# Compute T_array for all Z, V, A, E
T_array = _T_grid(interp, Eg, Vg, Ag, Zg[0], nodes)
if len(Zg) > 1:
    T_array = np.stack([_T_grid(interp, Eg, Vg, Ag, z, nodes) for z in Zg], axis=0)
else:
    T_array = T_array[None, ...]
T_array = np.moveaxis(T_array, [1, 2], [3, 1])  # (nZ, nV, nA, nE)

# --- Plot ---
plot_T_slice_3d(T_array, Zg, Vg, Ag, Eg, x_axis, y_axis, const_axis, const_value, component) 