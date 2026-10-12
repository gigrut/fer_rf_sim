import numpy as np
import pandas as pd
import plotly.graph_objects as go
# from skimage import measure
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
import plotly.colors as colors
from FER_sim_parallel_processing_V2 import (
    # physical constants
    a, E_F, phi_t, phi_s,
    FIELD_ENHANCEMENT_FACTOR, Lateral_Confinement_Potential,
    # arrays
    D_values, V_values, A_values, x_values, V_start, V_stop, V_step,
    nD, nV, nA
)
V_values += Lateral_Confinement_Potential

def upsample_current_array(current_array, D_values, V_values, A_values, factor_D=1, factor_V=1, factor_A=1):
    """
    Upsample the current_array along the D, V, and A axes by a given factor.

    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of original D values (length nD)
      V_values: 1D array of original V values (length nV)
      A_values: 1D array of original A values (length nA)
      factor: integer factor to upscale each of the D, V, and A dimensions.
      
    Returns:
      new_D_values: 1D array of upscaled D values (length nD * factor)
      new_V_values: 1D array of upscaled V values (length nV * factor)
      new_A_values: 1D array of upscaled A values (length nA * factor)
      upsampled_array: np.ndarray of shape (nD*factor, nV*factor, nA*factor, 2)
    """
    nD, nV, nA, _ = current_array.shape
    new_nD = nD * factor_D
    new_nV = nV * factor_V
    new_nA = nA * factor_A

    new_D_values = np.linspace(D_values.min(), D_values.max(), new_nD)
    new_V_values = np.linspace(V_values.min(), V_values.max(), new_nV)
    new_A_values = np.linspace(A_values.min(), A_values.max(), new_nA)

    # Create a new grid of points over (D, V, A)
    D_mesh, V_mesh, A_mesh = np.meshgrid(new_D_values, new_V_values, new_A_values, indexing='ij')
    new_points = np.stack([D_mesh, V_mesh, A_mesh], axis=-1)  # shape: (new_nD, new_nV, new_nA, 3)

    upsampled_array = np.empty((new_nD, new_nV, new_nA, 2))
    for ch in range(2):
        # data_slice has shape (nD, nV, nA)
        data_slice = current_array[..., ch]
        interpolator = RegularGridInterpolator((D_values, V_values, A_values),
                                               data_slice,
                                               bounds_error=False,
                                               fill_value=None)
        upsampled_array[..., ch] = interpolator(new_points)
    return new_D_values, new_V_values, new_A_values, upsampled_array

def invert_current_for_constant_I_4D_vectorized(I_target, D_values, M):
    """
    Given a 4D array M of shape (n_D, n_V, n_A, 2) where:
      M[:, j, k, 0] = current I as a function of D (for fixed V and A)
      M[:, j, k, 1] = corresponding dI/dV values
    and given a 1D array D_values (length n_D),
    this function inverts the function I(D; V,A) for each (V,A) pair to obtain the unique D (and dI/dV)
    such that I = I_target.
    
    Returns two 2D arrays (both of shape (n_V, n_A)):
      - D_target: the interpolated D value for which I equals I_target,
      - dIdV_target: the interpolated dI/dV value at that D.
    """
    n_D, n_V, n_A, _ = M.shape
    D_target = np.empty((n_V, n_A))
    dIdV_target = np.empty((n_V, n_A))
    
    # Loop over the independent axes V and A.
    # (Fully vectorizing the inversion across (V,A) is challenging because each column is a separate 1D inversion;
    #  we can vectorize over (V,A) with np.apply_along_axis, but that still has Python-level overhead.
    #  For clarity and robustness, here we loop over j and k.)
    for j in range(n_V):
        for k in range(n_A):
            I_vs_D = M[:, j, k, 0].flatten()
            dIdV_vs_D = M[:, j, k, 1].flatten()
            D_vals = np.asarray(D_values).flatten()
            
            # Ensure that I_vs_D is monotonically increasing with D.
            # If not, reverse the arrays.
            if I_vs_D[0] > I_vs_D[-1]:
                I_vs_D = I_vs_D[::-1]
                D_vals = D_vals[::-1]
                dIdV_vs_D = dIdV_vs_D[::-1]
            
            # Use linear interpolation to invert I -> D.
            f_inv = interp1d(I_vs_D, D_vals, bounds_error=False, fill_value="extrapolate")
            D_est = f_inv(I_target)
            D_target[j, k] = D_est
            
            # Now interpolate dI/dV as a function of D.
            f_dIdV = interp1d(D_vals, dIdV_vs_D, bounds_error=False, fill_value="extrapolate")
            dIdV_target[j, k] = f_dIdV(D_est)
    
    return D_target, dIdV_target

def constant_current_slice(I_target, D_values, M):
    """
    Given the full 4D array M of current data (shape: (n_D, n_V, n_A, 2)) and a target current I_target,
    compute for each (V, A) the corresponding D and dI/dV values along the constant-current contour.
    
    Returns:
      A 3D array of shape (n_V, n_A, 2) where:
         result[j, k, 0] = D (for constant I = I_target at (V[j], A[k]))
         result[j, k, 1] = dI/dV at that D.
    """
    D_target, dIdV_target = invert_current_for_constant_I_4D_vectorized(I_target, D_values, M)
    # Stack the two 2D arrays along a new last axis to form a 3D array.
    result = np.stack((D_target, dIdV_target), axis=-1)
    return result

def build_constant_current_4D(I_target_list, D_values, M):
    """
    For each target constant current in I_target_list, compute the constant-current slice.
    
    Parameters:
      I_target_list: list or array of constant current levels.
      D_values: 1D array of D values (length n_D).
      M: 4D array of shape (n_D, n_V, n_A, 2) containing current and dI/dV.
      
    Returns:
      A 4D array of shape (n_I, n_V, n_A, 2) where the first axis corresponds to the
      different constant current levels in I_target_list.
    """
    slices = []
    for I_target in I_target_list:
        slice_3D = constant_current_slice(I_target, D_values, M)  # shape (n_V, n_A, 2)
        slices.append(slice_3D)
    return np.stack(slices, axis=0)

def get_constant_current_slice(current_array, D_values, I_target, A_index):
    """
    Extract the constant-current slice for a given I_target and fixed A index.
    
    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2) containing [I, dI/dV] data.
      D_values: 1D array of D values corresponding to the nD dimension.
      I_target: The constant current value at which to extract the slice.
      A_index: The index for the A value to fix.
    
    Returns:
      D_slice: 1D array of D values along the constant current contour (length nV).
      dIdV_slice: 1D array of corresponding dI/dV values (length nV).
    """
    # Use your existing function to get the 3D constant-current data
    # resulting in an array of shape (nV, nA, 2) where:
    #   - result[j, k, 0] is D, and
    #   - result[j, k, 1] is dI/dV.
    slice_data = constant_current_slice(I_target, D_values, current_array)
    
    # Extract the slice corresponding to the fixed A value.
    # slice_data_fixed will have shape (nV, 2)
    slice_data_fixed = slice_data[:, A_index, :]
    
    # Return D and dI/dV separately.
    D_slice = slice_data_fixed[:, 0]
    dIdV_slice = slice_data_fixed[:, 1]
    return D_slice, dIdV_slice

def plot_constant_current_slices(current_array, D_values, V_values, I_target_list, A_index):
    """
    Given the full current_array, D_values, and V_values, this helper builds the
    constant current slices (using build_constant_current_4D) for each I in I_target_list,
    fixes the A value (using A_index), and plots a scatter plot with:
      - x-axis: Voltage (V)
      - y-axis: D (from the inverted array)
      - marker color: dI/dV
    """
    print('Plotting D vs V contours')
    # Build the constant current slices. The returned array has shape:
    # (n_I_targets, n_V, n_A, 2) where the last axis contains [D, dI/dV].
    constant_current_slices = build_constant_current_4D(I_target_list, D_values, current_array)
    A_val = A_values[A_index] if 'A_values' in globals() else None
    if A_val is None:
            title_extra = "A value unknown"
    else:
        title_extra = f"A = {A_val:.3e}"

    fig = go.Figure()
    
    for idx, I_target in enumerate(I_target_list):
        # For each constant current level, extract the slice corresponding to the fixed A value.
        # slice_data will have shape (n_V, 2): [:, 0] = D, [:, 1] = dI/dV.
        slice_data = constant_current_slices[idx, :, A_index, :]
        
        fig.add_trace(go.Scatter(
            x=V_values,                   # Voltage axis
            y=slice_data[:, 0],           # D values corresponding to I_target at fixed A
            mode='markers',
            marker=dict(
                size=8,
                color=slice_data[:, 1],   # Color indicates dI/dV
                colorscale='Cividis',
                colorbar=dict(title="dI/dV") if idx == 0 else None,
            ),
            name=f"I = {I_target:.2e}",
            hovertemplate=(
                "V: %{x}<br>" +
                "D: %{y}<br>" +
                "dI/dV: %{marker.color:.3e}<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=f"Constant Current Contours ({title_extra})",
        xaxis=dict(title="Voltage (V)", range=[2, 10]),
        yaxis=dict(title="D (nm)", range=[0, 5]),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(r=150),


    )
    
    fig.show()

def plot_V_dIdV_with_D_multi(current_array, D_values, V_values, A_axis, I_targets, A_values_to_plot):
    """
    Plot a scatter graph with:
      - x-axis: Voltage (V)
      - y-axis: dI/dV
      - Marker color: D
    for multiple constant current values and for multiple A values (passed as actual A values, not indices).
    
    For each A value in A_values_to_plot, the function finds the nearest index in A_axis.
    
    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of D values.
      V_values: 1D array of Voltage values.
      A_axis: 1D array of the A axis values corresponding to the third dimension of current_array.
      I_targets: A list (or array) of constant current values.
      A_values_to_plot: A list (or array) of A values (not indices) to plot.
    """
    print('Plotting dI/dV vs V for various I and A')
    fig = go.Figure()
    
    for I_target in I_targets:
        # Get constant current slice; returns shape (nV, nA, 2)
        slice_data = constant_current_slice(I_target, D_values, current_array)
        
        for A_val in A_values_to_plot:
            # Find the nearest index in A_axis to the desired A value.
            A_index = int(np.argmin(np.abs(A_axis - A_val)))
            
            # Extract the slice for the given A value (via index).
            # slice_data_fixed has shape (nV, 2): column 0 = D, column 1 = dI/dV.
            slice_data_fixed = slice_data[:, A_index, :]
            D_slice = slice_data_fixed[:, 0]
            dIdV_slice = slice_data_fixed[:, 1]
            
            trace_name = f"I = {I_target:.2e}, A = {A_val:.3g}"
            fig.add_trace(go.Scatter(
                x=V_values,
                y=dIdV_slice,
                mode='markers',
                marker=dict(
                    size=8,
                    color=D_slice,
                    colorscale='Cividis',
                    colorbar=dict(title="D") if (I_target == I_targets[0] and A_val == A_values_to_plot[0]) else None
                ),
                name=trace_name,
                hovertemplate=(
                    "V: %{x}<br>" +
                    "dI/dV: %{y}<br>" +
                    "D: %{marker.color:.3e}<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title="V vs. dI/dV with D as Marker Color",
        xaxis_title="Voltage (V)",
        yaxis_title="dI/dV",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(r=150)

    )
    fig.show()

def plot_V_dIdV_with_D_slider_A(current_array, D_values, V_values, A_axis, I_targets, A_values_to_slider):
    """
    Plot a scatter graph (V vs. dI/dV with marker color given by D) for multiple constant current values,
    and use a slider to select the A value.

    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of D values.
      V_values: 1D array of Voltage values.
      A_axis: 1D array of the A axis values corresponding to the third dimension of current_array.
      I_targets: List/array of constant current values.
      A_values_to_slider: List/array of A values (actual values, not indices) for which slider steps are created.
    """
    print('Plotting dI/dV vs V with slider for A')

    traces = []
    # For each A value in the slider, and for each constant current value, create a trace.
    for A_val in A_values_to_slider:
        # Find the nearest index in A_axis.
        A_index = int(np.argmin(np.abs(A_axis - A_val)))
        for I_target in I_targets:
            # Extract the constant current slice (shape: (nV, nA, 2)) for this I_target.
            slice_data = constant_current_slice(I_target, D_values, current_array)
            # Extract data corresponding to the selected A index.
            slice_data_fixed = slice_data[:, A_index, :]  # shape: (nV, 2)
            D_slice = slice_data_fixed[:, 0]
            dIdV_slice = slice_data_fixed[:, 1]

            trace_name = f"I = {I_target:.2e}, A = {A_val:.3g}"
            trace = go.Scatter(
                x=V_values,
                y=dIdV_slice,
                mode='markers',
                marker=dict(
                    size=8,
                    color=D_slice,
                    colorscale='Cividis',
                    # Optionally, show the colorbar only for the first trace.
                    colorbar=dict(title="D")  
                ),
                name=trace_name,
                hovertemplate=(
                    "V: %{x}<br>" +
                    "dI/dV: %{y}<br>" +
                    "D: %{marker.color:.3e}<extra></extra>"
                ),
                visible=False  # all traces start off hidden; slider will control visibility
            )
            traces.append(trace)

    total_traces = len(traces)
    traces_per_A = len(I_targets)
    num_A = len(A_values_to_slider)

    # Build slider steps: each step makes visible all traces corresponding to a given A value.
    steps = []
    for i, A_val in enumerate(A_values_to_slider):
        visible_array = [False] * total_traces
        start_index = i * traces_per_A
        for j in range(traces_per_A):
            visible_array[start_index + j] = True
        step = dict(
            method="update",
            args=[{"visible": visible_array},
                  {"title": f"Scatter Plot (V vs. dI/dV) for A = {A_val:.3g}"}],
            label=f"{A_val:.3g}"
        )
        steps.append(step)

    # Make the traces for the first A value visible by default.
    for j in range(traces_per_A):
        traces[j].visible = True

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "A: "},
        pad={"t": 50},
        steps=steps
    )]

    layout = go.Layout(
        title=f"Scatter Plot (V vs. dI/dV) for A = {A_values_to_slider[0]:.3g}",
        xaxis_title="Voltage (V)",
        yaxis_title="dI/dV",
        sliders=sliders,
        template="plotly_white"
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_heatmap_dIdV(current_array, D_values, V_values, A_values, I_target):
    """
    Create a heatmap plot for a constant current slice.
    
    For a given constant current I_target, this function extracts the constant-current slice
    (with shape (n_V, n_A, 2)) using constant_current_slice, then plots a heatmap where:
      - x-axis: Voltage (V_values)
      - y-axis: A (A_values)
      - Color (z): dI/dV values (extracted from the slice and transposed to shape (n_A, n_V))
    
    Parameters:
      current_array: np.ndarray of shape (n_D, n_V, n_A, 2) containing [I, dI/dV] data.
      D_values: 1D array of D values.
      V_values: 1D array of Voltage values.
      A_values: 1D array of A values.
      I_target: The constant current level to extract.
    """
    print('Plotting dI/dV heatmap')

    # Extract constant current slice (shape: (n_V, n_A, 2))
    slice_data = constant_current_slice(I_target, D_values, current_array)
    
    # Extract dI/dV from the slice. This has shape (n_V, n_A).
    dIdV = slice_data[:, :, 1]
    
    # For a heatmap with x=V and y=A, we need the z array to be shape (len(A_values), len(V_values)).
    # Transpose dIdV so that rows correspond to A.
    z_data = dIdV.T

    # Create the heatmap.
    fig = go.Figure(data=go.Heatmap(
        x=V_values,
        y=A_values,
        z=z_data,
        colorscale='Cividis',
        colorbar=dict(title="dI/dV", x=1.05)
    ))
    
    fig.update_layout(
        title=f"Heatmap of dI/dV for Constant I = {I_target:.2e}",
        xaxis_title="Voltage (V)",
        yaxis_title="A",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(r=150)
    )
    
    fig.show()

def plot_heatmap_slider(current_array, D_values, V_values, A_values, I_targets):
    """
    Create a heatmap plot with a slider. For each constant current (I_target) in I_targets:
      - Extract the constant-current slice (using constant_current_slice, which returns an array of shape (nV, nA, 2))
      - Use V_values as the x-axis and A_values as the y-axis.
      - Normalize the dI/dV data by dividing by the maximum value in that slice.
      - Create a heatmap of the normalized dI/dV.
      
    A slider is added so that the user can change which constant current slice is visible.
    
    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of D values (used for inversion)
      V_values: 1D array of Voltage values.
      A_values: 1D array of A values.
      I_targets: List or array of constant current values to generate slices for.
    """
    print('Plotting dI/dV heatmap with slider')
    traces = []
    for I_target in I_targets:
        # Extract the constant current slice; shape is (nV, nA, 2)
        slice_data = constant_current_slice(I_target, D_values, current_array)
        if slice_data is None or slice_data.size == 0:
            print(f"Warning: constant_current_slice returned empty data for I_target={I_target}")
            continue
        # Extract the dI/dV data (second channel); shape (nV, nA)
        dIdV = slice_data[:, :, 1]
        # Normalize by the max within this heatmap. Avoid division by zero.
        max_val = np.nanmax(dIdV)
        norm_dIdV = dIdV / max_val if max_val != 0 else dIdV

        # Transpose so that rows correspond to A (y-axis) and columns to V (x-axis)
        heatmap = go.Heatmap(
            x=V_values,
            y=A_values,
            z=norm_dIdV.T,
            colorscale='Cividis',
            zmin=0,
            zmax=1,
            visible=False,
            colorbar=dict(title="Normalized dI/dV", x=1.05)
        )
        traces.append(heatmap)
        print(f"Added heatmap for I_target={I_target:.2g} with shape {norm_dIdV.T.shape}")

    # Ensure at least one trace is added.
    if not traces:
        print("No traces to plot!")
        return

    # Make the first trace visible by default.
    traces[0].visible = True

    # Create slider steps.
    steps = []
    for i, I_target in enumerate(I_targets):
        # Create a visibility array with one True for the current step and False for the rest.
        visible = [False] * len(traces)
        visible[i] = True
        step = dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"Heatmap for I = {I_target:.2e}"}],
            label=f"{I_target:.2e}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Constant Current: "},
        pad={"t": 50},
        steps=steps
    )]

    layout = go.Layout(
        title=f"Heatmap for I = {I_targets[0]:.2e}",
        xaxis_title="Voltage (V)",
        yaxis_title="A",
        sliders=sliders,
        template="plotly_white",
        margin=dict(r=150)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_heatmap_slider_constant_D(current_array, D_values, V_values, A_values, D_targets):
    """
    Create a heatmap plot with a slider for constant D slices, using a logarithmic color scale.
    Ensures that dI/dV is being plotted, not current.

    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2) containing [I, dI/dV] data.
      D_values: 1D array of D values.
      V_values: 1D array of Voltage values.
      A_values: 1D array of A values.
      D_targets: List or array of D values to extract constant-D slices.
    """
    print('Plotting dI/dV heatmap for constant D with slider (log scale)')

    traces = []
    for D_target in D_targets:
        # Find the closest index in D_values
        D_index = np.argmin(np.abs(D_values - D_target))

        # Extract the slice at this D_index
        slice_data = current_array[D_index, :, :, 1]  # ✅ dI/dV values, shape (nV, nA)

        # Ensure all values are positive before taking log (small offset to avoid log(0) issues)
        slice_data = np.abs(slice_data) + 1e-10

        # Transpose so that rows correspond to A (y-axis) and columns to V (x-axis)
        heatmap = go.Heatmap(
            x=V_values,
            y=A_values,
            z=np.log10(np.gradient(slice_data.T, V_step, axis=-1)),  # ✅ Applying log scale to dI/dV
            colorscale='Cividis',
            colorbar=dict(
                title="log(dI/dV)",  # ✅ Updating colorbar title to dI/dV
                x=1.05
            ),
            visible=False
        )
        traces.append(heatmap)
        print(f"Added heatmap for D = {D_target:.3g} with shape {slice_data.T.shape}")

    # Make the first trace visible by default.
    if traces:
        traces[0].visible = True
    else:
        print("No valid traces to plot.")
        return

    # Create slider steps
    steps = []
    for i, D_target in enumerate(D_targets):
        visible = [False] * len(traces)
        visible[i] = True
        step = dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"Heatmap for D = {D_target:.3g}"}],
            label=f"{D_target:.3g}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Constant D: "},
        pad={"t": 50},
        steps=steps
    )]

    layout = go.Layout(
        title=f"Heatmap for D = {D_targets[0]:.3g}",
        xaxis_title="Voltage (V)",
        yaxis_title="A",
        sliders=sliders,
        template="plotly_white",
        margin=dict(r=150)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()




def save_data_files(current_array, D_values, V_values, A_values, base_filename="FER_data"):
    """
    Save the current_array and axis arrays in two formats:
    
    1. As separate .npy files (or an .npz archive) preserving the full shape.
    2. As a CSV file containing columns for D, V, A, I, and dI/dV.
    
    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of D axis values.
      V_values: 1D array of V axis values.
      A_values: 1D array of A axis values.
      base_filename: Base file name (without extension) for saving files.
    """
    
    # ------------------------------------
    # 1. Save as .npy files (or .npz archive)
    # ------------------------------------
    # Option A: Save as separate .npy files
    np.save(base_filename + "_current_array.npy", current_array)
    np.save(base_filename + "_D_values.npy", D_values)
    np.save(base_filename + "_V_values.npy", V_values)
    np.save(base_filename + "_A_values.npy", A_values)
    
    # Option B (alternative): Save everything in one .npz archive.
    # np.savez(base_filename + ".npz", current_array=current_array,
    #          D_values=D_values, V_values=V_values, A_values=A_values)
    
    # ------------------------------------
    # 2. Save as a CSV file.
    # ------------------------------------
    # For the CSV, we need to flatten the 4D array.
    # Assume current_array shape is (nD, nV, nA, 2), where channel 0 is I and channel 1 is dI/dV.
    nD, nV, nA, _ = current_array.shape
    
    # Create a meshgrid for the axes (using 'ij' indexing so the shapes match current_array axes).
    D_grid, V_grid, A_grid = np.meshgrid(D_values, V_values, A_values, indexing="ij")
    
    # Flatten all arrays.
    D_flat = D_grid.flatten()
    V_flat = V_grid.flatten()
    A_flat = A_grid.flatten()
    I_flat = current_array[..., 0].flatten()
    dIdV_flat = current_array[..., 1].flatten()
    
    # Create a DataFrame.
    df = pd.DataFrame({
        "D": D_flat,
        "V": V_flat,
        "A": A_flat,
        "I": I_flat,
        "dI/dV": dIdV_flat
    })
    
    # Save the DataFrame to a CSV file.
    df.to_csv(base_filename + "_data.csv", index=False)
    
    print("Data saved as .npy and CSV files.")

def save_npz_archive(current_array, D_values, V_values, A_values, filename="FER_full_dataset.npz"):
    """
    Save the current_array and axis arrays (D_values, V_values, A_values) into a single .npz archive.

    Parameters:
      current_array: np.ndarray of shape (nD, nV, nA, 2)
      D_values: 1D array of D axis values.
      V_values: 1D array of V axis values.
      A_values: 1D array of A axis values.
      filename: The name of the .npz file to save.
    """
    import numpy as np
    np.savez(filename,
             current_array=current_array,
             D_values=D_values,
             V_values=V_values,
             A_values=A_values)
    print(f"Data saved in {filename}")



def main():
  # Load and prepare the current_array
  current_array_path = "C:/Users/willh/countdown/current_array.npy"
  current_array = np.load(current_array_path)
  current_array = np.stack([current_array, np.gradient(current_array, V_step, axis=1)], axis=-1)
  D_vals, V_vals, A_vals, current_array_upscaled = upsample_current_array(current_array, D_values, V_values, A_values, factor_D=1, factor_V=2, factor_A=4)
  
  A_index = 1
  I_target_list = np.logspace(-22, -6, 100)  # Adjust these values as needed

  for A_val in A_values:
    plot_constant_current_slices(current_array_upscaled, D_vals, V_vals, I_target_list, A_val)

  #plot_V_dIdV_with_D_multi(current_array_upscaled, D_vals, V_vals, A_vals, I_target_list, A_values_to_plot=[0, 0.5, 1])
  #plot_V_dIdV_with_D_slider_A(current_array_upscaled, D_vals, V_vals, A_vals, [1e-10], A_values_to_slider=A_vals)

  # for I_target in I_target_list:
  #   plot_heatmap_dIdV(current_array_upscaled, D_vals, V_vals, A_vals, I_target)

  #plot_heatmap_slider(current_array_upscaled, D_vals, V_vals, A_vals, I_target_list)

  #plot_heatmap_slider_constant_D(current_array_upscaled, D_values, V_values/2, A_values/4, D_values[::4])


  save_data_files(current_array, D_values, V_values, A_values, base_filename="FER_full_dataset")
  save_npz_archive(current_array, D_values, V_values, A_values, filename="FER_full_dataset.npz")



if __name__ == "__main__":
  main()
  print('done')