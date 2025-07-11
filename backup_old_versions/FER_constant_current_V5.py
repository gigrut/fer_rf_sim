import numpy as np
import plotly.graph_objects as go
# from skimage import measure
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
import plotly.colors as colors
from FER_sim_parallel_processing import (
    # physical constants
    hbar, e, m, a, E_F, phi_t, phi_s,
    FIELD_ENHANCEMENT_FACTOR, Junction_Resistance,
    # arrays
    D_values, V_values, A_values, x_values, V_start, V_stop, V_step
)



def interpolate_current_array(current_array, D_values, V_values, upsample_factor=4, savgol_window=7, savgol_polyorder=2):
    """
    Interpolate the 3D current_array over the D and V dimensions while preserving the A dimension,
    and apply Savitzky-Golay filtering to smooth the interpolated data.

    Parameters:
    - current_array: 3D NumPy array of current values (shape: len(D_values) x len(V_values) x len(A_values)).
    - D_values: 1D array of original D values.
    - V_values: 1D array of original V values.
    - upsample_factor: Factor by which to increase the resolution of D and V dimensions.
    - savgol_window: Window size for the Savitzky-Golay filter (must be odd). Defaults to 7.
    - savgol_polyorder: Polynomial order for the Savitzky-Golay filter. Defaults to 2.

    Returns:
    - D_fine: 1D array of interpolated D values.
    - V_fine: 1D array of interpolated V values.
    - current_fine_smoothed: 3D NumPy array of interpolated and smoothed current values.
    """
    # Create finer grids for D and V
    D_fine = np.linspace(D_values.min(), D_values.max(), len(D_values) * upsample_factor)
    V_fine = np.linspace(V_values.min(), V_values.max(), len(V_values) * upsample_factor)
    D_fine_grid, V_fine_grid = np.meshgrid(D_fine, V_fine, indexing='ij')

    # Prepare the interpolated current array
    interpolator = RegularGridInterpolator((D_values, V_values), current_array, method='linear', bounds_error=False, fill_value=None)
    fine_points = np.array([D_fine_grid.flatten(), V_fine_grid.flatten()]).T
    current_fine = interpolator(fine_points).reshape(len(D_fine), len(V_fine), current_array.shape[2])

    # Apply Savitzky-Golay filter to smooth along the V-axis (axis=1)
    current_fine_smoothed = np.empty_like(current_fine)
    for a_idx in range(current_array.shape[2]):
        current_fine_smoothed[:, :, a_idx] = np.apply_along_axis(
            lambda x: savgol_filter(x, window_length=savgol_window, polyorder=savgol_polyorder, mode='interp'),
            axis=1,
            arr=current_fine[:, :, a_idx]
        )

    return D_fine, V_fine, current_fine_smoothed

def plot_constant_current_scatter_plotly(current_array, D_values, V_values, A_values, A_target=0, current_levels=None, cmap='Cividis'):
    """
    Plot scatter contours of constant current on the D-V plane for a specified A value using Plotly.

    Parameters:
    - current_array: 3D NumPy array of current values with shape (len(D_values), len(V_values), len(A_values)).
    - D_values: 1D array of tip heights (D).
    - V_values: 1D array of bias voltages (V).
    - A_values: 1D array of amplitude values (A).
    - A_target: The specific A value to plot. If not found exactly, the closest value is used. Defaults to 0.
    - current_levels: List or array of current levels to plot. If None, generates 5 log-spaced levels between min and max current. Defaults to None.
    - cmap: Colormap name for the contours. Defaults to 'Viridis'.
    
    Raises:
    - ValueError: If no positive current values are found for the specified A.
    """
    # Validate input dimensions
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")

    # Find the index closest to A_target
    A_diff = np.abs(A_values - A_target)
    A_index = A_diff.argmin()
    closest_A = A_values[A_index]
    if not np.isclose(closest_A, A_target, atol=1e-8):
        print(f"Warning: A_target={A_target} not found. Using closest value A={closest_A}.")

    # Extract current_slice for A=closest_A
    current_slice = current_array[:, :, A_index]  # Shape: (len(D_values), len(V_values))

    # Ensure there are positive current values for contour extraction
    positive_mask = current_slice > 0
    if not np.any(positive_mask):
        raise ValueError("No positive current values found for the specified A.")

    # Determine current levels
    if current_levels is None:
        current_min = current_slice[positive_mask].min()
        current_max = current_slice.max()
        current_levels = np.logspace(np.log10(current_min), np.log10(current_max), num=5)
        print(f"Info: Generated {len(current_levels)} log-spaced current levels between {current_min:.2e} A and {current_max:.2e} A.")
    else:
        # Ensure provided levels are within the data range
        current_min = current_slice[positive_mask].min()
        current_max = current_slice.max()
        current_levels = np.array(current_levels)
        out_of_bounds = (current_levels < current_min) | (current_levels > current_max)
        if np.any(out_of_bounds):
            valid_levels = current_levels[~out_of_bounds]
            invalid_levels = current_levels[out_of_bounds]
            if len(valid_levels) == 0:
                raise ValueError("All provided current_levels are outside the range of current data.")
            print(f"Warning: The following current_levels are outside the data range and will be ignored: {invalid_levels}")
            current_levels = valid_levels
        current_levels = np.unique(current_levels)  # Remove duplicates if any

    # Initialize a Plotly figure
    fig = go.Figure()

    # Generate a list of colors for the contour levels
    # Sample colors from the specified colormap
    sampled_colors = colors.sample_colorscale(cmap, np.linspace(0, 1, len(current_levels)))

    # Loop through each current level and find contours
    for i, (level, color) in enumerate(zip(current_levels, sampled_colors)):
        # Find contours at the specified current level
        contours = measure.find_contours(current_slice, level=level)

        if not contours:
            print(f"Warning: No contours found for level {level:.2e} A.")
            continue  # Skip to the next level if no contours are found

        # Loop through each contour found at this level
        for contour in contours:
            # contour is an (N, 2) array of (row, col) coordinates

            # Convert row, col indices to D, V values
            # row corresponds to D (vertical axis), col corresponds to V (horizontal axis)
            # Assuming D_values and V_values are sorted and evenly spaced
            # If not evenly spaced, interpolation is needed

            # Calculate the step sizes
            if len(D_values) > 1:
                D_step = (D_values[-1] - D_values[0]) / (len(D_values) - 1)
            else:
                D_step = 1
            if len(V_values) > 1:
                V_step = (V_values[-1] - V_values[0]) / (len(V_values) - 1)
            else:
                V_step = 1

            # Convert contour coordinates to actual D and V values
            D_contour = D_values[0] + contour[:, 0] * D_step
            V_contour = V_values[0] + contour[:, 1] * V_step

            # Add scatter trace for this contour
            fig.add_trace(go.Scatter(
                x=V_contour,
                y=D_contour,
                mode='markers',
                name=f'Current = {level:.2e} A',
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=color,  # Assign unique color per level
                    line=dict(width=0)  # No border
                ),
                hoverinfo='text',
                text=[f'V: {v:.2f} V<br>D: {d:.2f} nm<br>Current: {level:.2e} A' for v, d in zip(V_contour, D_contour)]
            ))

    # Create a colorbar manually using a dummy scatter trace
    # Plotly does not support multiple colorbars, so we create one using a separate trace
    # This is a workaround to display the current levels with their corresponding colors
    # The dummy trace will not be visible but will display the colorbar

    # Create a dummy scatter for the colorbar
    if len(current_levels) > 0:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=cmap,
                showscale=True,
                cmin=current_min,
                cmax=current_max,
                color=current_levels,
                colorbar=dict(
                    title='Current (A)',
                    tickmode='array',
                    tickvals=current_levels,
                    ticktext=[f"{level:.2e}" for level in current_levels],
                    lenmode='fraction',
                    len=0.75,
                    y=0.5
                )
            ),
            showlegend=False
        ))

    # Update layout for better aesthetics
    fig.update_layout(
        title=f'Constant Current Contours (A={closest_A} V)',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='Tip Height (D) [nm]',
        template='plotly_white',
        width=800,
        height=600
    )

    # Show the plot
    fig.show()

def extract_constant_current_contours_with_dIdV(current_array, D_values, V_values, A_values, 
                                                A_target=0, current_levels=None, delta_factor=0.01):
    """
    Extract contours of constant current and compute dI/dV along each contour.
    
    Parameters:
    - current_array: 3D NumPy array with shape (len(D_values), len(V_values), len(A_values)).
    - D_values: 1D array of tip heights (D).
    - V_values: 1D array of bias voltages (V).
    - A_values: 1D array of amplitude values (A).
    - A_target: The specific A value to plot. Defaults to 0.
    - current_levels: List or array of current levels to plot. If None, generates 5 log-spaced levels.
    - delta_factor: Fraction of V range to use as delta V for derivative calculation. Defaults to 1%.
    
    Returns:
    - contours_with_dIdV: List of dictionaries, each containing:
        - 'level': Current level.
        - 'D': 1D array of D values along the contour.
        - 'V': 1D array of V values along the contour.
        - 'dI_dV': 1D array of dI/dV values along the contour.
    """
    # Validate input dimensions
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")
    
    # Find the index closest to A_target
    A_diff = np.abs(A_values - A_target)
    A_index = A_diff.argmin()
    closest_A = A_values[A_index]
    if not np.isclose(closest_A, A_target, atol=1e-8):
        print(f"Warning: A_target={A_target} not found. Using closest value A={closest_A}.")
    
    # Extract current_slice for A=closest_A
    current_slice = current_array[:, :, A_index]  # Shape: (len(D_values), len(V_values))
    
    # Ensure there are positive current values for contour extraction
    positive_mask = current_slice > 0
    if not np.any(positive_mask):
        raise ValueError("No positive current values found for the specified A.")
    
    # Determine current levels
    if current_levels is None:
        current_min = current_slice[positive_mask].min()
        current_max = current_slice.max()
        current_levels = np.logspace(np.log10(current_min), np.log10(current_max), num=5)
        print(f"Info: Generated {len(current_levels)} log-spaced current levels between {current_min:.2e} A and {current_max:.2e} A.")
    else:
        # Ensure provided levels are within the data range
        current_min = current_slice[positive_mask].min()
        current_max = current_slice.max()
        current_levels = np.array(current_levels)
        out_of_bounds = (current_levels < current_min) | (current_levels > current_max)
        if np.any(out_of_bounds):
            valid_levels = current_levels[~out_of_bounds]
            invalid_levels = current_levels[out_of_bounds]
            if len(valid_levels) == 0:
                raise ValueError("All provided current_levels are outside the range of current data.")
            print(f"Warning: The following current_levels are outside the data range and will be ignored: {invalid_levels}")
            current_levels = valid_levels
        current_levels = np.unique(current_levels)  # Remove duplicates if any
    
    # Calculate delta V based on the V_values spacing
    if len(V_values) > 1:
        V_steps = np.diff(V_values)
        avg_delta_V = np.mean(V_steps)
    else:
        avg_delta_V = 1  # Arbitrary value if only one V_value exists
    
    delta_V = avg_delta_V * delta_factor  # e.g., 1% of V range
    
    # Initialize list to hold contour data with dI/dV
    contours_with_dIdV = []
    
    # Loop through each current level and find contours
    for i, level in enumerate(current_levels):
        # Find contours at the specified current level
        contours = measure.find_contours(current_slice, level=level)
        
        if not contours:
            print(f"Warning: No contours found for level {level:.2e} A.")
            continue  # Skip to the next level if no contours are found
        
        # Loop through each contour found at this level
        for contour in contours:
            # contour is an (N, 2) array of (row, col) coordinates
            
            # Convert row, col indices to D, V values
            # row corresponds to D (vertical axis), col corresponds to V (horizontal axis)
            # Assuming D_values and V_values are sorted and evenly spaced
            # If not evenly spaced, interpolation is needed
            
            # Calculate the step sizes
            if len(D_values) > 1:
                D_step = (D_values[-1] - D_values[0]) / (len(D_values) - 1)
            else:
                D_step = 1
            if len(V_values) > 1:
                V_step = (V_values[-1] - V_values[0]) / (len(V_values) - 1)
            else:
                V_step = 1
            
            # Convert contour coordinates to actual D and V values
            D_contour = D_values[0] + contour[:, 0] * D_step
            V_contour = V_values[0] + contour[:, 1] * V_step
            
            # Initialize list to hold dI/dV values
            dI_dV_contour = []

            for D, V in zip(D_contour, V_contour):
                try:
                    # Find the nearest D index
                    D_idx = np.searchsorted(D_values, D)
                    D_idx = np.clip(D_idx, 0, len(D_values) - 1)
                    
                    # Extract I values for central difference calculation
                    V_idx = np.searchsorted(V_values, V)
                    V_idx = np.clip(V_idx, 1, len(V_values) - 2)  # Ensure we have neighbors for central difference
                    
                    # Values on either side of V
                    V_left = V_values[V_idx - 1]
                    V_right = V_values[V_idx + 1]
                    I_left = current_slice[D_idx, V_idx - 1]
                    I_right = current_slice[D_idx, V_idx + 1]
                    
                    # Compute dI/dV using central difference
                    dI_dV = (I_right - I_left) / (V_right - V_left)
                except Exception as e:
                    print(f"Warning: Derivative calculation failed at D={D}, V={V}. Setting dI/dV to NaN.")
                    dI_dV = np.nan
                
                dI_dV_contour.append(dI_dV)
            
            # Convert to NumPy arrays
            D_contour = np.array(D_contour)
            V_contour = np.array(V_contour)
            dI_dV_contour = np.array(dI_dV_contour)
            
            # Append to the list
            contours_with_dIdV.append({
                'level': level,
                'D': D_contour,
                'V': V_contour,
                'dI_dV': dI_dV_contour
            })
    
    print(f"Info: Extracted {len(contours_with_dIdV)} contours with dI/dV.")
    return contours_with_dIdV

def plot_contours_scatter_with_normalized_dIdV(contours_with_dIdV, cmap='Cividis'):
    """
    Plot scatter contours of constant current on the D-V plane, colored by normalized dI/dV.
    
    Parameters:
    - contours_with_dIdV: List of dictionaries containing 'level', 'D', 'V', 'dI_dV'.
    - cmap: Plotly diverging color scale name. Defaults to 'Cividis'.
    
    Returns:
    - fig: Plotly Figure object.
    """
    import plotly.graph_objects as go
    import plotly.colors as colors
    import numpy as np

    # Initialize a Plotly figure
    fig = go.Figure()
    
    for contour in contours_with_dIdV:
        level = contour['level']
        D = contour['D']
        V = contour['V']
        dI_dV = contour['dI_dV']
        
        # Remove NaN values
        valid_mask = np.isfinite(dI_dV)
        D = D[valid_mask]
        V = V[valid_mask]
        dI_dV = dI_dV[valid_mask]
        
        if len(V) == 0:
            print(f"Warning: No valid dI/dV values for level {level:.2e} A.")
            continue
        
        # Normalize dI/dV for this contour
        max_dIdV = np.max(dI_dV)
        if max_dIdV == 0:
            print(f"Warning: Maximum dI/dV is zero for level {level:.2e} A. Skipping normalization.")
            continue
        normalized_dIdV = (dI_dV / max_dIdV) * 100  # Convert to percentage
        
        fig.add_trace(go.Scatter(
            x=V,
            y=D,
            mode='markers',
            name=f'Current = {level:.2e} A',
            marker=dict(
                size=4,
                color=normalized_dIdV,  # Color points by normalized dI/dV
                colorscale=cmap,
                cmin=0,  # Start the color scale at 0%
                cmax=100,  # End the color scale at 100%
                opacity=0.7,
                colorbar=dict(
                    title='Normalized dI/dV (%)',
                    # titleside='right',
                    tickvals=[0, 50, 100],
                    ticktext=['0%', '50%', '100%']
                )
            ),
            hoverinfo='text',
            text=[f'V: {v:.2f} V<br>D: {d:.2f} nm<br>dI/dV: {didv:.2e} A/V<br>Norm: {norm:.1f} %'
                  for v, d, didv, norm in zip(V, D, dI_dV, normalized_dIdV)]
        ))
    
    # Update layout for better aesthetics
    fig.update_layout(
        title='Constant Current Contours Colored by Normalized dI/dV (A=0)',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='Tip Height (D) [nm]',
        template='plotly_white',
        width=800,
        height=600
    )
    
    fig.show()
    return fig

def plot_dIdV_vs_V(contours_with_dIdV, cmap='Cividis'):
    """
    Plot dI/dV vs V for each constant current contour.
    
    Parameters:
    - contours_with_dIdV: List of dictionaries containing 'level', 'D', 'V', 'dI_dV'.
    - cmap: Plotly color scale name for differentiating contours. Defaults to 'Viridis'.
    
    Returns:
    - fig: Plotly Figure object.
    """
    # Initialize a Plotly figure
    fig = go.Figure()
    
    # Generate a list of colors for the contours
    num_contours = len(contours_with_dIdV)
    if num_contours == 0:
        raise ValueError("No contours with dI/dV data to plot.")
    sampled_colors = colors.sample_colorscale(cmap, np.linspace(0, 1, num_contours))
    
    for i, contour in enumerate(contours_with_dIdV):
        level = contour['level']
        V = contour['V']
        dI_dV = contour['dI_dV']
        
        # Remove NaN values
        valid_mask = np.isfinite(dI_dV)
        V = V[valid_mask]
        dI_dV = dI_dV[valid_mask]
        
        if len(V) == 0:
            print(f"Warning: No valid dI/dV values for level {level:.2e} A.")
            continue
        
        fig.add_trace(go.Scatter(
            x=V,
            y=dI_dV,
            mode='lines+markers',
            name=f'Current = {level:.2e} A',
            line=dict(color=sampled_colors[i], width=2),
            marker=dict(
                size=4,
                color=sampled_colors[i],
                opacity=0.7
            ),
            hoverinfo='text',
            text=[f'V: {v:.2f} V<br>dI/dV: {didv:.2e} A/V' 
                  for v, didv in zip(V, dI_dV)]
        ))
    
    # Update layout for better aesthetics
    fig.update_layout(
        title='dI/dV vs Bias Voltage (V) for Constant Current Contours (A=0)',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='dI/dV (A/V)',
        xaxis=dict(range=[V_start, V_stop]),
        template='plotly_white',
        width=800,
        height=600
    )
    
    fig.show()
    return fig

def simulate_feedback_loop(current_array, tip_heights, voltages, amplitudes, 
                           target_current=1e-3, voltage_step=0.01, target_amplitude=0):
    """
    Simulate a feedback loop to adjust tip height for a constant current as voltage changes.

    Parameters:
    - current_array: 3D NumPy array of interpolated current values (D x V x A).
    - tip_heights: 1D array of interpolated tip heights (D).
    - voltages: 1D array of interpolated bias voltages (V).
    - amplitudes: 1D array of amplitude values (A).
    - target_current: Desired constant current to maintain.
    - voltage_step: Step size for bias voltage increments.
    - target_amplitude: Specific amplitude to use for the simulation.

    Returns:
    - feedback_voltages: 1D array of voltages.
    - feedback_heights: 1D array of corresponding tip heights to maintain the target current.
    - feedback_dI_dV: 1D array of dI/dV values along the feedback path.
    """

    # Locate the slice for the target amplitude
    amplitude_index = np.searchsorted(amplitudes, target_amplitude, side='right') - 1
    current_grid = current_array[:, :, amplitude_index]

    # Determine valid voltage range for the target current
    valid_voltages = []
    for v_idx in range(current_grid.shape[1]):
        current_column = current_grid[:, v_idx]
        if np.min(current_column) <= target_current <= np.max(current_column):
            valid_voltages.append(voltages[v_idx])

    if len(valid_voltages) == 0:
        raise ValueError(f"Target current {target_current} does not appear in the provided data.")

    start_voltage = np.min(valid_voltages)
    end_voltage = np.max(valid_voltages)

    # Initialize outputs
    feedback_voltages = []
    feedback_heights = []
    feedback_dI_dV = []

    # Start with the initial voltage
    current_voltage = start_voltage

    # Interpolate the current grid across voltages for smooth voltage resolution
    interp_current_grid = RectBivariateSpline(tip_heights, voltages, current_grid)

    # Find the initial tip height for the target current
    current_at_voltage = interp_current_grid(tip_heights, current_voltage, grid=False)
    interp_height = interp1d(current_at_voltage, tip_heights, kind='linear', bounds_error=False, fill_value="extrapolate")
    current_height = interp_height(target_current)

    feedback_voltages.append(current_voltage)
    feedback_heights.append(current_height)

    # Iterate through voltages
    while current_voltage < end_voltage:
        next_voltage = current_voltage + voltage_step
        if next_voltage > voltages[-1]:
            break

        # Interpolate current at the next voltage
        current_at_next_voltage = interp_current_grid(tip_heights, next_voltage, grid=False)
        interp_height_next = interp1d(current_at_next_voltage, tip_heights, kind='linear', bounds_error=False, fill_value="extrapolate")
        next_height = interp_height_next(target_current)

        # Compute dI/dV using central difference
        current_plus = interp_current_grid(current_height, current_voltage + voltage_step, grid=False)
        current_minus = interp_current_grid(current_height, current_voltage - voltage_step, grid=False)
        dI_dV = (current_plus - current_minus) / (2 * voltage_step)

        feedback_dI_dV.append(dI_dV)

        # Append results to the feedback arrays
        feedback_voltages.append(next_voltage)
        feedback_heights.append(next_height)

        current_voltage = next_voltage
        current_height = next_height

    # Adjust sizes to match
    feedback_voltages = feedback_voltages[:-1]  # Trim last element to match dI/dV
    feedback_heights = feedback_heights[:-1]    # Trim last element to match dI/dV

    return np.array(feedback_voltages), np.array(feedback_heights), np.array(feedback_dI_dV)

def combine_contours_for_all_levels(all_V_feedback, all_D_feedback, all_dI_dV_feedback, all_I_targets):
    """
    Combine data for all current levels into a single contours_with_dIdV list.

    Parameters:
    - all_V_feedback: List of voltage arrays for all current levels.
    - all_D_feedback: List of tip height arrays for all current levels.
    - all_dI_dV_feedback: List of dI/dV arrays for all current levels.
    - all_I_targets: List of current levels.

    Returns:
    - contours_with_dIdV: List of dictionaries with combined data for all levels.
    """
    contours_with_dIdV = []
    for V_fb, D_fb, dIdV_fb, I_target in zip(all_V_feedback, all_D_feedback, all_dI_dV_feedback, all_I_targets):
        contours_with_dIdV.append({
            'level': I_target,
            'D': D_fb,
            'V': V_fb,
            'dI_dV': dIdV_fb
        })
    return contours_with_dIdV

def simulate_feedback_for_all_A(current_array, tip_heights, voltages, amplitudes, target_current, voltage_step=0.01):
    """
    Simulate feedback loop for every value of A in amplitudes.
    
    Parameters:
    - current_array: 3D NumPy array (D x V x A) of current values.
    - tip_heights: 1D array of interpolated tip heights (D).
    - voltages: 1D array of interpolated bias voltages (V).
    - amplitudes: 1D array of amplitude values (A).
    - target_current: Desired constant current to maintain.
    - voltage_step: Step size for voltage increments.
    
    Returns:
    - V_grid: 1D array of shared voltages across all amplitudes.
    - A_grid: 1D array of amplitudes.
    - dIdV_grid: 2D array of dI/dV values (rows: A, columns: V).
    """
    all_dIdV = []
    shared_voltages = None  # Shared V grid

    for target_amplitude in amplitudes:
        # Simulate feedback loop for the given amplitude
        V_feedback, _, dI_dV_feedback = simulate_feedback_loop(
            current_array=current_array,
            tip_heights=tip_heights,
            voltages=voltages,
            amplitudes=amplitudes,
            target_current=target_current,
            voltage_step=voltage_step,
            target_amplitude=target_amplitude
        )

        # Interpolate to create a consistent V grid
        if shared_voltages is None:
            shared_voltages = np.linspace(np.min(V_feedback), np.max(V_feedback), num=len(V_feedback))

        interp_dIdV = np.interp(shared_voltages, V_feedback, dI_dV_feedback)
        all_dIdV.append(interp_dIdV)

    # Convert results to a 2D grid
    dIdV_grid = np.array(all_dIdV)

    return shared_voltages, amplitudes, dIdV_grid

def plot_dIdV_colormap(V_grid, A_grid, dIdV_grid, current=None):
    """
    Plot dI/dV vs V vs A as a colormap.
    
    Parameters:
    - V_grid: 1D array of voltages.
    - A_grid: 1D array of amplitudes.
    - dIdV_grid: 2D array of dI/dV values.
    """
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=dIdV_grid,
        x=V_grid,
        y=A_grid,
        colorscale='Cividis',
        colorbar=dict(title='dI/dV (A/V)')
    ))

    # Update layout
    fig.update_layout(
        title=f'dI/dV vs V vs A (I = {current})',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='Amplitude (A)',
        template='plotly_white'
    )

    fig.show()

def plot_dIdV_colormap_with_slider(V_grid, A_grid, dIdV_grid, current=None):
    """
    Plot dI/dV vs V vs A as a colormap with a slider to tune the color scale.
    
    Parameters:
    - V_grid: 1D array of voltages.
    - A_grid: 1D array of amplitudes.
    - dIdV_grid: 2D array of dI/dV values.
    """
    # Initialize the heatmap
    initial_zmax = np.max(dIdV_grid)  # Set the initial max color scale value
    initial_zmin = 0  # Start the minimum color scale at 0

    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=dIdV_grid,
        x=V_grid,
        y=A_grid,
        colorscale='Cividis',
        zmin=initial_zmin,
        zmax=initial_zmax,
        colorbar=dict(title='dI/dV (A/V)')
    ))

    # Create steps for the slider
    zmax_values = np.linspace(0, initial_zmax, num=50)
    steps = []
    for zmax in zmax_values:
        step = dict(
            method="restyle",
            args=[{"zmax": [zmax]}],  # Update the color scale max value
            label=f"Max: {zmax:.2e}"
        )
        steps.append(step)

    # Add the slider
    sliders = [dict(
        active=len(zmax_values) - 1,
        currentvalue={"prefix": "Color Scale Max: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        title=f'dI/dV vs V vs A (I = {current}))',
        xaxis_title='Bias Voltage (V)',
        yaxis_title='Amplitude (A)',
        template='plotly_white',
        sliders=sliders,
        xaxis=dict(range=[V_start, V_stop]) # Set x-axis range
    )

    fig.show()






if __name__ == "__main__":
    current_array_path = "C:/Users/willh/valentines/current_array.npy"
    current_array = np.load(current_array_path)
    A_index = 0
    INTERPOLATE = True
    upsample_factor = 4
    if INTERPOLATE:
        D_values, V_values, current_array = interpolate_current_array(current_array, D_values, V_values, upsample_factor=upsample_factor)
    
    
    EXTRACT_USING_SKIMAGE = 0
    if EXTRACT_USING_SKIMAGE:
        custom_levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        # Step 0: Plot contours colored by current
        # plot_constant_current_scatter_plotly(current_array, D_values, V_values, A_values, A_target=0, current_levels=custom_levels)
        
        contours_with_dIdV = extract_constant_current_contours_with_dIdV(
            current_array=current_array,
            D_values=D_values,
            V_values=V_values,
            A_values=A_values,
            A_target=0,                # Focus on A=0
            current_levels=custom_levels,  # Provide your custom list of current levels
            delta_factor=0.0005          # 1% of V step size
        )
        
        scatter_fig = plot_contours_scatter_with_normalized_dIdV(
            contours_with_dIdV=contours_with_dIdV,
            cmap='Cividis'               # Diverging color scale suitable for dI/dV
        )
        
        # Step 3: Plot dI/dV vs V for each contour level
        dIdV_vs_V_fig = plot_dIdV_vs_V(
            contours_with_dIdV=contours_with_dIdV,
            cmap='Cividis'            # Sequential color scale for line differentiation
        )
           
    
    
    # Feedback loop simulation
    # Define a list of desired constant currents (ensure they are within the data range)
    I_targets = np.logspace(-13, -9, num=3)  # Example: 1 nA, 10 nA, 100 nA

    # Initialize lists to store all feedback paths
    V_end = V_values[-1]
    all_V_feedback = []
    all_D_feedback = []
    all_dI_dV_feedback = []
    all_I_targets = []

    plots_for_constant_A = 1
    if plots_for_constant_A:
        # Step 1: Simulate feedback loop for each target current
        for I_target in I_targets:
            # Ensure I_target is within the range of current_array
            I_min = np.nanmin(current_array[current_array > 0])
            I_max = np.nanmax(current_array)
            if not (I_min <= I_target <= I_max):
                print(f"I_target={I_target:.2e} A is outside the range [{I_min:.2e}, {I_max:.2e}] A. Skipping.")
                continue

            print(f"\nSimulating for I_target={I_target:.2e} A")
            V_fb, D_fb, dIdV_fb = simulate_feedback_loop(
                current_array=current_array,
                tip_heights=D_values,
                voltages=V_values,
                amplitudes=A_values,
                target_current=I_target,
                voltage_step=V_step,
                target_amplitude=1
            )
            if len(V_fb) > 0:
                all_V_feedback.append(V_fb)
                all_D_feedback.append(D_fb)
                all_dI_dV_feedback.append(dIdV_fb)
                all_I_targets.append(I_target)

    
        # Combine data for all current levels
        contours_with_dIdV = combine_contours_for_all_levels(all_V_feedback, all_D_feedback, all_dI_dV_feedback, all_I_targets)

        # Step 2: Plot D vs V scatter plot where color is normalized dI/dV
        plot_contours_scatter_with_normalized_dIdV(
            contours_with_dIdV=contours_with_dIdV,
            cmap='Cividis'  # Sequential color scale
        )

        # Step 3: Plot dI/dV vs V for all target currents
        plot_dIdV_vs_V(
            contours_with_dIdV=contours_with_dIdV,
            cmap='Cividis'  # Sequential color scale
        )

    shift_vs_A=1
    if shift_vs_A:
        for i,target_I in enumerate(I_targets):
            print(f'Simulating for I = {target_I} ({i}/{len(I_targets)})')
            V_grid, A_grid, dIdV_grid = simulate_feedback_for_all_A(
                current_array=current_array,
                tip_heights=D_values,
                voltages=V_values,
                amplitudes=A_values,
                target_current=target_I  # Example target current
            )

            plot_dIdV_colormap_with_slider(V_grid, A_grid, dIdV_grid, target_I)
    
    print('Done')