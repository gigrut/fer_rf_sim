import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as sp
from FER_sim_parallel_processing_V2 import (
    # physical constants
    E_F, phi_t, phi_s,
    FIELD_ENHANCEMENT_FACTOR, Lateral_Confinement_Potential,
    # arrays
    D_values, V_values, A_values, x_values,
)
V_values += Lateral_Confinement_Potential

# -------------------------------------------------------------------
# Function to plot slices of T_array
# -------------------------------------------------------------------
def plot_T_slice_const_D_A(T_array, D_values, V_values, A_values, x_values, D_value, A_value, component="magnitude"):
    """
    Plot a slice of T_array for a constant D value and A value.

    Parameters:
    - T_array: The array of complex T values, shape (len(D_values), len(V_values), len(A_values), len(x_values)).
    - D_values: Array of tip heights (D).
    - V_values: Array of bias voltages (V).
    - A_values: Array of amplitude values (A).
    - x_values: Array of x values (electron energy).
    - D_value: Constant D value to plot.
    - A_value: Constant A value to plot.
    - component: The component of T to plot ("real", "imaginary", "magnitude", or "phase").
    """
    if component not in ["real", "imaginary", "magnitude", "phase"]:
        raise ValueError("Invalid component. Choose from 'real', 'imaginary', 'magnitude', 'phase'.")

        # Find the index of the closest D_value in D_values
    D_diff = np.abs(D_values - D_value)
    D_index = D_diff.argmin()
    closest_D = D_values[D_index]
    if not np.isclose(closest_D, D_value, atol=1e-8):
        print(f"Warning: D_value={D_value} nm not found. Using closest value D={closest_D} nm.")

    # Find the index of the closest V_value in V_values
    A_diff = np.abs(A_values - A_value)
    A_index = A_diff.argmin()
    closest_A = A_values[A_index]
    if not np.isclose(closest_A, A_value, atol=1e-8):
        print(f"Warning: A_value={A_value} V not found. Using closest value A={closest_A} V.")

    # Extract the slice for the given D and A indices
    T_slice = T_array[D_index, :, A_index, :]  # Shape: (len(V_values), len(x_values))

    # Compute the requested component
    if component == "real":
        T_plot = T_slice.real
    elif component == "imaginary":
        T_plot = T_slice.imag
    elif component == "magnitude":
        T_plot = np.abs(T_slice)
    elif component == "phase":
        T_plot = np.angle(T_slice)

    # Create a meshgrid for x (electron energy) and V (bias voltage) values
    X_mesh, V_mesh = np.meshgrid(x_values, V_values, indexing="ij")

    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, V_mesh, T_plot.T, cmap="cividis", edgecolor='k')

    # Add labels and title
    ax.set_xlabel("Electron Energy (x)")
    ax.set_ylabel("Bias Voltage (V)")
    ax.set_zlabel(f"T ({component.capitalize()})")
    ax.set_title(f"3D Plot of T ({component.capitalize()}), D = {D_values[D_index]}, A = {A_values[A_index]}")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=f"T ({component.capitalize()})")

    plt.show()

def plot_T_slice_const_D_V(T_array, D_values, V_values, A_values, x_values, D_value, V_value, component="magnitude", A_start=0):
    """
    Plot a slice of T_array for a constant D value and V value, with A on the y-axis and x on the x-axis.

    Parameters:
    - T_array: The array of complex T values, shape (len(D_values), len(V_values), len(A_values), len(x_values)).
    - D_values: Array of tip heights (D).
    - V_values: Array of bias voltages (V).
    - A_values: Array of amplitude values (A).
    - x_values: Array of x values (electron energy).
    - D_value: The specific D value to plot. If not found exactly, the closest value is used.
    - V_value: The specific V value to plot. If not found exactly, the closest value is used.
    - component: The component of T to plot ("real", "imaginary", "magnitude", or "phase").
    """
    if component not in ["real", "imaginary", "magnitude", "phase"]:
        raise ValueError("Invalid component. Choose from 'real', 'imaginary', 'magnitude', 'phase'.")

    # Find the index of the closest D_value in D_values
    D_diff = np.abs(D_values - D_value)
    D_index = D_diff.argmin()
    closest_D = D_values[D_index]
    if not np.isclose(closest_D, D_value, atol=1e-8):
        print(f"Warning: D_value={D_value} nm not found. Using closest value D={closest_D} nm.")

    # Find the index of the closest V_value in V_values
    V_diff = np.abs(V_values - V_value)
    V_index = V_diff.argmin()
    closest_V = V_values[V_index]
    if not np.isclose(closest_V, V_value, atol=1e-8):
        print(f"Warning: V_value={V_value} V not found. Using closest value V={closest_V} V.")

    # Find the index of the closest A_value >= 0.1
    A_start_index = np.searchsorted(A_values, A_start, side='left')
    if A_start_index >= len(A_values):
        raise ValueError("A = 0.1 is outside the range of provided A_values.")

    # Extract the slice for the given D, V, and starting at A >= 0.1
    T_slice = T_array[D_index, V_index, A_start_index:, :]  # Shape: (len(A_values[A_start_index:]), len(x_values))
    A_values = A_values[A_start_index:]

    # Compute the requested component
    if component == "real":
        T_plot = T_slice.real
    elif component == "imaginary":
        T_plot = T_slice.imag
    elif component == "magnitude":
        T_plot = np.abs(T_slice)
    elif component == "phase":
        T_plot = np.angle(T_slice)

    # Create a meshgrid for x (electron energy) and A (amplitude) values
    X_mesh, A_mesh = np.meshgrid(x_values, A_values, indexing="ij")

    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, A_mesh, T_plot.T, cmap="cividis", edgecolor='k')

    # Add labels and title
    ax.set_xlabel("Electron Energy (x)")
    ax.set_ylabel("Amplitude (A)")
    ax.set_zlabel(f"T ({component.capitalize()})")
    ax.set_title(f"3D Plot of T ({component.capitalize()}), D = {D_values[D_index]:.2g}, V = {(V_values[V_index]):.2g}")

    plt.show()

def plot_T_slice_const_V_A(T_array, D_values, V_values, A_values, x_values, V_value, A_value, component="magnitude"):
    """
    Plot a slice of T_array for a constant V value and A value, with D on the y-axis and x on the x-axis.

    Parameters:
    - T_array: The array of complex T values, shape (len(D_values), len(V_values), len(A_values), len(x_values)).
    - D_values: Array of tip heights (D).
    - V_values: Array of bias voltages (V).
    - A_values: Array of amplitude values (A).
    - x_values: Array of x values (electron energy).
    - V_value: The specific V value to plot. If not found exactly, the closest value is used.
    - A_value: The specific A value to plot. If not found exactly, the closest value is used.
    - component: The component of T to plot ("real", "imaginary", "magnitude", or "phase").
    
    Raises:
    - ValueError: If the provided component is invalid.
    """
    if component not in ["real", "imaginary", "magnitude", "phase"]:
        raise ValueError("Invalid component. Choose from 'real', 'imaginary', 'magnitude', 'phase'.")

    # Find the index of the closest A_value in A_values
    A_diff = np.abs(A_values - A_value)
    A_index = A_diff.argmin()
    closest_A = A_values[A_index]
    if not np.isclose(closest_A, A_value, atol=1e-8):
        print(f"Warning: A_value={A_value} V not found. Using closest value A={closest_A} V.")

    # Find the index of the closest V_value in V_values
    V_diff = np.abs(V_values - V_value)
    V_index = V_diff.argmin()
    closest_V = V_values[V_index]
    if not np.isclose(closest_V, V_value, atol=1e-8):
        print(f"Warning: V_value={V_value} V not found. Using closest value V={closest_V} V.")
    
    # **Corrected Indexing: Fixing V and A, varying D and x**
    # Extract the slice for the given V and A indices
    # Shape of T_slice: (len(D_values), len(x_values))
    T_slice = T_array[:, V_index, A_index, :]  # Corrected from T_array[A_index, V_index, :, :]
    
    # Compute the requested component
    if component == "real":
        T_plot = T_slice.real
    elif component == "imaginary":
        T_plot = T_slice.imag
    elif component == "magnitude":
        T_plot = np.abs(T_slice)
    elif component == "phase":
        T_plot = np.angle(T_slice)

    # **Corrected meshgrid: D on y-axis, x on x-axis**
    # To plot D vs x, create meshgrid with D on the y-axis and x on the x-axis
    X_mesh, D_mesh = np.meshgrid(x_values, D_values, indexing="xy")  # Corrected from indexing="ij"

    # Plot the 3D surface
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, D_mesh, T_plot, cmap="cividis", edgecolor='k', linewidth=0.5, antialiased=True)

    # Add labels and title
    ax.set_xlabel("Electron Energy (x)")
    ax.set_ylabel("Tip height (D)")
    ax.set_zlabel(f"T ({component.capitalize()})")
    ax.set_title(f"3D Plot of T ({component.capitalize()}), V = {closest_V} V, A = {closest_A} V")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=f"T ({component.capitalize()})")

    plt.show()

def plot_T_slice_const_D_V_heatmap(T_array, D_values, V_values, A_values, x_values, D_value, V_value, component="magnitude", A_start=0):
    """
    Plot a heatmap of T_array for a constant D value and V value, with A on the y-axis and x on the x-axis.

    Parameters:
    - T_array: The array of complex T values, shape (len(D_values), len(V_values), len(A_values), len(x_values)).
    - D_values: Array of tip heights (D).
    - V_values: Array of bias voltages (V).
    - A_values: Array of amplitude values (A).
    - x_values: Array of x values (electron energy).
    - D_value: The specific D value to plot. If not found exactly, the closest value is used.
    - V_value: The specific V value to plot. If not found exactly, the closest value is used.
    - component: The component of T to plot ("real", "imaginary", "magnitude", or "phase").
    """
    if component not in ["real", "imaginary", "magnitude", "phase"]:
        raise ValueError("Invalid component. Choose from 'real', 'imaginary', 'magnitude', 'phase'.")

    # Find the index of the closest D_value in D_values
    D_diff = np.abs(D_values - D_value)
    D_index = D_diff.argmin()
    closest_D = D_values[D_index]
    if not np.isclose(closest_D, D_value, atol=1e-8):
        print(f"Warning: D_value={D_value} nm not found. Using closest value D={closest_D} nm.")

    # Find the index of the closest V_value in V_values
    V_diff = np.abs(V_values - V_value)
    V_index = V_diff.argmin()
    closest_V = V_values[V_index]
    if not np.isclose(closest_V, V_value, atol=1e-8):
        print(f"Warning: V_value={V_value} V not found. Using closest value V={closest_V} V.")

    # Find the index of the closest A_value >= 0.1
    A_start_index = np.searchsorted(A_values, A_start, side='left')
    if A_start_index >= len(A_values):
        raise ValueError("A = 0.1 is outside the range of provided A_values.")

    # Extract the slice for the given D, V, and starting at A >= 0.1
    T_slice = np.log(T_array[D_index, V_index, A_start_index:, :])  # Shape: (len(A_values[A_start_index:]), len(x_values))
    A_values = A_values[A_start_index:]

    # Compute the requested component
    if component == "real":
        T_plot = T_slice.real
    elif component == "imaginary":
        T_plot = T_slice.imag
    elif component == "magnitude":
        T_plot = np.abs(T_slice)
    elif component == "phase":
        T_plot = np.angle(T_slice)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(
        T_plot, 
        aspect='auto', 
        origin='lower', 
        extent=[x_values[0], x_values[-1], A_values[0], A_values[-1]],
        cmap="cividis"
    )
    plt.colorbar(label=f"T ({component.capitalize()})")
    plt.xlabel("Electron Energy (x)")
    plt.ylabel("Amplitude (A)")
    plt.title(f"Heatmap of T ({component.capitalize()}), D = {D_values[D_index]:.2g}nm, V = {V_values[V_index]:.2g}V")
    plt.show()

def plot_current_array_scatter(current_array, D_values=None, V_values=None, A_values=None, fixed_A_index=0, 
                               derivative=False, log_output=False):
    """
    Plot the current array as a 3D scatter plot with V on the x-axis, D on the y-axis, 
    and either current (I) or its derivative (dI/dV) on the z-axis for a specific A index.
    Optionally, plot the logarithm of the output to handle a wide range of values.
    """
    # Validate input dimensions
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")

    # Infer D_values, V_values, and A_values if not provided
    D_values_in = D_values if D_values is not None else np.arange(current_array.shape[0])
    V_values_in = V_values if V_values is not None else np.arange(current_array.shape[1])
    A_values_in = A_values if A_values is not None else np.arange(current_array.shape[2])

    if not (0 <= fixed_A_index < len(A_values_in)):
        raise ValueError(f"fixed_A_index must be between 0 and {len(A_values_in) - 1}.")

    # Extract the slice of the current array for the given A_index
    current_slice = current_array[:, :, fixed_A_index]  # Expected shape: (nD, nV)

    # Check if the provided D and V arrays match the dimensions of current_slice.
    nD, nV = current_slice.shape
    if len(D_values_in) != nD:
        print("Warning: Provided D_values length doesn't match current_array's first dimension. Using indices for D.")
        D_plot = np.arange(nD)
    else:
        D_plot = D_values_in

    if len(V_values_in) != nV:
        print("Warning: Provided V_values length doesn't match current_array's second dimension. Using indices for V.")
        V_plot = np.arange(nV)
    else:
        V_plot = V_values_in

    # Compute the desired z-values (current or its derivative) from current_slice.
    if derivative:
        # Ensure V values are in ascending order (and sort if necessary)
        if not np.all(np.diff(V_plot) > 0):
            sorted_indices = np.argsort(V_plot)
            V_sorted = np.array(V_plot)[sorted_indices]
            current_slice_sorted = current_slice[:, sorted_indices]
            print("Info: V_values were not sorted. Sorting V_values and current_slice accordingly.")
        else:
            V_sorted = np.array(V_plot)
            current_slice_sorted = current_slice

        # Compute the derivative dI/dV using numpy's gradient
        dI_dV = np.gradient(current_slice_sorted, V_sorted, axis=1)

        if log_output:
            # For d(log(I))/dV, mask non-positive current values
            positive_mask = current_slice_sorted > 0
            if not np.all(positive_mask):
                print("Warning: Some current values are non-positive. They will be masked in the derivative computation.")
            with np.errstate(divide='ignore', invalid='ignore'):
                log_I = np.log(current_slice_sorted)
                dlogI_dV = np.gradient(log_I, V_sorted, axis=1)
                dlogI_dV[~positive_mask] = np.nan
            z_values = dlogI_dV
            z_label = "d(log(I))/dV (1/V)"
            color_scale = "RdBu"
            z_title = "d(log(I))/dV (1/V)"
        else:
            z_values = dI_dV
            z_label = "dI/dV (A/V)"
            color_scale = "RdBu"
            z_title = "dI/dV (A/V)"
    else:
        if log_output:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_I = np.log10(current_slice)
                log_I[~np.isfinite(log_I)] = np.nan
            z_values = log_I
            z_label = "Log(Current) (log A)"
            color_scale = "Cividis"
            z_title = "Log(Current) (log A)"
        else:
            z_values = current_slice
            z_label = "Current (A)"
            color_scale = "Cividis"
            z_title = "Current (A)"

    # Create a meshgrid for D and V using the (possibly corrected) coordinate arrays.
    D_grid, V_grid = np.meshgrid(D_plot, V_plot, indexing="ij")
    # Flatten arrays for the scatter plot.
    D_flat = D_grid.flatten()
    V_flat = V_grid.flatten()
    z_flat = z_values.flatten()

    # Filter out non-finite values.
    valid_mask = np.isfinite(z_flat)
    D_flat = D_flat[valid_mask]
    V_flat = V_flat[valid_mask]
    z_flat = z_flat[valid_mask]

    # For derivative plots, clip extreme values if needed.
    if derivative:
        z_flat = np.clip(z_flat, -1e3, 1e3)

    # Create the 3D scatter plot using Plotly.
    import plotly.graph_objects as go
    fig = go.Figure()

    scatter = go.Scatter3d(
        x=V_flat,
        y=D_flat,
        z=z_flat,
        mode='markers',
        marker=dict(
            size=3,
            color=z_flat,
            colorscale=color_scale,
            colorbar=dict(title=z_label),
            opacity=0.8
        )
    )

    fig.add_trace(scatter)

    # Set the title based on what is being plotted.
    if derivative and log_output:
        title_component = "d(log(I))/dV"
    elif derivative and not log_output:
        title_component = "dI/dV"
    elif not derivative and log_output:
        title_component = "Log(Current)"
    else:
        title_component = "Current"

    fig.update_layout(
        title=f"{title_component} Scatter Plot at A = {A_values_in[fixed_A_index]:.3f} V",
        scene=dict(
            xaxis_title="Bias Voltage (V)",
            yaxis_title="Tip Height (D)",
            zaxis_title=z_title
        ),
        xaxis=dict(range=[0,10]),
        template="plotly_white",
        width=900,
        height=700
    )

    fig.show()

def plot_current_array_scatter_multiple_A(current_array, D_values=None, V_values=None, A_values=None):
    """
    Plot the current array as a 3D scatter plot with V on the x-axis, D on the y-axis, 
    and log(Current) on the z-axis. Represent multiple A_values by color.

    Parameters:
    - current_array: 3D numpy array of current values with dimensions 
                     (len(D_values), len(V_values), len(A_values)).
    - D_values: Optional. Array of tip heights (D). If None, inferred as range(len(current_array.shape[0])).
    - V_values: Optional. Array of bias voltages (V). If None, inferred as range(len(current_array.shape[1])).
    - A_values: Optional. Array of amplitudes (A). If None, inferred as range(len(current_array.shape[2])).
    """
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")

    # Infer D_values, V_values, and A_values if not provided
    D_values = D_values if D_values is not None else np.arange(current_array.shape[0])
    V_values = V_values if V_values is not None else np.arange(current_array.shape[1])
    A_values = A_values if A_values is not None else np.arange(current_array.shape[2])

    # Create a meshgrid for D, V, A
    D_grid, V_grid, A_grid = np.meshgrid(D_values, V_values, A_values, indexing='ij')

    # Flatten arrays for scatter plot
    D_flat = D_grid.flatten()
    V_flat = V_grid.flatten()
    A_flat = A_grid.flatten()
    current_flat = current_array.flatten()

    # Optional: Filter out non-positive current values to avoid log issues
    valid_mask = current_flat > 0
    D_flat = D_flat[valid_mask]
    V_flat = V_flat[valid_mask]
    A_flat = A_flat[valid_mask]
    current_flat = current_flat[valid_mask]

    # Create the figure
    fig = go.Figure()

    # Add Scatter3d trace with color representing A_values
    fig.add_trace(go.Scatter3d(
        x=V_flat,
        y=D_flat,
        z=current_flat,
        mode='markers',
        marker=dict(
            size=5,
            color=A_flat,  # Color by A values
            colorscale="Cividis",
            colorbar=dict(title="A Value"),
            opacity=0.8
        ),
        name='Current Data Points'
    ))

    # Update layout
    fig.update_layout(
        title="Current Array Scatter Plot with Multiple A Values",
        scene=dict(
            xaxis_title="Bias Voltage (V)",
            yaxis_title="Tip Height (D)",
            zaxis_title="Current (A)"
        ),
        legend=dict(
            x=0.7,
            y=0.95
        ),
        template="plotly_white",
        width=800,
        height=600
    )

    # Show the plot
    fig.show()

def plot_current_array_scatter_faceted(current_array, D_values=None, V_values=None, A_values=None, num_facet_cols=2):
    """
    Plot the current array as multiple 3D scatter plots, each representing a subset of A_values.

    Parameters:
    - current_array: 3D numpy array of current values with dimensions 
                     (len(D_values), len(V_values), len(A_values)).
    - D_values: Optional. Array of tip heights (D). If None, inferred as range(len(current_array.shape[0])).
    - V_values: Optional. Array of bias voltages (V). If None, inferred as range(len(current_array.shape[1])).
    - A_values: Optional. Array of amplitudes (A). If None, inferred as range(len(current_array.shape[2])).
    - num_facet_cols: Number of columns in the facet grid.
    """
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")

    # Infer D_values, V_values, and A_values if not provided
    D_values = D_values if D_values is not None else np.arange(current_array.shape[0])
    V_values = V_values if V_values is not None else np.arange(current_array.shape[1])
    A_values = A_values if A_values is not None else np.arange(current_array.shape[2])

    num_A = len(A_values)
    num_facet_rows = int(np.ceil(num_A / num_facet_cols))

    # Create subplots
    fig = sp.make_subplots(rows=num_facet_rows, cols=num_facet_cols,
                           subplot_titles=[f"A = {A_values[i]:.3f}" for i in range(num_A)],
                           specs=[[{'type': 'scatter3d'} for _ in range(num_facet_cols)] for _ in range(num_facet_rows)])

    # Add traces for each A_value
    for i, A in enumerate(A_values):
        row = i // num_facet_cols + 1
        col = i % num_facet_cols + 1

        current_slice = current_array[:, :, i]
        D_grid, V_grid = np.meshgrid(D_values, V_values, indexing="ij")
        D_flat = D_grid.flatten()
        V_flat = V_grid.flatten()
        current_flat = current_slice.flatten()

        # Optional: Filter out non-positive current values
        valid_mask = current_flat > 0
        D_flat = D_flat[valid_mask]
        V_flat = V_flat[valid_mask]
        current_flat = current_flat[valid_mask]

        fig.add_trace(go.Scatter3d(
            x=V_flat,
            y=D_flat,
            z=np.log(current_flat),
            mode='markers',
            marker=dict(
                size=3,
                color=current_flat,
                colorscale="Cividis",
                colorbar=dict(title="Current (A)", len=0.5, y=0.5, x=1.05),
                opacity=0.8
            ),
            name=f"A={A:.3f}"
        ), row=row, col=col)

    # Update layout
    fig.update_layout(
        height=600 * num_facet_rows,
        width=800 * num_facet_cols,
        title_text="Current Array Scatter Plots for Multiple A Values",
        template="plotly_white"
    )

    # Show the plot
    fig.show()

def plot_current_array_scatter_animation(current_array, D_values=None, V_values=None, A_values=None):
    """
    Create an animated 3D scatter plot where each frame represents a different A_value.

    Parameters:
    - current_array: 3D numpy array of current values with dimensions 
                     (len(D_values), len(V_values), len(A_values)).
    - D_values: Optional. Array of tip heights (D). If None, inferred as range(len(current_array.shape[0])).
    - V_values: Optional. Array of bias voltages (V). If None, inferred as range(len(current_array.shape[1])).
    - A_values: Optional. Array of amplitudes (A). If None, inferred as range(len(current_array.shape[2])).
    """
    if current_array.ndim != 3:
        raise ValueError("current_array must be a 3D array.")

    # Infer D_values, V_values, and A_values if not provided
    D_values = D_values if D_values is not None else np.arange(current_array.shape[0])
    V_values = V_values if V_values is not None else np.arange(current_array.shape[1])
    A_values = A_values if A_values is not None else np.arange(current_array.shape[2])

    frames = []
    for i, A in enumerate(A_values):
        current_slice = current_array[:, :, i]
        D_grid, V_grid = np.meshgrid(D_values, V_values, indexing="ij")
        D_flat = D_grid.flatten()
        V_flat = V_grid.flatten()
        current_flat = current_slice.flatten()

        # Optional: Filter out non-positive current values
        valid_mask = current_flat > 0
        D_flat = D_flat[valid_mask]
        V_flat = V_flat[valid_mask]
        current_flat = current_flat[valid_mask]

        frame = go.Frame(
            data=[go.Scatter3d(
                x=V_flat,
                y=D_flat,
                z=current_flat,
                mode='markers',
                marker=dict(
                    size=5,
                    color=current_flat,
                    colorscale="Cividis",
                    colorbar=dict(title="Current (A)", len=0.5, y=0.5, x=1.05),
                    opacity=0.8
                ),
                name=f"A={A:.3f}"
            )],
            name=f"Frame {i}"
        )
        frames.append(frame)

    # Initial data
    initial_A = A_values[0]
    current_slice = current_array[:, :, 0]
    D_grid, V_grid = np.meshgrid(D_values, V_values, indexing="ij")
    D_flat = D_grid.flatten()
    V_flat = V_grid.flatten()
    current_flat = current_slice.flatten()
    valid_mask = current_flat > 0
    D_flat = D_flat[valid_mask]
    V_flat = V_flat[valid_mask]
    current_flat = current_flat[valid_mask]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=V_flat,
            y=D_flat,
            z=current_flat,
            mode='markers',
            marker=dict(
                size=5,
                color=current_flat,
                colorscale="Cividis",
                colorbar=dict(title="Current (A)", len=0.5, y=0.5, x=1.05),
                opacity=0.8
            ),
            name=f"A={initial_A:.3f}"
        )],
        layout=go.Layout(
            title="Current Array Scatter Plot Animation Over A Values",
            scene=dict(
                xaxis_title="Bias Voltage (V)",
                yaxis_title="Tip Height (D)",
                zaxis_title="Current (A)"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 500, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 300}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top"
                )
            ]
        ),
        frames=frames
    )

    fig.show()


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load the saved T_array and associated values
    print('loading arrays')
    #T_array = np.load("countdown/T_array.npy")
    T_array_path = "C:/Users/willh/countdown/T_array.npy"
    current_array_path = "C:/Users/willh/countdown/current_array.npy"
    print('loaded arrays')

    # Plot the real part of T for D_index = 0 and A_index = 0
    D_value = 3  # Change to select a different D value
    V_value = 5
    A_value = 0  # Change to select a different A value
    
    
 

    
    
    # current_array_path = "C:/Users/willh/Desktop/Data/results_for_fer_Jan_8/current_array.npy"
    print('loading data')
    T_array = np.load(T_array_path)
    current_array = np.load(current_array_path)
    print('loaded data')






    plot_T_slice_const_D_V_heatmap(T_array, D_values, V_values, A_values, x_values, D_value, V_value, component="magnitude", A_start=0)
    # plot_T_slice_const_D_V(T_array, D_values, V_values, A_values, x_values, D_value, V_value, component="magnitude", A_start=0)
    #plot_T_slice_const_D_A(T_array, D_values, V_values, A_values, x_values, D_value, A_value, component="magnitude")
    # plot_T_slice_const_V_A(T_array, D_values, V_values, A_values, x_values, V_value, A_value, component="real")   
    # plot_T_slice_const_D_V(T_array, D_values, V_values, A_values, x_values, D_value, V_value, component="magnitude")
    # plot_T_slice_const_D_A(T_array, D_values, V_values, A_values, x_values, D_value, A_value, component="magnitude")
    #plot_T_slice_const_V_A(T_array, D_values, V_values, A_values, x_values, V_value, A_value, component="magnitude")













    # plot_current_array_scatter_multiple_A(current_array, D_values, V_values, A_values)
    # plot_current_array_scatter_faceted(current_array, D_values, V_values, A_values)
    # plot_current_array_scatter_animation(current_array, D_values, V_values, A_values)
    print('plotting')
    # plot_current_array_scatter(current_array, D_values, V_values, A_values, fixed_A_index=0)
    print('plotted')
    
    plot_current_array = 0
    if plot_current_array:

        A_index = 0
    
        plot_current_array_scatter(
            current_array=current_array,
            D_values=D_values,
            V_values=V_values,
            A_values=A_values,
            fixed_A_index=A_index,      # Assuming A=0 corresponds to index 0
            derivative=False,      # Plotting derivative
            log_output=False       # Logarithmic scaling
        )
        plot_current_array_scatter(
            current_array=abs(current_array),
            D_values=D_values,
            V_values=V_values,
            A_values=A_values,
            fixed_A_index=A_index,      # Assuming A=0 corresponds to index 0
            derivative=False,      # Plotting derivative
            log_output=True       # Logarithmic scaling
        )
        plot_current_array_scatter(
            current_array=abs(current_array),
            D_values=D_values,
            V_values=V_values,
            A_values=A_values,
            fixed_A_index=A_index,      # Assuming A=0 corresponds to index 0
            derivative=True,      # Plotting derivative
            log_output=True       # Logarithmic scaling
        )
        # plot_current_array_scatter(current_array, D_values, V_values, A_values, fixed_A_index=2)