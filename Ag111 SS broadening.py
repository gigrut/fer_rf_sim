import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


bias_col_name = 'Bias (mV)'
BIAS_UNITS = 'mV'
CALIBRATION_FREQUENCY = 10.0 #MHz
CALIBRATION_POWER = -38.1 #dBm

file_path = 'C:/Users/willh/OneDrive/Desktop/Ag111_SS_Broadening.csv'
dI_dV_col_name = 'dI/dV'  # Column name for dI/dV data

# Read the CSV file
df = pd.read_csv(file_path)


def dbm_to_mw(dBm):
        mW = 10**(dBm/10)
        return mW
def dbm_to_mv(dBm):
    """Convert dBm to mV peak amplitude assuming 50 ohm impedance
    P(mW) = 10^(dBm/10)
    V_rms = sqrt(P(mW) * R(ohm) * 1e-3) where R=50 ohm
    V_peak = V_rms * sqrt(2)
    """
    mW = dbm_to_mw(dBm)
    # V_rms = sqrt(P * R) where P is in Watts and R = 50 ohms
    V_rms_volts = (mW * 1e-3 * 50)**0.5  # RMS voltage in volts
    # Convert RMS to peak amplitude
    V_peak_volts = V_rms_volts * (2**0.5)  # Peak = RMS * sqrt(2)
    mV_peak = V_peak_volts * 1000  # Convert V to mV
    return mV_peak
def CGQ(func, V_DC, V_RF, n=100):
    """Interpret func as I(V), V_DC as the DC bias, V_RF as the RF amplitude, and n as the number of terms in the series.
    This can handles V_DC as a vector (pandas Series or NumPy array)"""
    i = np.array(range(1,n+1))
    x = np.cos(np.pi*(2*i-1)/(2*n))
    V_DC = np.asarray(V_DC)  # Convert V_DC to array if it's not already to ensure compatibility with broadcasting
    arg = V_DC[:, None] + V_RF * x  # Broadcasting happens here
    return np.sum(func(arg), axis=1) / n if V_DC.ndim > 0 else np.sum(func(arg)) / n
def broadenCGQ(spec, x_axis_name, y_axis_name, V_RF, n):
    """This is the function that does all the heavy lifting.
    It uses Chebeshev-Gauss quadrature (CGQ) to compute RF broadening
    params:
    spec: a pandas dataframe representing a point spectrum.  It can be any size.
    x_axis_name, y_axis_name: strings that correspond to the names of the two columns involved in the broadening
        the x-axis is something like 'bias' and the y-axis is something like 'di/dv'
    V_RF: a float that represents the amplitude of the broadening.  It can have any units, but should match the units of the x-axis
    n: an integer that tells the functions how many terms to use in the series approximation of the integral
        n=100 seems to be plenty accurate and fast enough.
    Returns a column corresponding to spec[y_axis_name] but broadened"""
    if type(V_RF) == np.ndarray:
        V_RF = V_RF.item()  
    
    # Remove duplicates and sort for interpolation
    df_sorted = spec.sort_values(x_axis_name).drop_duplicates(subset=[x_axis_name])
    
    # Use cubic interpolation if we have enough points, otherwise fall back to linear
    try:
        if len(df_sorted) >= 4:  # Need at least 4 points for cubic
            interp_func = interp1d(df_sorted[x_axis_name], df_sorted[y_axis_name], kind='cubic', 
                                  bounds_error=False, fill_value=0.0)
        else:
            interp_func = interp1d(df_sorted[x_axis_name], df_sorted[y_axis_name], kind='linear', 
                                  bounds_error=False, fill_value=0.0)
    except ValueError:
        # Fall back to linear if cubic fails
        interp_func = interp1d(df_sorted[x_axis_name], df_sorted[y_axis_name], kind='linear', 
                              bounds_error=False, fill_value=0.0)
    df = spec.copy()
    x_min = df[x_axis_name].min()
    x_max = df[x_axis_name].max()
    
    # More conservative masking - only mask points very close to the edges
    # This reduces NaN artifacts while still avoiding edge effects
    edge_buffer = V_RF * 1.5  # Only mask if within 1.5*V_RF of edges
    mask = (df[x_axis_name] - x_min < edge_buffer) | (x_max - df[x_axis_name] < edge_buffer)
    
    # Apply the CGQ function where the mask is False, and np.nan where True
    broadened_col = np.where(mask, np.nan, CGQ(interp_func, df[x_axis_name], V_RF, n))
    return broadened_col
def rms_error(params, spec1, spec2, x_axis_name, y_axis_name, n=100):
    """Returns the rms error between spec1 (assumed RF on), and a broadened version of spec2 (assumed RF off)."""
    if len(params) == 1:
        V_RF = params[0]
        y = 0  # Default y to 0 for single-variable case
    else:
        V_RF, y = params
    
    # Broaden spec2 (RF off) and compare to spec1 (RF on)
    broadened_col = broadenCGQ(spec2, x_axis_name, y_axis_name, V_RF, n) + y 
    
    # Ensure both DataFrames have the same x_axis and valid (non-nan) values for comparison
    valid_mask = ~np.isnan(broadened_col) & ~np.isnan(spec1[y_axis_name])
    
    # Check if we have enough valid points for meaningful comparison
    if np.sum(valid_mask) < 10:
        return 1e6  # Return large error if insufficient valid points
    
    # Calculate RMS error only on valid points
    error = np.sqrt(np.mean((broadened_col[valid_mask] - spec1[y_axis_name][valid_mask]) ** 2))
    
    # Add penalty for excessive NaN values to discourage over-broadening
    nan_penalty = (1 - np.sum(valid_mask) / len(broadened_col)) * 0.1 * error
    
    return error + nan_penalty
def extract_rf(spec1, spec2, x_axis_name, y_axis_name, method='Nelder-Mead', variables=1):
    """Spec1 should be RF on, spec2 should be RF off.  
    Set variables=1 for single variable optimization (v_rf)
    Set variables=2 for double variable optimization (v_rf and y)"""
    from scipy.optimize import Bounds
    
    # Better initial guess based on typical broadening range
    bias_range = spec2[x_axis_name].max() - spec2[x_axis_name].min()
    initial_v_rf = bias_range * 0.01  # Start with 1% of the bias range
    
    if variables == 1:
        initial_guess = [initial_v_rf]
        bounds = Bounds([0], [bias_range])  # RF amplitude can be up to the full bias range
        result = minimize(rms_error, x0=initial_guess, args=(spec1, spec2, x_axis_name, y_axis_name), 
                          method=method, bounds=bounds, options={'maxiter': 1000, 'disp': True})
        optimal_v_rf = result.x[0]
        optimal_y = 0
        print(f"Optimization result: success={result.success}, message='{result.message}'")
        print(f"Final error: {result.fun:.6f}")
        return optimal_v_rf, 0
    elif variables == 2:
        # Better initial guess for Y offset based on mean difference
        mean_diff = np.mean(spec1[y_axis_name]) - np.mean(spec2[y_axis_name])
        initial_y = mean_diff
        initial_guess = [initial_v_rf, initial_y]
        
        # More reasonable bounds for Y offset
        y_std = np.std(spec1[y_axis_name])
        y_range = y_std * 3  # Allow ±3 standard deviations
        bounds = Bounds([0, -y_range], [bias_range, y_range])
        
        # Try different optimization methods if Nelder-Mead fails
        methods_to_try = [method, 'L-BFGS-B', 'trust-constr']
        
        for opt_method in methods_to_try:
            try:
                if opt_method == 'Nelder-Mead':
                    # Nelder-Mead doesn't use bounds, so try without them first
                    result = minimize(rms_error, x0=initial_guess, args=(spec1, spec2, x_axis_name, y_axis_name), 
                                     method=opt_method, options={'maxiter': 2000, 'disp': True})
                else:
                    result = minimize(rms_error, x0=initial_guess, args=(spec1, spec2, x_axis_name, y_axis_name), 
                                     method=opt_method, bounds=bounds, options={'maxiter': 2000, 'disp': True})
                
                if result.success:
                    print(f"Optimization successful with method: {opt_method}")
                    break
                else:
                    print(f"Method {opt_method} failed: {result.message}")
                    
            except Exception as e:
                print(f"Method {opt_method} crashed: {e}")
                continue
        
        optimal_v_rf = result.x[0]
        optimal_y = result.x[1]
        print(f"Final optimization result: success={result.success}, message='{result.message}'")
        print(f"Final error: {result.fun:.6f}")
        return optimal_v_rf, optimal_y


# Debug: Print the actual column names
# print("Available columns in CSV file:")
# print(df.columns.tolist())
# print("\nFirst few rows:")
# print(df.head())

# Check if the expected bias column exists, if not use the first column
if bias_col_name not in df.columns:
    actual_bias_col = df.columns[0]
    print(f"\nWarning: '{bias_col_name}' not found. Using '{actual_bias_col}' as bias column.")
else:
    actual_bias_col = bias_col_name

# Extract available power values from column headers
power_columns = []
available_powers = []
for col in df.columns:
    if col.startswith('pow='):
        power_value = float(col.split('=')[1])
        power_columns.append(col)
        available_powers.append(power_value)

print(f"Available powers: {available_powers}")

# Find the closest available power to the calibration power
if available_powers:
    closest_power = min(available_powers, key=lambda x: abs(x - CALIBRATION_POWER))
    closest_power_col = f'pow={closest_power}'
    print(f"Requested calibration power: {CALIBRATION_POWER} dBm")
    print(f"Using closest available power: {closest_power} dBm (column: '{closest_power_col}')")
    print(f"Equivalent voltage at generator: {dbm_to_mv(closest_power)} mV")
else:
    print("Warning: No power columns found in the data!")
    closest_power_col = 'pow=-30'  # fallback

# Create dIdV_off_df from the 'Off' column
dIdV_off_df = pd.DataFrame({
    bias_col_name: df[actual_bias_col],
    dI_dV_col_name: df['Off'] / 500  # Scale down by factor of 500 to get nA/V units
})

# Create dIdV_on_df from the closest power column  
dIdV_on_df = pd.DataFrame({
    bias_col_name: df[actual_bias_col],
    dI_dV_col_name: df[closest_power_col] / 500  # Scale down by factor of 500 to get nA/V units
})

# Create a smoothed version of RF ON data for fitting purposes only
# Apply conservative Savitzky-Golay filter to reduce noise
window_length = min(21, len(dIdV_on_df) // 4)  # Conservative window length
if window_length % 2 == 0:  # Ensure odd window length
    window_length += 1
if window_length < 5:  # Minimum window length
    window_length = 5

dIdV_on_df_smoothed = dIdV_on_df.copy()
dIdV_on_df_smoothed[dI_dV_col_name] = savgol_filter(dIdV_on_df[dI_dV_col_name], window_length, polyorder=3)


# ********** COMPUTING CALIBRATION FACTOR VIA RF BROADENING **********
# Set variables=1 for single variable optimization (RF amplitude only)
# Set variables=2 for double variable optimization (RF amplitude and a vertical offset between RF off and RF on spectra)
num_of_variables = 2
CALIBRATION_VOLTAGE = dbm_to_mv(closest_power)  # Use the actual power that was selected
print('Extracting RF amplitude')
VRF, Y = extract_rf(dIdV_on_df_smoothed, dIdV_off_df, bias_col_name, dI_dV_col_name, method='Nelder-Mead', variables=num_of_variables) # Use smoothed RF ON data for fitting
print('RF amplitude extracted')
print(f'VRF = {VRF} {BIAS_UNITS}, Y = {Y} {BIAS_UNITS}')
print(f'Source voltage was {CALIBRATION_VOLTAGE} {BIAS_UNITS}')
print(f'Transmission at {CALIBRATION_FREQUENCY} MHz is {round(20*np.log10(VRF/CALIBRATION_VOLTAGE),2)} dB')

# Calculate and print amplitude and power ratios
amplitude_ratio = VRF / CALIBRATION_VOLTAGE
power_ratio = (VRF / CALIBRATION_VOLTAGE) ** 2
print(f'Amplitude ratio (VRF/V_source): {amplitude_ratio:.3f}')
print(f'Power ratio (VRF/V_source)²: {power_ratio:.3f}')
if BIAS_UNITS=='mV':
    actual_voltage_at_junction_in_dBmV = 20*np.log10(VRF)
elif BIAS_UNITS=='V':
    actual_voltage_at_junction_in_dBmV = 20*np.log10(VRF*1000)
# Giving dIdV_off_df an extra column for the numerically broadened dIdV
dIdV_off_df[f'{dI_dV_col_name} numerically broadened'] = broadenCGQ(dIdV_off_df, bias_col_name, dI_dV_col_name, VRF, 100)+Y




# Configure matplotlib for publication-quality fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical expressions

# Calculate the broadened spectrum for plotting
broadened_spectrum = broadenCGQ(dIdV_off_df, bias_col_name, dI_dV_col_name, VRF, 100) + Y

# Create the plot with consistent styling
fig, ax = plt.subplots(figsize=(7,4), dpi=150)

# Keep bias in mV for this plot
bias_mV_off = dIdV_off_df[bias_col_name]
bias_mV_on = dIdV_on_df[bias_col_name]

# Plot the data with lines only (no markers) - classic color scheme
ax.plot(bias_mV_off, dIdV_off_df[dI_dV_col_name], linewidth=2.0, color='black', label='RF off')
ax.plot(bias_mV_on, dIdV_on_df[dI_dV_col_name], linewidth=2.0, color='blue', 
        label=f'RF on \n(Source voltage: {CALIBRATION_VOLTAGE:.1f} mV)',)
ax.plot(bias_mV_off, broadened_spectrum, linewidth=2.0, color='crimson', linestyle='-', 
        label=f'Simulated broadening \n(RF amplitude: {VRF:.1f} mV)')

# Apply consistent styling with much larger fonts
ax.set_xlabel("Sample Bias (mV)", fontsize=20)
ax.set_ylabel("dI/dV (nA/mV)", fontsize=20)

# Add legend in bottom right corner
ax.legend(loc='lower right', fontsize=14)

# Remove ticks for clean publication style
ax.tick_params(top=False, bottom=False, left=False, right=False)

# Keep frame all the way around (don't remove any spines)
# All spines remain visible for complete frame

# Much larger font sizes for tick labels
ax.tick_params(labelsize=16)

fig.tight_layout()
plt.show()

# Print some diagnostic information
print(f"\nDiagnostic Information:")
print(f"Data range: {dIdV_off_df[bias_col_name].min():.1f} to {dIdV_off_df[bias_col_name].max():.1f} mV")
print(f"RF amplitude: {VRF:.3f} mV ({VRF/abs(dIdV_off_df[bias_col_name].max() - dIdV_off_df[bias_col_name].min())*100:.1f}% of data range)")
print(f"Y offset: {Y:.3f} mV")
valid_points = ~np.isnan(broadened_spectrum)
print(f"Valid broadened points: {np.sum(valid_points)}/{len(broadened_spectrum)} ({np.sum(valid_points)/len(broadened_spectrum)*100:.1f}%)")