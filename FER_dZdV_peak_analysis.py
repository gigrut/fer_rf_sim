# dZ/dV Peak Analysis for FER Data
# Based on FER_peak_index_analysis.py but adapted for dZ/dV data from large CSV files
# Uses the same data filtering and processing as FER_spectrum_plotter.py

import os
import shutil
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, ifft, fftfreq
from collections import defaultdict
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib for publication-quality fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical expressions

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy", message="divide by zero encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy", message="invalid value encountered in multiply")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# ---------------------------
# Data Loading Configuration
# ---------------------------
INPUT_PATH = r"C:\Users\willh\Downloads\FullSpectraData_4Averaged.csv"  # file OR directory
DELIM = ","
TARGET_SETPOINT = 1000
TARGET_RF_FREQ = 1e8
TARGET_RF_POWER = -30
TARGET_RF_POWER_2 = 0  # Second power level to analyze

# ---------------------------
# Helper Functions (from both files)
# ---------------------------
def canonicalize(col: str) -> str:
    """Normalize a column name to a lowercase token w/o spaces/slashes for easier matching."""
    return col.strip().lower().replace(" ", "").replace("\\", "").replace("/", "")

def map_columns(cols):
    """
    Build a dict mapping canonical keys to actual column names found in the file.
    We try to be robust to name variants.
    """
    canon = {canonicalize(c): c for c in cols}
    
    # Canonical keys we want to find
    keymap = {}
    
    # V
    for k in ["v", "bias", "voltage"]:
        if k in canon:
            keymap["V"] = canon[k]; break
    
    # Current
    for k in ["current", "i"]:
        if k in canon:
            keymap["Current"] = canon[k]; break
    
    # dI/dV (we won't plot it now, but keep mapping)
    for k in ["di/dv", "didv", "dIdV".lower()]:
        if k in canon:
            keymap["dIdV"] = canon[k]; break
    
    # Z
    if "z" in canon:
        keymap["Z"] = canon["z"]
    
    # Setpoint
    for k in ["setpoint", "isetpoint", "currentsetpoint"]:
        if k in canon:
            keymap["Setpoint"] = canon[k]; break
    
    # RF Freq
    for k in ["rffreq", "rfreq", "rffrequency", "rf_freq"]:
        if k in canon:
            keymap["RFFreq"] = canon[k]; break
    
    # RF Power (try 'rfpower'; else a plain 'power' that is NOT 'vnapower')
    rf_power_col = None
    for k in ["rfpower", "rf_pow", "rfpwr"]:
        if k in canon:
            rf_power_col = canon[k]; break
    if rf_power_col is None:
        # try plain 'power' but avoid 'vnapower'
        if "power" in canon and "vnapower" not in canon:
            rf_power_col = canon["power"]
    if rf_power_col is not None:
        keymap["RFPower"] = rf_power_col
    
    # RF Voltage (optional)
    for k in ["rfvoltage", "rfv", "rf_v"]:
        if k in canon:
            keymap["RFVoltage"] = canon[k]; break
    
    # FileName
    for k in ["filename", "file", "fname"]:
        if k in canon:
            keymap["FileName"] = canon[k]; break
    
    # LineGroup
    for k in ["linegroup", "group", "spectrumindex", "spectrumid"]:
        if k in canon:
            keymap["LineGroup"] = canon[k]; break
    
    # dZ/dV
    for k in ["dz/dv", "dzdv"]:
        if k in canon:
            keymap["dZdV"] = canon[k]; break
    
    # RelativeZ
    for k in ["relativez", "zrelative", "z_rel"]:
        if k in canon:
            keymap["RelativeZ"] = canon[k]; break
    
    return keymap

def read_header(path) -> list:
    """Read only header from a CSV."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    # Use pandas for robust parsing (handles quoted headers with commas)
    try:
        df0 = pd.read_csv(path, nrows=0)
        return list(df0.columns)
    except Exception:
        # fallback naive split
        return [c.strip() for c in first_line.strip().split(DELIM)]

def files_to_process(input_path: str):
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        # All CSV-like files in the directory
        files = sorted([str(x) for x in p.glob("*.csv")])
        if not files:
            # If the folder/file happens to omit extension, try the exact path as a file
            maybe_file = str(p)
            if os.path.exists(maybe_file):
                return [maybe_file]
        return files
    else:
        return []

def try_to_numeric(series):
    """Convert series to numeric if possible; leaves as-is otherwise."""
    return pd.to_numeric(series, errors="coerce")

def dbm_to_mw(dBm):
    mW = 10**(dBm/10)
    return mW

def dbm_to_mv(dBm):
    # Convert dBm to zero-to-peak mV (50 ohm impedance)
    # P(mW) = 10^(dBm/10), V_rms = sqrt(P * R), V_zero_to_peak = V_rms * sqrt(2)
    mV = (2**0.5) * (50/1000)**(0.5) * (10**(dBm/20)) * 1000  # Convert V to mV
    return mV

def round_to_1sf(x):
    if x<=0:
        return x
    if x >= 1:
        return round(x, 1)
    else:
        power_of_10 = 10 ** math.floor(math.log10(x))
        return round(x, -int(math.log10(power_of_10)))

# ---------------------------
# RF Broadening Functions (adapted from Ag111 SS broadening.py)
# ---------------------------

def CGQ(func, V_DC, V_RF, n=100):
    """Chebyshev-Gauss Quadrature for RF broadening.
    Interpret func as dZ/dV(V), V_DC as the DC bias, V_RF as the RF amplitude, and n as the number of terms in the series.
    This can handles V_DC as a vector (pandas Series or NumPy array)"""
    i = np.array(range(1,n+1))
    x = np.cos(np.pi*(2*i-1)/(2*n))
    V_DC = np.asarray(V_DC)  # Convert V_DC to array if it's not already to ensure compatibility with broadcasting
    arg = V_DC[:, None] + V_RF * x  # Broadcasting happens here
    return np.sum(func(arg), axis=1) / n if V_DC.ndim > 0 else np.sum(func(arg)) / n

def broadenCGQ(spec, x_axis_name, y_axis_name, V_RF, n):
    """Apply RF broadening using Chebyshev-Gauss quadrature.
    params:
    spec: a pandas dataframe representing a point spectrum.  It can be any size.
    x_axis_name, y_axis_name: strings that correspond to the names of the two columns involved in the broadening
        the x-axis is something like 'V' and the y-axis is something like 'dZdV'
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
    """Returns the rms error between spec1 (assumed RF on/high power), and a broadened version of spec2 (assumed RF off/low power)."""
    if len(params) == 1:
        V_RF = params[0]
        y = 0  # Default y to 0 for single-variable case
    else:
        V_RF, y = params
    
    # Broaden spec2 (RF off/low power) and compare to spec1 (RF on/high power)
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
    """Extract RF amplitude by fitting broadened low power to high power data.
    Spec1 should be high power (+1dBm), spec2 should be low power (-30dBm).  
    Set variables=1 for single variable optimization (v_rf)
    Set variables=2 for double variable optimization (v_rf and y)"""
    
    # Better initial guess based on typical broadening range
    bias_range = spec2[x_axis_name].max() - spec2[x_axis_name].min()
    initial_v_rf = bias_range * 0.01  # Start with 1% of the bias range
    
    if variables == 1:
        initial_guess = [initial_v_rf]
        bounds = [(0, bias_range)]  # RF amplitude can be up to the full bias range
        result = minimize(rms_error, x0=initial_guess, args=(spec1, spec2, x_axis_name, y_axis_name), 
                          method=method, bounds=bounds, options={'maxiter': 1000})
        optimal_v_rf = result.x[0]
        optimal_y = 0
        print(f"RF extraction optimization: success={result.success}, message='{result.message}'")
        print(f"Final RMS error: {result.fun:.6f}")
        return optimal_v_rf, 0
    elif variables == 2:
        # Better initial guess for Y offset based on mean difference
        mean_diff = np.mean(spec1[y_axis_name]) - np.mean(spec2[y_axis_name])
        initial_y = mean_diff
        initial_guess = [initial_v_rf, initial_y]
        
        # More reasonable bounds for Y offset
        y_std = np.std(spec1[y_axis_name])
        y_range = y_std * 3  # Allow ±3 standard deviations
        bounds = [(0, bias_range), (-y_range, y_range)]
        
        # Try different optimization methods if the first fails
        methods_to_try = [method, 'L-BFGS-B', 'trust-constr']
        
        for opt_method in methods_to_try:
            try:
                result = minimize(rms_error, x0=initial_guess, args=(spec1, spec2, x_axis_name, y_axis_name), 
                                 method=opt_method, bounds=bounds, options={'maxiter': 2000})
                
                if result.success:
                    print(f"RF extraction successful with method: {opt_method}")
                    break
                else:
                    print(f"Method {opt_method} failed: {result.message}")
                    
            except Exception as e:
                print(f"Method {opt_method} crashed: {e}")
                continue
        
        optimal_v_rf = result.x[0]
        optimal_y = result.x[1]
        print(f"Final RF extraction result: success={result.success}, message='{result.message}'")
        print(f"Final RMS error: {result.fun:.6f}")
        return optimal_v_rf, optimal_y

def fit_quality(params, spectrum):
    z, phi = params
    for peak in spectrum:
        peak['field'] = peak['bias'] / (peak['z-topo'] + z)
        peak['fn'] = peak['field'] * peak['number']
    peak_fn = np.array([peak['fn'] for peak in spectrum])
    peak_biases = np.array([peak['bias'] for peak in spectrum])
    peak_biases_transformed = (peak_biases - phi) ** (3/2)  
    coeffs = np.polyfit(peak_fn[w:], peak_biases_transformed[w:], 1)
    fit_biases_transformed = np.polyval(coeffs, peak_fn)
    errors = peak_biases_transformed - fit_biases_transformed
    return np.sum(errors**2)

# ---------------------------
# Analysis Parameters
# ---------------------------
# w = number of points to ignore in linear fit (relaxed for smaller datasets)
w = 2
# Parameters for the Savitzky-Golay filter for peak detection
window_length = 51  # Must be a positive odd integer
polyorder = 4       # The order of the polynomial used to fit the samples
# Peak detection parameters (adjusted for dZ/dV data which may have different scale)
prominence = None  # Let's start with no prominence requirement and adjust
distance = 10

# Color and shape mappings (adapted for single spectrum analysis)
current_to_shape = {
    0.03 : 'circle',  
    0.1 : 'square',  
    0.3 : 'diamond',  
    1 : 'cross',  
    3 : 'x'  
}

RF_to_color = {
    1.8 : 'rgb(31, 119, 180)',  # Blue
    1.3 : 'rgb(255, 127, 14)',  # Orange
    1 : 'rgb(44, 160, 44)',   # Green
    0.6 : 'rgb(214, 39, 40)',   # Red
    0 : 'rgb(148, 103, 189)'  # Purple
}

# ---------------------------
# Data Loading and Processing
# ---------------------------
print("Loading and filtering data...")

selected_frames_power1 = []
selected_frames_power2 = []
files = files_to_process(INPUT_PATH)

if not files:
    raise FileNotFoundError(f"No CSV files found at '{INPUT_PATH}'. If it's a single CSV, ensure the extension or correct path.")

for fpath in files:
    # Determine column mapping from header
    header_cols = read_header(fpath)
    cmap = map_columns(header_cols)
    
    # We need at least these to filter and plot
    required = ["V", "Setpoint", "RFFreq", "RFPower", "dZdV", "RelativeZ"]
    missing = [r for r in required if r not in cmap]
    if missing:
        print(f"Skipping {fpath}: missing columns {missing}")
        continue
    
    # Read only the columns we need to filter/plot (+ optional grouping columns)
    usecols = list({cmap[k] for k in required} | set([cmap.get("LineGroup", ""), cmap.get("FileName", "")]) - {""})
    
    for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=200_000):
        # Canonical column names
        chunk = chunk.rename(columns={v: k for k, v in cmap.items() if v in usecols})
        
        # Ensure numeric for filters/axes
        for col in ["V", "Setpoint", "RFFreq", "RFPower", "dZdV", "RelativeZ"]:
            chunk[col] = try_to_numeric(chunk[col])
        
        # Apply filters (with small tolerances for float repr)
        tol_freq = 1.0  # Hz tolerance
        tol_power = 0.1  # dBm tolerance for power (handles floating point errors near 0)
        
        # Filter for first power level (-30dBm)
        mask1 = (
            (chunk["Setpoint"] == TARGET_SETPOINT) &
            (np.isfinite(chunk["RFFreq"])) & (np.abs(chunk["RFFreq"] - TARGET_RF_FREQ) <= tol_freq) &
            (np.abs(chunk["RFPower"] - TARGET_RF_POWER) <= tol_power)
        )
        sel1 = chunk.loc[mask1, ["V", "dZdV", "RelativeZ", *([ "LineGroup"] if "LineGroup" in chunk.columns else []), *([ "FileName"] if "FileName" in chunk.columns else [])]]
        if not sel1.empty:
            selected_frames_power1.append(sel1)
        
        # Filter for second power level (1dBm)
        mask2 = (
            (chunk["Setpoint"] == TARGET_SETPOINT) &
            (np.isfinite(chunk["RFFreq"])) & (np.abs(chunk["RFFreq"] - TARGET_RF_FREQ) <= tol_freq) &
            (np.abs(chunk["RFPower"] - TARGET_RF_POWER_2) <= tol_power)
        )
        sel2 = chunk.loc[mask2, ["V", "dZdV", "RelativeZ", *([ "LineGroup"] if "LineGroup" in chunk.columns else []), *([ "FileName"] if "FileName" in chunk.columns else [])]]
        if not sel2.empty:
            selected_frames_power2.append(sel2)

# Process first power level (-30dBm)
if not selected_frames_power1:
    print(f"No rows matched the filter for RF Power={TARGET_RF_POWER}dBm")
    df_power1 = None
else:
    df_power1 = pd.concat(selected_frames_power1, ignore_index=True)

# Process second power level (-15dBm)  
if not selected_frames_power2:
    print(f"No rows matched the filter for RF Power={TARGET_RF_POWER_2}dBm")
    df_power2 = None
else:
    df_power2 = pd.concat(selected_frames_power2, ignore_index=True)

def select_best_spectrum(df, power_label):
    """Select the best spectrum from a dataframe with multiple spectra."""
    if df is None or df.empty:
        return None, None
    
    # If multiple spectra are present, pick one deterministically (smallest LineGroup else first filename)
    if "LineGroup" in df.columns and df["LineGroup"].notna().any():
        # Prefer the first complete spectrum by LineGroup (i.e., the one with the most rows)
        counts = df.groupby("LineGroup")["V"].count().sort_values(ascending=False)
        chosen_group = counts.index[0]
        df_selected = df[df["LineGroup"] == chosen_group].copy()
        chosen_desc = f"LineGroup={chosen_group}, {power_label}"
    elif "FileName" in df.columns and df["FileName"].notna().any():
        counts = df.groupby("FileName")["V"].count().sort_values(ascending=False)
        chosen_file = counts.index[0]
        df_selected = df[df["FileName"] == chosen_file].copy()
        chosen_desc = f"FileName='{chosen_file}', {power_label}"
    else:
        df_selected = df.copy()
        chosen_desc = f"first matching block, {power_label}"
        
    # Sort by V (increasing) for plotting
    df_selected = df_selected.sort_values("V")
    
    return df_selected, chosen_desc

# Select best spectra for both power levels
df_plot_power1, desc1 = select_best_spectrum(df_power1, f"RF Power={TARGET_RF_POWER}dBm")
df_plot_power2, desc2 = select_best_spectrum(df_power2, f"RF Power={TARGET_RF_POWER_2}dBm")

if df_plot_power1 is not None:
    print(f"Data loaded for -30dBm: {len(df_plot_power1)} points from {desc1}")
if df_plot_power2 is not None:
    print(f"Data loaded for -15dBm: {len(df_plot_power2)} points from {desc2}")

# Use the first power level (-30dBm) as the primary dataset for the original analysis
if df_plot_power1 is not None:
    df_plot = df_plot_power1
    chosen_desc = desc1
else:
    raise RuntimeError(f"No data found for primary power level {TARGET_RF_POWER}dBm")

# ---------------------------
# Signal Processing (same as FER_spectrum_plotter.py)
# ---------------------------
print("Processing dZ/dV signal...")

# Convert V from mV to V by dividing by 1000
V_volts = df_plot["V"] / 1000
# Convert RelativeZ from DAC units to nm by multiplying by 1.721e-4
RelativeZ_nm = df_plot["RelativeZ"] * 1.721e-4

# Interpolate dZ/dV data by a factor of 4
V_min, V_max = V_volts.min(), V_volts.max()
V_interp = np.linspace(V_min, V_max, len(V_volts) * 4)
interp_func = interp1d(V_volts, df_plot["dZdV"], kind='cubic', bounds_error=False, fill_value='extrapolate')
dZdV_interp = interp_func(V_interp)

# Apply conservative Savitzky-Golay filter (window length should be odd and >= 5)
window_length_interp = min(21, len(dZdV_interp) // 4)  # Conservative window length
if window_length_interp % 2 == 0:  # Ensure odd window length
    window_length_interp += 1
if window_length_interp < 5:  # Minimum window length
    window_length_interp = 5
    
dZdV_filtered = savgol_filter(dZdV_interp, window_length_interp, polyorder=3)

# Regauge the dZ/dV data by comparing to numerical derivative of RelativeZ
# Compute numerical derivative of RelativeZ vs V
dV = np.diff(V_volts)
dZ = np.diff(RelativeZ_nm)
dZdV_numerical = dZ / dV
V_numerical = (V_volts[:-1] + V_volts[1:]) / 2  # Midpoint voltages

# Apply some smoothing to the numerical derivative if needed
if len(dZdV_numerical) > 10:
    smooth_window = min(11, len(dZdV_numerical) // 3)
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window >= 5:
        dZdV_numerical_smooth = savgol_filter(dZdV_numerical, smooth_window, polyorder=2)
    else:
        dZdV_numerical_smooth = dZdV_numerical
else:
    dZdV_numerical_smooth = dZdV_numerical

# Find overlapping voltage range for comparison
V_min_common = max(V_interp.min(), V_numerical.min())
V_max_common = min(V_interp.max(), V_numerical.max())

# Interpolate both datasets to common voltage grid for comparison
V_common = np.linspace(V_min_common, V_max_common, 100)
dZdV_filtered_interp = interp1d(V_interp, dZdV_filtered, bounds_error=False, fill_value='extrapolate')(V_common)

# Check array lengths before interpolation
if len(V_numerical) != len(dZdV_numerical_smooth):
    # Ensure arrays have same length
    min_len = min(len(V_numerical), len(dZdV_numerical_smooth))
    V_numerical = V_numerical[:min_len]
    dZdV_numerical_smooth = dZdV_numerical_smooth[:min_len]

dZdV_numerical_interp = interp1d(V_numerical, dZdV_numerical_smooth, bounds_error=False, fill_value='extrapolate')(V_common)

# Calculate scaling factor (RMS ratio to avoid sign issues)
valid_mask = np.isfinite(dZdV_filtered_interp) & np.isfinite(dZdV_numerical_interp)
if np.sum(valid_mask) > 10:
    rms_lockin = np.sqrt(np.mean(dZdV_filtered_interp[valid_mask]**2))
    rms_numerical = np.sqrt(np.mean(dZdV_numerical_interp[valid_mask]**2))
    if rms_lockin > 0:
        scale_factor = rms_numerical / rms_lockin
        dZdV_scaled = dZdV_filtered * scale_factor
        print(f"Scaling factor applied to dZ/dV: {scale_factor:.6f}")
        print(f"dZ/dV units after scaling: nm/V")
    else:
        dZdV_scaled = dZdV_filtered
        print("Warning: Could not determine scaling factor, using original dZ/dV data")
else:
    dZdV_scaled = dZdV_filtered
    print("Warning: Insufficient overlap for scaling, using original dZ/dV data")

# ---------------------------
# Peak Detection
# ---------------------------
print("Detecting peaks in dZ/dV data...")

# Convert back to mV for consistency with original analysis
V_interp_mV = V_interp * 1000

# Apply additional smoothing for peak detection (similar to original file)
if len(dZdV_scaled) > window_length:
    dZdV_smoothed_for_peaks = savgol_filter(dZdV_scaled, window_length, polyorder)
else:
    # Use smaller window if data is too short
    peak_window = min(window_length, len(dZdV_scaled))
    if peak_window % 2 == 0:
        peak_window -= 1
    if peak_window >= 5:
        dZdV_smoothed_for_peaks = savgol_filter(dZdV_scaled, peak_window, min(polyorder, peak_window-1))
    else:
        dZdV_smoothed_for_peaks = dZdV_scaled

# Check the full voltage range first
print(f"Full voltage range: {V_interp_mV.min():.1f} to {V_interp_mV.max():.1f} mV")

# Filter voltage range (similar to original analysis)
voltage_mask = (V_interp_mV > 3000) & (V_interp_mV < 9700)
V_filtered = V_interp_mV[voltage_mask]
dZdV_filtered_for_peaks = dZdV_smoothed_for_peaks[voltage_mask]

print(f"After voltage filtering: {V_filtered.min():.1f} to {V_filtered.max():.1f} mV")
print(f"Voltage filtering removed {len(V_interp_mV) - len(V_filtered)} points")

# Check if the last peak might be outside our voltage range
if V_interp_mV.max() > 9700:
    print(f"WARNING: Data extends beyond 9700 mV - last peak might be filtered out!")
    print("Trying extended voltage range...")
    # Try with extended range
    extended_mask = (V_interp_mV > 3000) & (V_interp_mV < V_interp_mV.max())
    V_extended = V_interp_mV[extended_mask]
    dZdV_extended = dZdV_smoothed_for_peaks[extended_mask]
    print(f"Extended range: {V_extended.min():.1f} to {V_extended.max():.1f} mV ({len(V_extended)} points)")
    
    # Use extended range for peak detection
    V_filtered = V_extended
    dZdV_filtered_for_peaks = dZdV_extended

# Debug: Print data statistics for peak detection
print(f"Voltage range for peak detection: {V_filtered.min():.1f} to {V_filtered.max():.1f} mV")
print(f"dZ/dV data range: {dZdV_filtered_for_peaks.min():.3e} to {dZdV_filtered_for_peaks.max():.3e} nm/V")
print(f"dZ/dV data std: {dZdV_filtered_for_peaks.std():.3e} nm/V")
print(f"Number of points for peak detection: {len(dZdV_filtered_for_peaks)}")

# Calculate voltage spacing and appropriate distance parameter
voltage_range_V = (V_filtered.max() - V_filtered.min()) / 1000  # Convert to volts
voltage_spacing_V = voltage_range_V / len(dZdV_filtered_for_peaks)
print(f"Voltage spacing per point: {voltage_spacing_V*1000:.3f} mV")
print(f"Points per 0.5V: {0.5 / voltage_spacing_V:.1f}")

# Adjust distance based on expected peak separation
min_peak_separation_V = 0.3  # Minimum expected separation in volts
adaptive_distance = max(3, int(min_peak_separation_V / voltage_spacing_V))
print(f"Adaptive distance parameter: {adaptive_distance} (for {min_peak_separation_V}V separation)")

# Simple, permissive peak detection - let scipy find all the peaks
print("Using simple peak detection with minimal restrictions...")

# Very basic peak detection - just find local maxima with minimal constraints
peaks, peak_properties = find_peaks(dZdV_filtered_for_peaks, 
                                   height=None,      # No height requirement
                                   distance=3,       # Very small distance (allow close peaks)
                                   prominence=None,  # No prominence requirement
                                   width=None)       # No width requirement

print(f"Basic peak detection found {len(peaks)} peaks")

# If we get too many peaks, add minimal prominence filter
if len(peaks) > 10:
    print("Too many peaks found, adding minimal prominence filter...")
    # Use a very small prominence - just to filter out noise
    min_prominence = dZdV_filtered_for_peaks.std() * 0.01  # Very permissive
    peaks, _ = find_peaks(dZdV_filtered_for_peaks, 
                         distance=3, 
                         prominence=min_prominence)
    print(f"With minimal prominence filter: {len(peaks)} peaks")

print(f"Final peak detection: {len(peaks)} peaks found")

print(f"Raw peak detection found {len(peaks)} peaks")
if len(peaks) > 0:
    print(f"Peak positions (indices): {peaks}")
    print(f"Peak voltages (mV): {V_filtered[peaks]}")
    print(f"Peak values (nm/V): {dZdV_filtered_for_peaks[peaks]}")

# Extract peak information
peak_info = []
for k, peak_idx in enumerate(peaks):
    # Use the filtered arrays directly since we have the peak index in the filtered data
    peak_dict = {
        'number': k + 1,
        'index': peak_idx,  # Index in the filtered arrays
        'bias': V_filtered[peak_idx],  # in mV
        'dZdV': dZdV_filtered_for_peaks[peak_idx],  # scaled dZ/dV value
        'voltage_V': V_filtered[peak_idx] / 1000,  # in V
        # Add experimental parameters from the data
        'setpoint': TARGET_SETPOINT,
        'rf_freq': TARGET_RF_FREQ,
        'rf_power': TARGET_RF_POWER,
        # For compatibility with original analysis, we'll use placeholder values
        'z-topo': 0.0,  # Not available in this dataset
        'tip-sample-distance': 5.0  # Placeholder value
    }
    peak_info.append(peak_dict)

print(f"Found {len(peak_info)} peaks in dZ/dV data")

# Print peak information
for peak in peak_info:
    print(f"Peak {peak['number']}: {peak['bias']:.1f} mV, dZ/dV = {peak['dZdV']:.3e} nm/V")

# Optional diagnostic plot (set to False to disable)
show_diagnostic = False
if show_diagnostic:
    fig_debug, ax_debug = plt.subplots(figsize=(10, 6), dpi=150)
    # Convert to volts for display
    V_filtered_V = V_filtered / 1000
    ax_debug.plot(V_filtered_V, dZdV_filtered_for_peaks, linewidth=1.5, color='blue', label='Smoothed dZ/dV (for peak detection)')
    if len(peaks) > 0:
        ax_debug.scatter(V_filtered_V[peaks], dZdV_filtered_for_peaks[peaks], 
                        marker='*', s=150, color='red', edgecolors='black', 
                        linewidth=2, label=f'Detected Peaks (n={len(peaks)})', zorder=5)
    ax_debug.set_xlabel("Sample Bias (V)", fontsize=12)
    ax_debug.set_ylabel("dZ/dV (nm/V)", fontsize=12)
    ax_debug.set_title("Peak Detection Diagnostic", fontsize=14)
    ax_debug.legend()
    ax_debug.grid(True, alpha=1.0)
    plt.tight_layout()
    plt.show()
    print("Diagnostic plot shown")

# ---------------------------
# Plotting (using matplotlib with PRL styling)
# ---------------------------
print("Creating plots...")

# Plot 1: dZ/dV spectrum with detected peaks and Z on secondary axis
fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=150)

# Plot the processed dZ/dV data (convert to volts)
ax1.plot(V_interp, dZdV_scaled, linewidth=2.0, color='black', label='dZ/dV')

# Create secondary axis for Z data
ax2 = ax1.twinx()
ax2.plot(V_volts, RelativeZ_nm, linewidth=2.0, color='crimson', alpha=1.0, label='Z position')

# Apply PRL styling
ax1.set_xlabel("Sample Bias (V)", fontsize=14)
ax1.set_ylabel("dZ/dV (nm/V)", fontsize=14, color='black')
ax2.set_ylabel("Z Position (nm)", fontsize=14, color='crimson', rotation=270, labelpad=20)

# Color the y-axis labels to match the data
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax2.tick_params(axis='y', labelcolor='crimson', labelsize=12)

# Remove all x-axis ticks for clean publication style
ax1.tick_params(top=False, bottom=False, left=False, right=False, labelsize=12)
ax2.tick_params(top=False, bottom=False, left=False, right=False)

# Apply Physical Review Letters styling with full frame
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)

# No legend needed

fig1.tight_layout()
fig1.savefig("dZdV_spectrum_with_peaks.png", bbox_inches="tight", dpi=300)
plt.show()

# Process second power data for zoom plot (if available)
dZdV_scaled_p2 = None
V_interp_p2 = None

if df_plot_power2 is not None:
    print("Processing second power level data for zoom plot...")
    
    # Process the second power level data with same scaling as first power
    V_volts_p2 = df_plot_power2["V"] / 1000
    RelativeZ_nm_p2 = df_plot_power2["RelativeZ"] * 1.721e-4
    
    # Interpolate dZ/dV data by a factor of 4
    V_min_p2, V_max_p2 = V_volts_p2.min(), V_volts_p2.max()
    V_interp_p2 = np.linspace(V_min_p2, V_max_p2, len(V_volts_p2) * 4)
    interp_func_p2 = interp1d(V_volts_p2, df_plot_power2["dZdV"], kind='cubic', bounds_error=False, fill_value='extrapolate')
    dZdV_interp_p2 = interp_func_p2(V_interp_p2)
    
    # Apply same filtering to second power data
    window_length_interp_p2 = min(21, len(dZdV_interp_p2) // 4)
    if window_length_interp_p2 % 2 == 0:
        window_length_interp_p2 += 1
    if window_length_interp_p2 < 5:
        window_length_interp_p2 = 5
    dZdV_filtered_p2 = savgol_filter(dZdV_interp_p2, window_length_interp_p2, polyorder=3)
    
    # Regauge the second power dZ/dV data by comparing to numerical derivative of RelativeZ
    # Compute numerical derivative of RelativeZ vs V
    dV_p2 = np.diff(V_volts_p2)
    dZ_p2 = np.diff(RelativeZ_nm_p2)
    dZdV_numerical_p2 = dZ_p2 / dV_p2
    V_numerical_p2 = (V_volts_p2[:-1] + V_volts_p2[1:]) / 2  # Midpoint voltages
    
    # Apply some smoothing to the numerical derivative if needed
    if len(dZdV_numerical_p2) > 10:
        smooth_window_p2 = min(11, len(dZdV_numerical_p2) // 3)
        if smooth_window_p2 % 2 == 0:
            smooth_window_p2 += 1
        if smooth_window_p2 >= 5:
            dZdV_numerical_smooth_p2 = savgol_filter(dZdV_numerical_p2, smooth_window_p2, polyorder=2)
        else:
            dZdV_numerical_smooth_p2 = dZdV_numerical_p2
    else:
        dZdV_numerical_smooth_p2 = dZdV_numerical_p2
    
    # Find overlapping voltage range for comparison
    V_min_common_p2 = max(V_interp_p2.min(), V_numerical_p2.min())
    V_max_common_p2 = min(V_interp_p2.max(), V_numerical_p2.max())
    
    # Interpolate both datasets to common voltage grid for comparison
    V_common_p2 = np.linspace(V_min_common_p2, V_max_common_p2, 100)
    dZdV_filtered_interp_p2 = interp1d(V_interp_p2, dZdV_filtered_p2, bounds_error=False, fill_value='extrapolate')(V_common_p2)
    
    # Check array lengths before interpolation
    if len(V_numerical_p2) != len(dZdV_numerical_smooth_p2):
        # Ensure arrays have same length
        min_len_p2 = min(len(V_numerical_p2), len(dZdV_numerical_smooth_p2))
        V_numerical_p2 = V_numerical_p2[:min_len_p2]
        dZdV_numerical_smooth_p2 = dZdV_numerical_smooth_p2[:min_len_p2]
    
    dZdV_numerical_interp_p2 = interp1d(V_numerical_p2, dZdV_numerical_smooth_p2, bounds_error=False, fill_value='extrapolate')(V_common_p2)
    
    # Calculate scaling factor (RMS ratio to avoid sign issues)
    valid_mask_p2 = np.isfinite(dZdV_filtered_interp_p2) & np.isfinite(dZdV_numerical_interp_p2)
    if np.sum(valid_mask_p2) > 10:
        rms_lockin_p2 = np.sqrt(np.mean(dZdV_filtered_interp_p2[valid_mask_p2]**2))
        rms_numerical_p2 = np.sqrt(np.mean(dZdV_numerical_interp_p2[valid_mask_p2]**2))
        if rms_lockin_p2 > 0:
            scale_factor_p2 = rms_numerical_p2 / rms_lockin_p2
            dZdV_scaled_p2 = dZdV_filtered_p2 * scale_factor_p2
            print(f"Scaling factor applied to -15dBm dZ/dV: {scale_factor_p2:.6f}")
            print(f"dZ/dV units after scaling: nm/V")
        else:
            dZdV_scaled_p2 = dZdV_filtered_p2
            print("Warning: Could not determine scaling factor for -15dBm, using original dZ/dV data")
    else:
        dZdV_scaled_p2 = dZdV_filtered_p2
        print("Warning: Insufficient overlap for scaling -15dBm data, using original dZ/dV data")
else:
    print("No -15dBm data found")


lower_bound = 5.5
upper_bound = 7.0
# Plot 2: RF Broadening Analysis (if both power levels available)
if dZdV_scaled_p2 is not None and V_interp_p2 is not None:
    print("\n" + "="*60)
    print("PERFORMING RF BROADENING ANALYSIS")
    print("="*60)
    
    # Create DataFrames for RF broadening analysis
    # Low power (-30dBm) = "RF OFF" equivalent
    df_low_power = pd.DataFrame({
        'V': V_interp,  # Keep in volts for FER analysis
        'dZdV': dZdV_scaled
    })
    
    # High power (0dBm) = "RF ON" equivalent  
    df_high_power = pd.DataFrame({
        'V': V_interp_p2,  # Keep in volts for FER analysis
        'dZdV': dZdV_scaled_p2
    })
    
    # Apply smoothing to high power data for fitting (like in Ag111 file)
    window_length = min(21, len(df_high_power) // 4)
    if window_length % 2 == 0:
        window_length += 1
    if window_length < 5:
        window_length = 5
    
    df_high_power_smoothed = df_high_power.copy()
    df_high_power_smoothed['dZdV'] = savgol_filter(df_high_power['dZdV'], window_length, polyorder=3)
    
    # Extract RF amplitude using broadening analysis
    print("Extracting RF amplitude from dZ/dV broadening...")
    print(f"Low power level: {TARGET_RF_POWER} dBm")
    print(f"High power level: {TARGET_RF_POWER_2} dBm")
    
    # Use 2-variable optimization (RF amplitude + vertical offset)
    try:
        VRF_V, Y_offset = extract_rf(df_high_power_smoothed, df_low_power, 'V', 'dZdV', method='Nelder-Mead', variables=2)
        
        # Convert to mV for comparison with source
        VRF_mV = VRF_V * 1000
        
        # Calculate broadened spectrum for plotting
        broadened_spectrum = broadenCGQ(df_low_power, 'V', 'dZdV', VRF_V, 100) + Y_offset
        
        # Calculate source voltages and transmission
        source_voltage_low_mV = dbm_to_mv(TARGET_RF_POWER)
        source_voltage_high_mV = dbm_to_mv(TARGET_RF_POWER_2)
        
        # Calculate transmission coefficient
        transmission_dB = 20 * np.log10(VRF_mV / source_voltage_high_mV)
        amplitude_ratio = VRF_mV / source_voltage_high_mV
        power_ratio = amplitude_ratio ** 2
        
        print(f"\nRF BROADENING ANALYSIS RESULTS:")
        print(f"RF amplitude at junction: {VRF_mV:.1f} mV ({VRF_V:.4f} V)")
        print(f"Source voltage ({TARGET_RF_POWER_2} dBm): {source_voltage_high_mV:.1f} mV")
        print(f"Transmission coefficient: {transmission_dB:.2f} dB")
        print(f"Amplitude ratio: {amplitude_ratio:.3f}")
        print(f"Power ratio: {power_ratio:.3f}")
        print(f"Vertical offset: {Y_offset:.3e} nm/V")
        
        # Create RF broadening plot
        fig2, ax_zoom = plt.subplots(figsize=(7, 4), dpi=150)

        
        # Filter data for voltage range for plotting
        V_mask_low = (df_low_power['V'] >= lower_bound) & (df_low_power['V'] <= upper_bound)  # V
        V_mask_high = (df_high_power['V'] >= lower_bound) & (df_high_power['V'] <= upper_bound)  # V
        V_mask_broad = (df_low_power['V'] >= lower_bound) & (df_low_power['V'] <= upper_bound)  # V
        
        # Plot the data (already in volts)
        ax_zoom.plot(df_low_power['V'][V_mask_low], df_low_power['dZdV'][V_mask_low], 
                    marker='o', linestyle='', markersize=4, color='black', label='RF off')
        ax_zoom.plot(df_high_power['V'][V_mask_high], df_high_power['dZdV'][V_mask_high], 
                    marker='o', linestyle='', markersize=4, color='blue', 
                    label=f'RF on \n(Source voltage: {source_voltage_high_mV:.1f} mV)')
        ax_zoom.plot(df_low_power['V'][V_mask_broad], broadened_spectrum[V_mask_broad], 
                    marker='o', linestyle='', markersize=4, color='crimson',
                    label=f'Simulated broadening\n(RF amplitude: {VRF_mV:.1f} mV)')
        
        # Apply PRL styling
        ax_zoom.set_xlabel("Sample Bias (V)", fontsize=14)
        ax_zoom.set_ylabel("dZ/dV (nm/V)", fontsize=14)
        
        # Remove all x-axis ticks for clean publication style
        ax_zoom.tick_params(top=False, bottom=False, left=False, right=False, labelsize=12)
        
        # Apply Physical Review Letters styling with full frame
        ax_zoom.spines['top'].set_visible(True)
        ax_zoom.spines['right'].set_visible(True)
        ax_zoom.spines['bottom'].set_visible(True)
        ax_zoom.spines['left'].set_visible(True)
        
        # Set x-axis limits to voltage range
        ax_zoom.set_xlim(lower_bound, upper_bound)
        
        # Add legend
        ax_zoom.legend(fontsize=10, loc='upper left')
        
        fig2.tight_layout()
        fig2.savefig("dZdV_RF_broadening_analysis.png", bbox_inches="tight", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"RF broadening analysis failed: {e}")
        print("Falling back to simple power comparison plot...")
        
        # Fallback to simple comparison plot
        fig2, ax_zoom = plt.subplots(figsize=(7, 4), dpi=150)
        
        # Filter data for voltage range (first power level)
        V_mask = (V_interp >= lower_bound) & (V_interp <= upper_bound)
        V_zoom = V_interp[V_mask]
        dZdV_zoom = dZdV_scaled[V_mask]
        
        # Plot both power levels
        ax_zoom.plot(V_zoom, dZdV_zoom, marker='o', linestyle='', markersize=4, color='black', label='RF off')
        
        V_mask_p2 = (V_interp_p2 >= lower_bound) & (V_interp_p2 <= upper_bound)
        V_zoom_p2 = V_interp_p2[V_mask_p2]
        dZdV_zoom_p2 = dZdV_scaled_p2[V_mask_p2]
        ax_zoom.plot(V_zoom_p2, dZdV_zoom_p2, marker='o', linestyle='', markersize=4, color='red', label=f'{TARGET_RF_POWER_2} dBm')
        
        # Apply styling
        ax_zoom.set_xlabel("Sample Bias (V)", fontsize=14)
        ax_zoom.set_ylabel("dZ/dV (nm/V)", fontsize=14)
        ax_zoom.tick_params(top=False, bottom=False, left=False, right=False, labelsize=12)
        ax_zoom.spines['top'].set_visible(True)
        ax_zoom.spines['right'].set_visible(True)
        ax_zoom.spines['bottom'].set_visible(True)
        ax_zoom.spines['left'].set_visible(True)
        ax_zoom.set_xlim(lower_bound, upper_bound)
        ax_zoom.legend(fontsize=12, loc='upper left')
        
        fig2.tight_layout()
        fig2.savefig(f"dZdV_spectrum_{lower_bound}-{upper_bound}V_zoom.png", bbox_inches="tight", dpi=300)
        plt.show()

else:
    print(f"No {TARGET_RF_POWER_2}dBm data found, skipping RF broadening analysis")
    
    # Plot just the low power data
    fig2, ax_zoom = plt.subplots(figsize=(7, 4), dpi=150)
    
    # Filter data for voltage range
    V_mask = (V_interp >= lower_bound) & (V_interp <= upper_bound)
    V_zoom = V_interp[V_mask]
    dZdV_zoom = dZdV_scaled[V_mask]
    
    # Plot the zoomed dZ/dV data
    ax_zoom.plot(V_zoom, dZdV_zoom, marker='o', linestyle='', markersize=4, color='black', label='RF off')
    
    # Apply styling
    ax_zoom.set_xlabel("Sample Bias (V)", fontsize=14)
    ax_zoom.set_ylabel("dZ/dV (nm/V)", fontsize=14)
    ax_zoom.tick_params(top=False, bottom=False, left=False, right=False, labelsize=12)
    ax_zoom.spines['top'].set_visible(True)
    ax_zoom.spines['right'].set_visible(True)
    ax_zoom.spines['bottom'].set_visible(True)
    ax_zoom.spines['left'].set_visible(True)
    ax_zoom.set_xlim(lower_bound, upper_bound)
    
    fig2.tight_layout()
    fig2.savefig(f"dZdV_spectrum_{lower_bound}-{upper_bound}V_zoom.png", bbox_inches="tight", dpi=300)
    plt.show()

# Plot 3: FER analysis plot (adapted for single spectrum with relaxed requirements)
if len(peak_info) >= 3:  # Relaxed requirement - need at least 3 peaks
    fig3, ax3 = plt.subplots(figsize=(7, 4), dpi=150)
    
    # Extract data for FER analysis
    peak_numbers = [peak['number'] for peak in peak_info]
    peak_biases_V = [peak['voltage_V'] for peak in peak_info]
    
    # FER transformation: (n - 0.25)^(2/3) vs bias
    x = np.array([(n - 0.25)**(2/3) for n in peak_numbers])
    y = np.array(peak_biases_V)
    
    # Plot the data points
    ax3.scatter(x, y, s=100, color='black', marker='o', 
               edgecolors='black', linewidth=2)
    
    # Fit a line using only points 4-6 (indices 3-5), but extend line to left
    if len(x) >= 6:
        # Use points 4-6 for fitting
        fit_indices = [3, 4, 5]  # Points 4, 5, 6 (0-indexed)
        coeffs = np.polyfit(x[fit_indices], y[fit_indices], 1)
        fit_points_used = "4-6"
    elif len(x) >= 4:
        # Use last 3 points if we don't have 6
        fit_indices = [-3, -2, -1]
        coeffs = np.polyfit(x[fit_indices], y[fit_indices], 1)
        fit_points_used = f"{len(x)-2}-{len(x)}"
    else:
        # Fallback to all points
        coeffs = np.polyfit(x, y, 1)
        fit_points_used = "all"
        
    # Extend line all the way back to left (start from 0)
    x_fit = np.linspace(0, max(x) * 1.1, 100)
    y_fit = np.polyval(coeffs, x_fit)
    
    # Plot dashed line with no label
    ax3.plot(x_fit, y_fit, linewidth=2.0, color='black', linestyle='--')
    
    print(f"FER analysis: slope = {coeffs[0]:.6f}, intercept = {coeffs[1]:.6f}")
    print(f"Using points {fit_points_used} out of {len(x)} peaks for linear fit")
    
    # Calculate field F using the physically true equation: E_n = workfunction + α * F^(2/3) * (n-0.25)^(2/3)
    # where α = 0.9458 in the desired units
    # The slope = α * F^(2/3), therefore F = (slope/α)^(3/2)
    alpha = 0.9458
    field_F = (coeffs[0] / alpha)**(3/2)
    print(f"Field F = (slope / α)^(3/2) = ({coeffs[0]:.6f} / {alpha})^(3/2) = {field_F:.6f}")
    
    # Calculate work function from Y intercept
    work_function = coeffs[1]
    print(f"Work function = Y intercept = {work_function:.6f} V")
    
    # Apply PRL styling with proper mathematical typesetting
    ax3.set_xlabel(r"(peak number $- 1/4)^{2/3}$", fontsize=14)
    ax3.set_ylabel("Peak Bias (V)", fontsize=14)
    
    # Remove all ticks for clean publication style
    ax3.tick_params(top=False, bottom=False, left=False, right=False, labelsize=12)
    
    # Apply Physical Review Letters styling with full frame
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['left'].set_visible(True)
    
    # No legend needed
    # ax3.legend(fontsize=12)
    ax3.set_xlim(left=0)
    
    fig3.tight_layout()
    fig3.savefig("FER_analysis_dZdV.png", bbox_inches="tight", dpi=300)
    plt.show()
    
else:
    print(f"Not enough peaks ({len(peak_info)}) for FER analysis (need >= 3)")


print("Analysis complete!")
