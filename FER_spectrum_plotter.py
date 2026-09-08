# Tailored loader + filter + plot for very large CSV(s) of STM spectra
# - Works if the given path is a single CSV file *or* a folder of CSV files.
# - Streams in chunks so you don't have to load everything into memory.
# - Robust to slight header name variations (e.g., "DI/dV" vs "dI/dV", "RF Freq" vs "RFFreq", "Power" vs "RF Power").
# - Filters for Setpoint=1000, RF Freq=1e8, RF Power=-30, then plots dZ/dV and RelativeZ vs V on dual y-axes.
#
# Adjust the filters near the top if needed. The output includes:
#   1) A CSV of the filtered rows
#   2) A PNG figure
#
# You can rerun safely; outputs get overwritten.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Configure matplotlib for publication-quality fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical expressions

# ---------------------------
# User inputs
# ---------------------------
INPUT_PATH = r"C:\Users\willh\Downloads\FullSpectraData_4Averaged.csv"  # file OR directory
DELIM = ","
TARGET_SETPOINT = 1000
TARGET_RF_FREQ = 2e8
TARGET_RF_POWER = -30

# Output files
out_csv = "filtered_setpoint1000_rf1e8_pwr-30.csv"
out_png = "dZdV_RelativeZ_vs_V.png"

# ---------------------------
# Helpers
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

# ---------------------------
# Streaming filter
# ---------------------------
selected_frames = []
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
        # Continue to next file if this one doesn't have the required columns
        # (You can print or log this to know which files were skipped)
        # print(f"Skipping {fpath}: missing columns {missing}")
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
        mask = (
            (chunk["Setpoint"] == TARGET_SETPOINT) &
            (np.isfinite(chunk["RFFreq"])) & (np.abs(chunk["RFFreq"] - TARGET_RF_FREQ) <= tol_freq) &
            (chunk["RFPower"] == TARGET_RF_POWER)
        )
        sel = chunk.loc[mask, ["V", "dZdV", "RelativeZ", *([ "LineGroup"] if "LineGroup" in chunk.columns else []), *([ "FileName"] if "FileName" in chunk.columns else [])]]
        if not sel.empty:
            selected_frames.append(sel)

# Concatenate all matching rows
if not selected_frames:
    raise RuntimeError("No rows matched the filter (Setpoint=1000, RF Freq=1e8, RF Power=-30). "
                       "Check column names & units or adjust tolerances.")

df = pd.concat(selected_frames, ignore_index=True)

# If multiple spectra are present, pick one deterministically (smallest LineGroup else first filename)
if "LineGroup" in df.columns and df["LineGroup"].notna().any():
    # Prefer the first complete spectrum by LineGroup (i.e., the one with the most rows)
    counts = df.groupby("LineGroup")["V"].count().sort_values(ascending=False)
    chosen_group = counts.index[0]
    df_plot = df[df["LineGroup"] == chosen_group].copy()
    chosen_desc = f"LineGroup={chosen_group}"
elif "FileName" in df.columns and df["FileName"].notna().any():
    counts = df.groupby("FileName")["V"].count().sort_values(ascending=False)
    chosen_file = counts.index[0]
    df_plot = df[df["FileName"] == chosen_file].copy()
    chosen_desc = f"FileName='{chosen_file}'"
else:
    df_plot = df.copy()
    chosen_desc = "first matching block (no LineGroup/FileName available)"
    
# Sort by V (increasing) for plotting
df_plot = df_plot.sort_values("V")

# Save filtered data (the chosen spectrum/group)
df_plot.to_csv(out_csv, index=False)

# ---------------------------
# Plot: V (x) vs dZ/dV (left) & RelativeZ (right)
# ---------------------------
fig, ax1 = plt.subplots(figsize=(7,4), dpi=150)

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
window_length = min(21, len(dZdV_interp) // 4)  # Conservative window length
if window_length % 2 == 0:  # Ensure odd window length
    window_length += 1
if window_length < 5:  # Minimum window length
    window_length = 5
    
dZdV_filtered = savgol_filter(dZdV_interp, window_length, polyorder=3)

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
print(f"V_numerical length: {len(V_numerical)}, dZdV_numerical_smooth length: {len(dZdV_numerical_smooth)}")
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

ax1.plot(V_interp, dZdV_scaled, linewidth=2.0, color='black')
ax1.set_xlabel("Sample Bias (V)")
ax1.set_ylabel("dZ/dV (nm/V)")

ax2 = ax1.twinx()
ax2.plot(V_volts, RelativeZ_nm, linewidth=2.0, linestyle="-", color='crimson')
ax2.set_ylabel("Tip Position (nm)", color='crimson', rotation=270, labelpad=20)
ax2.tick_params(axis='y', labelcolor='crimson')

# No title needed for publication

# Remove all ticks for clean publication style
ax1.tick_params(top=False, bottom=False, left=False, right=False)
ax2.tick_params(top=False, bottom=False, left=False, right=False)

# Apply Physical Review Letters styling
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Increase font sizes for publication quality
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax1.set_xlabel("Sample Bias (V)", fontsize=14)
ax1.set_ylabel("dZ/dV (nm/V)", fontsize=14)
ax2.set_ylabel("Tip Position (nm)", color='crimson', rotation=270, labelpad=20, fontsize=14)

# Line widths already set in plot commands above

fig.tight_layout()
fig.savefig(out_png, bbox_inches="tight")

# Provide a small textual summary as output
summary = {
    "rows_matched_total": int(len(df)),
    "rows_plotted": int(len(df_plot)),
    "group_choice": chosen_desc,
    "output_csv": out_csv,
    "output_png": out_png,
}

# Print the summary so user can see what happened
print("Processing complete!")
print(f"Total rows matched: {summary['rows_matched_total']}")
print(f"Rows plotted: {summary['rows_plotted']}")
print(f"Data source: {summary['group_choice']}")
print(f"Setpoint: {TARGET_SETPOINT}")
print(f"RF Frequency: {TARGET_RF_FREQ:g} Hz")
print(f"RF Power: {TARGET_RF_POWER} dBm")
print(f"Saved CSV: {summary['output_csv']}")
print(f"Saved PNG: {summary['output_png']}")

# Show the plot
plt.show()

print('Done!')