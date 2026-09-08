# pip install pandas plotly
import scipy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.special import lambertw
from scipy.optimize import curve_fit, least_squares, brentq

# --- 1) Load only needed columns ---
from fer_sim_config import REAL_DATA_CSV_PATH, REAL_DATA_SETPOINT, REAL_DATA_PEAK_NUMBER
csv_path = Path(REAL_DATA_CSV_PATH)
setpoint = REAL_DATA_SETPOINT
peak_number = 1 # 1-indexed  

peak_number = peak_number - 1 # to be 0-indexed  

usecols = [
    "PeakNumber", "RF Power", "Setpoint", "LineGroup", "FileName",
    "FitPeakCenter", "FitPeakWidth", "FitPeakCenterUncertainty", "FitPeakHeight",
]

df = pd.read_csv(
    csv_path,
    usecols=usecols,
    index_col=None,       # set to 0 if the first CSV column is a saved index you want to drop
    low_memory=False,
)

# --- 2) Clean types ---
for c in usecols:
    if c in df.columns and c != "FileName":  # Don't convert FileName to numeric
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --- 3) Convert RF Power (dBm) -> amplitude (V) with R=50Ω ---
# P(W) = 10^((dBm - 30)/10)
P_W = 10.0 ** ((df["RF Power"] - 30.0) / 10.0)
df["RF Amplitude (V)"] = np.sqrt(2.0 * P_W * 50.0)

# --- 4) Filter: Setpoint==1000 and PeakNumber==0 ---
mask = (df["Setpoint"] == setpoint) & (df["PeakNumber"] == peak_number)
df0 = df.loc[mask].copy()

# --- 5) Compute baseline using first file (RF off) for each setpoint/RF Power combo ---
# For each unique combination of setpoint and RF Power, find the baseline (first file = RF off)
df0 = df0.sort_values(["Setpoint", "RF Power", "FileName"])  # Sort to ensure consistent ordering

# Create a baseline lookup: for each (Setpoint, RF Power), use the first file's FitPeakCenter
baseline_lookup = (
    df0.groupby(["Setpoint", "RF Power"])["FitPeakCenter"]
    .first()  # First file = RF off baseline
    .reset_index()
    .rename(columns={"FitPeakCenter": "BaselineCenter"})
)

# Merge baseline back to main dataframe
df0 = df0.merge(baseline_lookup, on=["Setpoint", "RF Power"], how="left")

# Calculate the shift
df0["FitPeakCenterShift"] = df0["FitPeakCenter"] - df0["BaselineCenter"]

print(f"Created {len(baseline_lookup)} baselines")

# --- 6) Filter out RF off data (baseline measurements) for plotting ---
# Get the baseline files (first file for each RF Power)
baseline_files = df0.groupby(["RF Power"])["FileName"].first().reset_index()
baseline_filenames = set(baseline_files["FileName"])

# Keep only RF on data (non-baseline files) for plotting
df0_plot = df0[~df0["FileName"].isin(baseline_filenames)].copy()

# Drop rows with missing essentials
df0_plot = df0_plot.dropna(subset=["RF Amplitude (V)", "FitPeakHeight", "FitPeakWidth"])

# Sort by amplitude for nicer lines
df0_plot = df0_plot.sort_values("RF Amplitude (V)")

print(f"Filtered out {len(baseline_filenames)} baseline files for plotting")
print(f"Plotting {len(df0_plot)} RF on measurements")

# --- 6.5) Perform fitting on the data ---
# Toggle to enable/disable the low-amplitude quadratic fit. The Lambert W fit
# already captures the small-amplitude behavior, so default is disabled.
ENABLE_QUADRATIC_FIT = False
# Separate data for different fitting regimes
high_amp_mask = df0_plot["RF Amplitude (V)"] > 0.8
low_amp_mask = df0_plot["RF Amplitude (V)"] < 0.1
lambert_mask = (df0_plot["RF Amplitude (V)"] >= 0.0) & (df0_plot["RF Amplitude (V)"] <= 0.175)

high_amp_data = df0_plot[high_amp_mask]
low_amp_data = df0_plot[low_amp_mask]
lambert_data = df0_plot[lambert_mask]

# Helper function to calculate R²
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Linear fit for high amplitude (RF Amplitude > 0.5)
if len(high_amp_data) > 1:
    x_high = high_amp_data["RF Amplitude (V)"].values
    y_high = high_amp_data["FitPeakCenterShift"].values
    
    # Linear fit using numpy polyfit (degree 1)
    linear_coef = np.polyfit(x_high, y_high, 1)
    slope, intercept = linear_coef
    
    # Predict values for R² calculation
    y_high_pred = slope * x_high + intercept
    r2_linear = calculate_r2(y_high, y_high_pred)
    
    # Create smooth line for plotting
    x_high_smooth = np.linspace(0.8, df0_plot["RF Amplitude (V)"].max(), 100)
    y_high_smooth = slope * x_high_smooth + intercept
    
    print(f"\nLinear fit (RF Amplitude > 0.8):")
    print(f"Equation: Shift = {slope:.2f} * Amplitude + {intercept:.2f}")
    print(f"R² = {r2_linear:.4f}")
else:
    print("Not enough data points for linear fit (RF Amplitude > 0.8)")
    x_high_smooth, y_high_smooth = None, None
    r2_linear = None

# Enhanced parabolic fit for low amplitude (RF Amplitude < 0.1) with robust techniques
# Disabled by default since Lambert W fit supersedes it
x_low_smooth, y_low_smooth = None, None
r2_parabolic = None
if ENABLE_QUADRATIC_FIT and len(low_amp_data) > 3:  # Need at least 4 points for robust fitting
    x_low = low_amp_data["RF Amplitude (V)"].values
    y_low = low_amp_data["FitPeakCenterShift"].values
    
    print(f"\nStarting parabolic fit with {len(x_low)} data points...")
    
    # Step 1: Multiple outlier detection methods
    def detect_outliers_multiple_methods(x, y):
        """Use multiple outlier detection methods and combine results"""
        n_points = len(x)
        outlier_scores = np.zeros(n_points)
        
        # Method 1: Z-score on y values
        if len(y) > 2:
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std > 0:
                z_scores = np.abs((y - y_mean) / y_std)
                outlier_scores += (z_scores > 2.0).astype(float)
        
        # Method 2: Interquartile range (IQR) method
        if len(y) > 4:
            q25, q75 = np.percentile(y, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                iqr_outliers = (y < lower_bound) | (y > upper_bound)
                outlier_scores += iqr_outliers.astype(float)
        
        # Method 3: Modified Z-score using median
        if len(y) > 2:
            median_y = np.median(y)
            mad = np.median(np.abs(y - median_y))
            if mad > 0:
                modified_z_scores = 0.6745 * (y - median_y) / mad
                outlier_scores += (np.abs(modified_z_scores) > 3.5).astype(float)
        
        # Keep points that are flagged as outliers by fewer than 2 methods
        good_mask = outlier_scores < 2
        return good_mask
    
    # Apply outlier detection
    good_mask = detect_outliers_multiple_methods(x_low, y_low)
    x_clean = x_low[good_mask]
    y_clean = y_low[good_mask]
    
    print(f"Outlier detection: kept {len(x_clean)} out of {len(x_low)} points")
    
    # Step 2: Try multiple fitting approaches
    best_fit = None
    best_r2 = -np.inf
    best_method = ""
    
    if len(x_clean) >= 3:
        # Approach 1: Direct polynomial fit on cleaned data
        try:
            coef1 = np.polyfit(x_clean, y_clean, 2)
            y_pred1 = np.polyval(coef1, x_clean)
            r2_1 = calculate_r2(y_clean, y_pred1)
            if r2_1 > best_r2:
                best_fit = coef1
                best_r2 = r2_1
                best_method = "direct polynomial"
        except:
            pass
        
        # Approach 2: Weighted polynomial fit (weight by inverse distance to median)
        try:
            if len(x_clean) > 3:
                # Calculate weights based on distance from median
                x_median = np.median(x_clean)
                y_median = np.median(y_clean)
                distances = np.sqrt((x_clean - x_median)**2 + (y_clean - y_median)**2)
                weights = 1.0 / (1.0 + distances)  # Higher weight for points closer to median
                
                coef2 = np.polyfit(x_clean, y_clean, 2, w=weights)
                y_pred2 = np.polyval(coef2, x_clean)
                r2_2 = calculate_r2(y_clean, y_pred2)
                if r2_2 > best_r2:
                    best_fit = coef2
                    best_r2 = r2_2
                    best_method = "weighted polynomial"
        except:
            pass
        
        # Approach 3: Binning approach with adaptive bins
        try:
            if len(x_clean) >= 6:
                # Sort data by x for binning
                sort_idx = np.argsort(x_clean)
                x_sorted = x_clean[sort_idx]
                y_sorted = y_clean[sort_idx]
                
                # Create adaptive bins based on data density
                n_bins = max(3, min(6, len(x_clean) // 3))
                bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
                
                bin_centers = []
                bin_means = []
                bin_weights = []
                
                for i in range(n_bins):
                    if i == n_bins - 1:
                        mask = (x_sorted >= bin_edges[i]) & (x_sorted <= bin_edges[i+1])
                    else:
                        mask = (x_sorted >= bin_edges[i]) & (x_sorted < bin_edges[i+1])
                    
                    if np.sum(mask) > 0:
                        bin_x = x_sorted[mask]
                        bin_y = y_sorted[mask]
                        
                        bin_centers.append(np.mean(bin_x))
                        bin_means.append(np.mean(bin_y))
                        bin_weights.append(len(bin_x))  # Weight by number of points in bin
                
                if len(bin_centers) >= 3:
                    bin_centers = np.array(bin_centers)
                    bin_means = np.array(bin_means)
                    bin_weights = np.array(bin_weights)
                    
                    coef3 = np.polyfit(bin_centers, bin_means, 2, w=bin_weights)
                    y_pred3 = np.polyval(coef3, x_clean)
                    r2_3 = calculate_r2(y_clean, y_pred3)
                    if r2_3 > best_r2:
                        best_fit = coef3
                        best_r2 = r2_3
                        best_method = f"adaptive binning ({len(bin_centers)} bins)"
        except:
            pass
    
    # Step 3: Use the best fit found
    if best_fit is not None and best_r2 > -0.5:  # Accept reasonable fits
        coef_quad, coef_linear, coef_const = best_fit
        
        # Create smooth curve for plotting
        x_low_smooth = np.linspace(0, 0.1, 100)
        y_low_smooth = coef_quad * x_low_smooth**2 + coef_linear * x_low_smooth + coef_const
        
        # Calculate R² on original cleaned data
        y_clean_pred = coef_quad * x_clean**2 + coef_linear * x_clean + coef_const
        r2_parabolic = calculate_r2(y_clean, y_clean_pred)
        
        print(f"Parabolic fit (RF Amplitude < 0.1, {best_method}):")
        print(f"Equation: Shift = {coef_quad:.3f} * Amplitude² + {coef_linear:.3f} * Amplitude + {coef_const:.3f}")
        print(f"R² = {r2_parabolic:.4f} (used {len(x_clean)} points, removed {len(x_low) - len(x_clean)} outliers)")
    else:
        print("No suitable parabolic fit found - all methods failed or gave poor results")
        x_low_smooth, y_low_smooth = None, None
        r2_parabolic = None
elif ENABLE_QUADRATIC_FIT:
    print("Not enough data points for parabolic fit (need > 3)")

# Lambert W function fit for amplitude 0 <= x <= 0.175, with robust fitting and outlier removal
if len(lambert_data) > 3:
    x_lambert_raw = lambert_data["RF Amplitude (V)"].values
    y_lambert_raw = lambert_data["FitPeakCenterShift"].values

    # Strict mask for 0 <= x <= 0.175
    mask_0_02 = (x_lambert_raw >= 0.0) & (x_lambert_raw <= 0.175) & np.isfinite(x_lambert_raw) & np.isfinite(y_lambert_raw)
    x_lambert = x_lambert_raw[mask_0_02]
    y_lambert = y_lambert_raw[mask_0_02]

    # More aggressive outlier removal using multiple methods
    if len(y_lambert) > 4:
        # Method 1: MAD-based outlier removal (more aggressive)
        y_med = np.median(y_lambert)
        mad = np.median(np.abs(y_lambert - y_med))
        if mad > 0:
            mz = 0.6745 * (y_lambert - y_med) / mad
            mad_good = np.abs(mz) < 2.5  # More aggressive than 3.5
        else:
            mad_good = np.ones(len(y_lambert), dtype=bool)
        
        # Method 2: IQR-based outlier removal
        q25, q75 = np.percentile(y_lambert, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            lower_bound = q25 - 1.0 * iqr  # More aggressive than 1.5
            upper_bound = q75 + 1.0 * iqr
            iqr_good = (y_lambert >= lower_bound) & (y_lambert <= upper_bound)
        else:
            iqr_good = np.ones(len(y_lambert), dtype=bool)
        
        # Combine both methods - keep points that pass both tests
        good = mad_good & iqr_good
        x_lambert = x_lambert[good]
        y_lambert = y_lambert[good]
        
        print(f"Lambert W outlier removal: kept {len(x_lambert)} out of {len(x_lambert_raw[mask_0_02])} points")

    # Define direct Lambert W model: y = a * W(b*x) + c
    def make_direct_model(branch):
        def model(p, x):
            a, b, c = p
            x_safe = np.maximum(x, 1e-12)
            return a * np.real(lambertw(b * x_safe, k=branch)) + c
        def residuals(p, x, y, w):
            return np.sqrt(w) * (model(p, x) - y)
        return model, residuals

    # Density-balanced weighting to avoid over-emphasis of dense low-amplitude region
    try:
        nbins = 30
        x_min_w, x_max_w = float(np.min(x_lambert)), float(np.max(x_lambert))
        if x_max_w > x_min_w:
            bin_edges = np.linspace(x_min_w, x_max_w, nbins + 1)
            bin_idx = np.digitize(x_lambert, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, nbins - 1)
            counts = np.bincount(bin_idx, minlength=nbins).astype(float)
            local_counts = counts[bin_idx]
            weights_density = 1.0 / np.maximum(local_counts, 1.0)
            # Normalize weights to mean ~1
            weights_density *= (len(weights_density) / np.sum(weights_density))
        else:
            weights_density = np.ones_like(x_lambert, dtype=float)
    except Exception:
        weights_density = np.ones_like(x_lambert, dtype=float)

    # Heuristic initialization
    x_rng = np.ptp(x_lambert) if np.ptp(x_lambert) > 0 else 1.0
    y_rng = np.ptp(y_lambert) if np.ptp(y_lambert) > 0 else 1.0
    c0 = np.median(y_lambert[x_lambert <= max(0.05, 0.02 * x_rng)]) if np.any(x_lambert <= max(0.05, 0.02 * x_rng)) else np.min(y_lambert)
    a0 = y_rng if y_rng > 0 else 1.0
    b0 = 1.0 / max(0.175, np.median(x_lambert))  # scale b by typical x
    p0 = np.array([a0, b0, c0], dtype=float)

    # Bounds to keep parameters sensible
    lower_bounds = np.array([-1e6, -1e3, np.min(y_lambert) - 5.0*y_rng - 1e3], dtype=float)
    upper_bounds = np.array([ 1e6,  1e3,  np.max(y_lambert) + 5.0*y_rng + 1e3], dtype=float)

    # Robust least squares with soft L1 loss (DIRECT FORM), try both branches k=0 and k=-1
    best_direct = None
    best_direct_r2 = -np.inf
    direct_x_smooth, direct_y_smooth = None, None
    popt_dir, r2_dir, direct_branch = None, None, None
    for branch in (0, -1):
        model, residuals_direct = make_direct_model(branch)
        try:
            lsq = least_squares(
                residuals_direct,
                p0,
                args=(x_lambert, y_lambert, weights_density),
                bounds=(lower_bounds, upper_bounds),
                loss='soft_l1',
                f_scale=1.0,
                max_nfev=20000,
            )
            popt_try = lsq.x
            y_pred_try = model(popt_try, x_lambert)
            r2_try = calculate_r2(y_lambert, y_pred_try)
            if r2_try > best_direct_r2:
                best_direct_r2 = r2_try
                best_direct = (popt_try, branch)
        except Exception:
            continue
    if best_direct is not None:
        popt_dir, direct_branch = best_direct
        r2_dir = best_direct_r2
        x_max_plot = min(0.175, float(np.nanmax(df0_plot["RF Amplitude (V)"].values)))
        direct_x_smooth = np.linspace(0.01, x_max_plot, 200)
        model, _ = make_direct_model(direct_branch)
        direct_y_smooth = model(popt_dir, direct_x_smooth)
        print(f"\nLambert W (direct robust, branch k={direct_branch}) 0<=Amp<=0.175: R²={r2_dir:.4f}; a={popt_dir[0]:.3g}, b={popt_dir[1]:.3g}, c={popt_dir[2]:.3g}")
    else:
        print("Lambert W direct robust fit failed for both branches")

    # Inverse form: x = a * W(b * (y - c)) + d
    def lambert_w_inverse_func_params(p, y):
        a, b, c, d = p
        y_shifted = np.maximum(y - c, 1e-12)
        return a * np.real(lambertw(b * y_shifted)) + d

    def residuals_inverse(p, y, x, w):
        return np.sqrt(w) * (lambert_w_inverse_func_params(p, y) - x)

    # Density weights in y-space to counter uneven sampling in y if present
    try:
        nbins_y = 30
        y_min_w, y_max_w = float(np.min(y_lambert)), float(np.max(y_lambert))
        if y_max_w > y_min_w:
            y_edges = np.linspace(y_min_w, y_max_w, nbins_y + 1)
            y_idx = np.digitize(y_lambert, y_edges) - 1
            y_idx = np.clip(y_idx, 0, nbins_y - 1)
            y_counts = np.bincount(y_idx, minlength=nbins_y).astype(float)
            y_local_counts = y_counts[y_idx]
            weights_y_density = 1.0 / np.maximum(y_local_counts, 1.0)
            weights_y_density *= (len(weights_y_density) / np.sum(weights_y_density))
        else:
            weights_y_density = np.ones_like(y_lambert, dtype=float)
    except Exception:
        weights_y_density = np.ones_like(y_lambert, dtype=float)

    # Initial guesses for inverse parameters
    y_rng = np.ptp(y_lambert) if np.ptp(y_lambert) > 0 else 1.0
    x_med = np.median(x_lambert)
    p0_inv = np.array([max(x_med, 0.05), 1.0 / max(0.1, y_rng), np.min(y_lambert), np.median(x_lambert)], dtype=float)
    lb_inv = np.array([-1e3, 1e-6, np.min(y_lambert) - 10.0*y_rng - 1e3, -1.0], dtype=float)
    ub_inv = np.array([ 1e3, 1e3,  np.max(y_lambert) + 10.0*y_rng + 1e3,  1.0], dtype=float)

    try:
        lsq_inv = least_squares(
            residuals_inverse,
            p0_inv,
            args=(y_lambert, x_lambert, weights_y_density),
            bounds=(lb_inv, ub_inv),
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=20000,
        )
        popt_inv = lsq_inv.x
        x_pred_inv = lambert_w_inverse_func_params(popt_inv, y_lambert)
        r2_inv = calculate_r2(x_lambert, x_pred_inv)

        # Generate y(x) curve via numeric inversion using brentq
        x_max_plot = min(0.175, float(np.nanmax(df0_plot["RF Amplitude (V)"].values)))
        inv_x_smooth = np.linspace(0.01, x_max_plot, 200)
        inv_y_smooth = []
        y_data_min, y_data_max = float(np.min(y_lambert)), float(np.max(y_lambert))
        y_span = max(1e-6, y_data_max - y_data_min)
        y_lo_base = y_data_min - 3.0 * y_span
        y_hi_base = y_data_max + 3.0 * y_span
        for xt in inv_x_smooth:
            def g(y):
                return lambert_w_inverse_func_params(popt_inv, y) - xt
            y_lo, y_hi = y_lo_base, y_hi_base
            glo, ghi = g(y_lo), g(y_hi)
            # Expand bracket if needed up to a few times
            expand = 0
            while glo * ghi > 0 and expand < 5:
                y_lo -= y_span
                y_hi += y_span
                glo, ghi = g(y_lo), g(y_hi)
                expand += 1
            try:
                y_root = brentq(g, y_lo, y_hi, maxiter=200)
                inv_y_smooth.append(y_root)
            except Exception:
                inv_y_smooth.append(np.nan)
        inv_y_smooth = np.array(inv_y_smooth)
        valid = np.isfinite(inv_y_smooth)
        inv_x_smooth = inv_x_smooth[valid]
        inv_y_smooth = inv_y_smooth[valid]
        print(f"Lambert W (inverse robust) fit on 0<=Amplitude<=0.175: R²={r2_inv:.4f}; params a={popt_inv[0]:.3g}, b={popt_inv[1]:.3g}, c={popt_inv[2]:.3g}, d={popt_inv[3]:.3g}")
    except Exception as e:
        print(f"Lambert W inverse robust fit failed: {e}")
        popt_inv = None
        r2_inv = None
        inv_x_smooth, inv_y_smooth = None, None

    # Choose which curve to display: prefer inverse if it yields convex upward shape and sufficient coverage
    def is_convex_up(xc, yc):
        if xc is None or yc is None or len(xc) < 5:
            return False
        # compute discrete second derivative sign
        dy = np.diff(yc)
        d2y = np.diff(dy)
        return np.nanmedian(d2y) > 0

    use_inverse = False
    if inv_x_smooth is not None and inv_y_smooth is not None and len(inv_x_smooth) > 20:
        if is_convex_up(inv_x_smooth, inv_y_smooth):
            use_inverse = True

    if use_inverse:
        x_lambert_smooth, y_lambert_smooth = inv_x_smooth, inv_y_smooth
        # If inverse curve does not reach down to 0.01, and direct model exists, prepend using direct
        if (
            x_lambert_smooth is not None and len(x_lambert_smooth) > 0 and
            popt_dir is not None and direct_x_smooth is not None and direct_y_smooth is not None
        ):
            min_x = float(np.min(x_lambert_smooth))
            if min_x > 0.011:  # leave a tiny gap tolerance
                x_fill = np.linspace(0.01, min_x, 50, endpoint=False)
                model_fill, _ = make_direct_model(direct_branch)
                y_fill = model_fill(popt_dir, x_fill)
                x_lambert_smooth = np.concatenate([x_fill, x_lambert_smooth])
                y_lambert_smooth = np.concatenate([y_fill, y_lambert_smooth])
        r2_lambert = r2_inv
    else:
        x_lambert_smooth, y_lambert_smooth = direct_x_smooth, direct_y_smooth
        r2_lambert = r2_dir
else:
    print("Not enough data points for Lambert W fit (need > 3)")
    x_lambert_smooth, y_lambert_smooth = None, None
    r2_lambert = None

# Logistic step fit for width vs amplitude
logistic_x_smooth, logistic_y_smooth = None, None
r2_logistic_width = None
try:
    xw = df0_plot["RF Amplitude (V)"].values
    yw = df0_plot["FitPeakWidth"].values
    valid = np.isfinite(xw) & np.isfinite(yw)
    xw = xw[valid]
    yw = yw[valid]

    # Outlier removal on width (IQR)
    if len(yw) > 4:
        q1, q3 = np.percentile(yw, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            low_b = q1 - 1.5 * iqr
            high_b = q3 + 1.5 * iqr
            keep = (yw >= low_b) & (yw <= high_b)
            xw = xw[keep]
            yw = yw[keep]

    # Logistic function: y = y_low + (y_high - y_low) / (1 + exp(-(x - x0)/k))
    def logistic_func(x, y_low, y_high, x0, k):
        return y_low + (y_high - y_low) / (1.0 + np.exp(-(x - x0) / k))

    # Initial guesses
    y_low0 = np.percentile(yw, 10)
    y_high0 = np.percentile(yw, 90)
    x0_0 = np.median(xw)
    k0 = max(1e-3, 0.05)
    p0_log = [y_low0, y_high0, x0_0, k0]

    popt_log, _ = curve_fit(logistic_func, xw, yw, p0=p0_log, maxfev=8000)
    y_pred_log = logistic_func(xw, *popt_log)
    r2_logistic_width = calculate_r2(yw, y_pred_log)

    # Smooth curve for plotting
    x_min, x_max = np.min(xw), np.max(xw)
    logistic_x_smooth = np.linspace(x_min, x_max, 200)
    logistic_y_smooth = logistic_func(logistic_x_smooth, *popt_log)

    print(f"Logistic fit (width): R²={r2_logistic_width:.4f}; params y_low={popt_log[0]:.2f}, y_high={popt_log[1]:.2f}, x0={popt_log[2]:.3f}, k={popt_log[3]:.3f}")
except Exception as e:
    print(f"Logistic fit failed: {e}")

# --- 7) Plot: FitPeakCenterShift vs Amplitude with fits ---
fig_h = px.scatter(
    df0_plot,
    x="RF Amplitude (V)",
    y="FitPeakCenterShift",
    hover_data=["FileName", "RF Power", "FitPeakCenter", "BaselineCenter"],
    title=f"Peak #{peak_number+1} @ Setpoint {setpoint/1e3} nA — FitPeakCenterShift vs RF Amplitude",
    labels={"RF Amplitude (V)": "RF Amplitude (V)", "FitPeakCenterShift": "FitPeakCenterShift (mV)"},
)
fig_h.update_traces(mode="markers", marker=dict(color='black', size=6))

# Add fit lines
if x_high_smooth is not None and y_high_smooth is not None:
    fig_h.add_trace(go.Scatter(
        x=x_high_smooth,
        y=y_high_smooth,
        mode='lines',
        name=f'Linear Fit (R²={r2_linear:.3f})',
        line=dict(color='red', width=4, dash='dot')
    ))

if x_low_smooth is not None and y_low_smooth is not None:
    fig_h.add_trace(go.Scatter(
        x=x_low_smooth,
        y=y_low_smooth,
        mode='lines',
        name=f'quadratic Fit (R²={r2_parabolic:.3f})',
        line=dict(color='orange', width=4, dash='dash')
    ))

if x_lambert_smooth is not None and y_lambert_smooth is not None:
    fig_h.add_trace(go.Scatter(
        x=x_lambert_smooth,
        y=y_lambert_smooth,
        mode='lines',
        name=f'Lambert W Fit (R²={r2_lambert:.3f})' if r2_lambert is not None else 'Lambert W Fit',
        line=dict(color='blue', width=4, dash='dot')
    ))

# Set y-axis range and update layout
fig_h.update_yaxes(range=[-100, 1100])

# Remove color bar and move legend to top
fig_h.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig_h.update_coloraxes(showscale=False)  # Remove color bar

# Make axis labels and tick labels larger for visibility
fig_h.update_layout(
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24),
)
fig_h.update_xaxes(tickfont=dict(size=18))
fig_h.update_yaxes(tickfont=dict(size=18))

# --- 8) Plot: FitPeakWidth vs Amplitude (RF Amplitude on x-axis) ---
fig_w = px.scatter(
    df0_plot,
    x="RF Amplitude (V)",
    y="FitPeakWidth",
    hover_data=["FileName", "RF Power", "FitPeakCenter", "BaselineCenter"],
    title=f"Peak #{peak_number+1} @ Setpoint {setpoint/1e3} nA — FitPeakWidth vs RF Amplitude",
    labels={"RF Amplitude (V)": "RF Amplitude (V)", "FitPeakWidth": "FitPeakWidth"},
)
fig_w.update_traces(mode="markers", marker=dict(color='black', size=6))

# Set requested y-axis range for width plot
fig_w.update_yaxes(range=[280, 400])

# Add logistic fit line to width plot
if logistic_x_smooth is not None and logistic_y_smooth is not None:
    fig_w.add_trace(go.Scatter(
        x=logistic_x_smooth,
        y=logistic_y_smooth,
        mode='lines',
        name=f'Logistic Fit (R²={r2_logistic_width:.3f})' if r2_logistic_width is not None else 'Logistic Fit',
        line=dict(color='purple', width=4, dash='dash')
    ))

# Remove color bar and move legend to top for width plot
fig_w.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig_w.update_coloraxes(showscale=False)  # Remove color bar

# Make axis labels and tick labels larger for visibility
fig_w.update_layout(
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24),
)
fig_w.update_xaxes(tickfont=dict(size=18))
fig_w.update_yaxes(tickfont=dict(size=18))

fig_h.show()
fig_w.show()

# --- Summary info ---
print(f"Loaded {len(df0):,} measurements for Setpoint={setpoint}, Peak={peak_number}")
print(f"RF Power range: {df0['RF Power'].min():.1f} to {df0['RF Power'].max():.1f} dBm")
