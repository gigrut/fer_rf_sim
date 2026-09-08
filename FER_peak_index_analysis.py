import os
import shutil
import pandas as pd

import numpy as np
#from ipywidgets import interact, widgets
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, ifft, fftfreq
from collections import defaultdict
#import statsmodels.api as sm
#import sympy as sp
import math
import warnings
import plotly.graph_objects as go
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy", message="divide by zero encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy", message="invalid value encountered in multiply")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def dbm_to_mw(dBm):
    mW = 10**(dBm/10)
    return mW
def dbm_to_mv(dBm):
    mV = 2*2**0.5 * (50/1000)**(0.5) * (10**(dBm/20))
    return mV
def broadenCGQ(channel, bias, v_rf, n):
    """channel is the thing to be broadened"""
    # n is going to be the number of points in the convolution
    if type(v_rf) == np.ndarray:
        v_rf = v_rf.item()
    #Step 1: Account for uneven spacing
    regular_bias = np.linspace(min(bias), max(bias), len(bias))
    bias_step = regular_bias[1] - regular_bias[0]
    invalid_pts = int(v_rf // bias_step) # might be off by 1
    truncated_bias = regular_bias[invalid_pts:-invalid_pts] # might be off by 1
    samples = np.array(range(1,n+1))
    interp_function = interp1d(bias, channel, kind='linear', fill_value='extrapolate')
    broadened_data = np.zeros(len(bias)-2*invalid_pts)
    for i,v in enumerate(truncated_bias):
        num = np.cos(np.pi*(2*samples-1)/(2*n))
        x = v_rf * num + v
        broadened_data[i] = np.sum(interp_function(x)/n)
    M = len(bias) - len(broadened_data)
    pad1 = pad2 = M // 2
    if M % 2 != 0:
        pad2 += 1
    padded = np.pad(broadened_data, (pad1, pad2), 'constant', constant_values=(np.nan, np.nan))
    conv_result  = pd.DataFrame({'bias': regular_bias, 'broadened channel': padded})
    return conv_result
def round_to_1sf(x):
    if x<=0:
        return x
    if x >= 1:
        return round(x, 1)
    else:
        power_of_10 = 10 ** math.floor(math.log10(x))
        return round(x, -int(math.log10(power_of_10)))
def fit_quality(params, spectrum):
    z, phi = params
    for peak in spectrum:
        peak['field'] = peak['bias'] / (peak['z-topo'] + z)
        peak['fn'] = peak['field'] * peak['number']
    peak_fn = np.array([peak['fn'] for peak in spectrum])
    peak_biases = np.array([peak['bias'] for peak in spectrum])
    #print('peak biases[w:]', peak_biases[w:])
    peak_biases_transformed = (peak_biases - phi) ** (3/2)  
    #print(peak_biases_transformed)
    coeffs = np.polyfit(peak_fn[w:], peak_biases_transformed[w:], 1)
    fit_biases_transformed = np.polyval(coeffs, peak_fn)
    errors = peak_biases_transformed - fit_biases_transformed
    return np.sum(errors**2)

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

# w = number of points to ignore in linear fit
w=4
# Parameters for the Savitzky-Golay filter
window_length = 51  # Must be a positive odd integer
polyorder = 4       # The order of the polynomial used to fit the samples
prominence = 25
distance = 10

folder_path = "C:/Users/willh/OneDrive/Desktop/Data/z_dependent_fer_shift_csv"
read = 1
if read:
    file_list = os.listdir(folder_path)
    dfs = []
    file_names = []
    for file in file_list:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
            file_names.append(file_path[-17:])
    # Powers (in dBm) = ['off', 10, 15, 17.5, 20]
    # Z shift for each power:
    zs = [0, 0.0035, 0.01, 0.016, 0.027]
    #zs = [0,.056,.18,.427,.747]
    # No attenuation
    #vrfs = [0, 1, 1.7783, 2.37135, 3.1623]
    # 5dB attenuation
    vrfs = [0, 0.56, 1, 1.33, 1.78] 
    # 10 dB attenuation
    #vrfs = [0, 0.32, 0.56, 0.75, 1]
    peak_data = []
    j=0
    for i,df in enumerate(dfs):
        df['RF (V)'] = vrfs[j]
        drift_correction = df['z-topo (nm)'].iloc[185] - dfs[i%5]['z-topo (nm)'].iloc[275] + (i%5)*0.2/4
        df['tip-sample-distance (nm)'] = 5.2 + zs[j] + df['z-topo (nm)'] - drift_correction
        # Apply Savitzky-Golay filter to the 'lockin-x (mV)' column
        df['smoothed'] = savgol_filter(df['lockin-x (mV)'], window_length, polyorder)
        # Filter
        filtered_df = df[(df['bias (mV)'] > 3000) & (df['bias (mV)'] < 9700)]
        peaks, _ = find_peaks(filtered_df['smoothed'], distance=distance, prominence=prominence)
        peak_indices = [filtered_df.index[peak] for peak in peaks]
        peak_info = [
            {
                'number': 1+k,
                'index': peak_indices[k],
                'bias': df['bias (mV)'].iloc[peak_indices[k]],
                'lockin-x': df['smoothed'].iloc[peak_indices[k]],
                'current': round_to_1sf(df['curr (nA)'].iloc[-1]),
                'RF': round_to_1sf(df['RF (V)'].iloc[-1]),
                'z-topo': round(df['z-topo (nm)'].iloc[-1], 3),
                'tip-sample-distance': round(df['tip-sample-distance (nm)'].iloc[-1], 3)
            }
            for k in range(len(peak_indices))
        ] 
        peak_data.append(peak_info)
        # peak_data is a list of lists. Each sublist corresponds to a spectrum. The elements of the sublist are dictionaries. Each dictionary contains information about a peak in that spectrum.
        if i%5==4:
            j+=1
            
    print('finished reading files')


    
a = 'bias (mV)'
#b = 'tip-sample-distance (nm)'
#b = 'z-topo (nm)'
b = 'lockin-x (mV)'
#b = 'F'
c = 'lockin-x (mV)'
#c = 'curr (nA)'
y_axis_type = 'linear'
if c == 'curr (nA)':
    marker_dict = dict(
                size=5,
                color=np.log10(df[c]),
                colorscale='Viridis',
                cmin=np.log10(0.01),
                cmax=np.log10(5.0),
                colorbar=dict(
                    title=c, 
                    tickvals=[
                        np.log10(0.03), np.log10(0.1), np.log10(0.3), np.log10(1.0), np.log10(3.0)
                    ], 
                    ticktext=['0.03','0.1','0.3','1.0','3.0'], tickmode='array'),
                showscale=True if i==0 else True,
            )
elif c == 'lockin-x (mV)':
    marker_dict = dict(
                size=5,
                color=np.log10(df[c]),
                colorscale='Viridis',
                cmin=np.log10(2E4),
                cmax=np.log10(1E2),
                colorbar=dict(title=c),
                showscale=True if i==0 else True,
            )

fig = go.Figure()
fig.update_layout(
    title='FER',
    xaxis_title=a,
    yaxis_title=b,
    yaxis=dict(type=y_axis_type),
    margin=dict(l=50, r=50, t=50, b=50),  # Adjust left, right, top, bottom margins
    showlegend=True,
    legend=dict(
        x=1.1,  # Position the legend to the right of the plot
        xanchor='left',
        y=1,
        yanchor='top'
    )
)
fig.update_traces(marker=dict(colorbar=dict(nticks=5)))

# Plotting spectra
spectra_traces = []
smoothed_traces = []
peak_traces = []
j=0
for i,df in enumerate(dfs):
    current = round_to_1sf(df["curr (nA)"].iloc[-1])
    rf = df["RF (V)"][0]
    df['F'] = df['bias (mV)'] / df['tip-sample-distance (nm)']
    trace1 = go.Scatter(
        x=df[a],
        y=df[b],
        mode='lines+markers',
        marker=marker_dict,
        line=dict(
        width=1,
        color=RF_to_color[round(df["RF (V)"][0],1)],
        ),
        hovertext=f'RAW: RF = {df["RF (V)"][0]}V, I = {round_to_1sf(df["curr (nA)"].iloc[-1])} nA',
        name=f'RAW: RF = {df["RF (V)"][0]}V, I = {round_to_1sf(df["curr (nA)"].iloc[-1])} nA'
    )
    trace_smoothed = go.Scatter(
        x=df[a],
        y=df['smoothed'],
        mode='lines',
        line=dict(
        width=1,
        color='black',
        ),
        hovertext=f'RF = {df["RF (V)"][0]}V, I = {round_to_1sf(df["curr (nA)"].iloc[-1])} nA',
        name=f'SMOOTHED, RF = {df["RF (V)"][0]}V, I = {round_to_1sf(df["curr (nA)"].iloc[-1])} nA'
    )
    spectra_traces.append(trace1)
    smoothed_traces.append(trace_smoothed)

    peak_biases = []
    peak_lockins = []
    for peak_dict in peak_data[i]:
        bias = peak_dict['bias']
        lockin_x = peak_dict['lockin-x']
        index = peak_dict['index']
        number = peak_dict['number']
        current = peak_dict['current']
        RF = peak_dict['RF']
        z_topo = peak_dict['z-topo']
        peak_biases.append(bias)
        peak_lockins.append(lockin_x)
        
    trace_peaks = go.Scatter(
        x = peak_biases,
        y = peak_lockins,
        mode='markers',
        marker_symbol='star',
        marker_size=10,
        marker_color=RF_to_color[RF],
        name=f'Peaks: Current: {current}, RF: {rf}',
        hoverinfo='text',
        showlegend=True
    )
    peak_traces.append(trace_peaks)          
    if i%5==4:
        j+=1
for trace1, trace2, trace3 in zip(spectra_traces, smoothed_traces, peak_traces):
    fig.add_trace(trace1)
    #fig.add_trace(trace2)
    #fig.add_trace(trace3)
    pass

plot_raw_spectra = 0
if plot_raw_spectra:
    fig.show()





















fig = go.Figure()

variable_mapping = {
    'a': 'number',
    'b': 'bias',
    'c': 'z-topo (nm)'
}

TARGET_CURRENT = 1.0
TARGET_RF = 0.0

for i,spectrum in enumerate(peak_data):
    line_color = 'black'
    # # Polynomial fit
    # initial_guess = (5.0, 4)
    # bounds = [(4.7,6),(4,5)]
    # # Optimize
    # optimal = minimize(lambda params: fit_quality(params, spectrum), initial_guess, method='L-BFGS-B', bounds=bounds)
    # Z = optimal.x[0]
    # phi = optimal.x[1]

    Z=5
    phi = 4.5e3
    
    for peak in spectrum:
        peak['field'] = peak['bias'] / (peak['z-topo'] + Z)
        peak['fn'] = peak['field'] * peak['number']

    # Extract data for plotting
    peak_numbers = [peak['number'] for peak in spectrum]
    peak_biases = [peak['bias'] for peak in spectrum]
    peak_z_topos = [peak['z-topo'] for peak in spectrum]
    peak_z_rels = [peak['tip-sample-distance'] for peak in spectrum]
    peak_fields = [peak['field'] for peak in spectrum]
    peak_fn = [peak['fn'] for peak in spectrum]
    peak_RFs = [peak['RF'] for peak in spectrum]
    peak_currents = [peak['current'] for peak in spectrum]
    current = peak_currents[0]
    rf_value = peak_RFs[0]
    
    if current == TARGET_CURRENT and rf_value == TARGET_RF: 
        x = np.array([peak[variable_mapping['a']] for peak in spectrum])
        x = (x - 0.25) ** (2/3)
        y = np.array([peak[variable_mapping['b']] for peak in spectrum]) / 1000  # Convert mV to V
        #Trace for bias vs n
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color='black', size=20, symbol='circle'),
            name=f'Data (I={TARGET_CURRENT} nA, RF={TARGET_RF} V, Z=5)',
            text=[f'Peak Index: {peak["index"]}, Bias: {peak["bias"]:.1f} mV, Current: {peak["current"]}, RF: {peak["RF"]}, Z-Topo: {peak["z-topo"]}' for peak in spectrum],
            hoverinfo='text'
        )  
        coeffs = np.polyfit(x[w:], y[w:] , 1)
        dfs[i]['slope'] = coeffs[0]
        dfs[i]['intercept'] = coeffs[1]
        
        # Calculate field F and work function
        alpha = 0.9458  # Physical constant in desired units
        field_F = coeffs[0] / alpha  # slope / alpha gives field F
        work_function = coeffs[1]  # y-intercept gives work function
        
        print(f"Field F = {field_F:.4f}")
        print(f"Work function = {work_function:.4f} V")
        x_fit = np.linspace(0, max(x), 100)
        y_fit = np.polyval(coeffs, x_fit) 
        trace_fit = go.Scatter(
            x=x_fit,
            y=y_fit ,
            mode='lines',
            #line_color=line_color,
            line=dict(color='black', width=5),
            name=f'Z={round(Z,2)}, phi={round(phi,3)}'
        )
        # Add a scatter plot trace for the peaks in this sublist
        fig.add_trace(trace)
        fig.add_trace(trace_fit)

# Add labels and title
fig.update_layout(
    title='FER',
    xaxis_title='(peak ' + variable_mapping['a'] + '-1/4)^(2/3)',
    yaxis_title='peak ' + variable_mapping['b'] + ' (V)',
    xaxis=dict(title_font=dict(size=40), tickfont=dict(size=32)),
    yaxis=dict(title_font=dict(size=40), tickfont=dict(size=32)),
    showlegend=False
)
fig.update_xaxes(range=[0, 5])
# Show the plot
fig.show()





def add_trace_to_figure(fig, RF, values, x_values, y_key, current_to_shape, RF_to_color, plot_title, x_title, y_title):
    fig.add_trace(go.Scatter(
        x=x_values,
        y=values[y_key],
        mode='markers',
        marker=dict(
            size=10,
            color=RF_to_color[RF],
            symbol=[current_to_shape[current] for current in values['currents']]
        ),
        name=f'RF: {RF} V'
    ))
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title='RF Values'
    )
# Initialize the data structure for organizing data by RF
data_by_RF = defaultdict(lambda: {'currents': [], 'slopes': [], 'intercepts': []})
# Process each DataFrame and organize data by RF
for df in dfs:
    try:
        current = round_to_1sf(df['curr (nA)'].iloc[-1])
        if current > 0.03:
            RF = round_to_1sf(df['RF (V)'][0])
            data_by_RF[RF]['currents'].append(current)
            data_by_RF[RF]['slopes'].append(round_to_1sf(df['slope'][0]))
            data_by_RF[RF]['intercepts'].append(round_to_1sf(df['intercept'][0]))
    except KeyError:
        continue
# Slope vs. Current
fig_slope_vs_current = go.Figure()
for RF, values in data_by_RF.items():
    add_trace_to_figure(fig_slope_vs_current, RF, values, values['currents'], 'slopes', current_to_shape, RF_to_color, 'Slope vs. Current for Different RF Values', 'Current (nA)', 'Slope')
# Intercept vs. RF
fig_intercept_vs_rf = go.Figure()
for RF, values in data_by_RF.items():
    add_trace_to_figure(fig_intercept_vs_rf, RF, values, [RF] * len(values['intercepts']), 'intercepts', current_to_shape, RF_to_color, 'Intercept vs. RF', 'RF (V)', 'Intercept')
# Slopes vs. RF
fig_rf_vs_slopes = go.Figure()
for RF, values in data_by_RF.items():
    add_trace_to_figure(fig_rf_vs_slopes, RF, values, [RF] * len(values['slopes']), 'slopes', current_to_shape, RF_to_color, 'Slopes vs. RF', 'RF (V)', 'Slope')
# Current vs. Intercepts
fig_current_vs_intercepts = go.Figure()
for RF, values in data_by_RF.items():
    add_trace_to_figure(fig_current_vs_intercepts, RF, values, values['currents'], 'intercepts', current_to_shape, RF_to_color, 'Intercepts vs. Current', 'Current (nA)', 'Intercept')
fit_plots = 0
if fit_plots:
    fig_slope_vs_current.show()
    fig_intercept_vs_rf.show()
    fig_rf_vs_slopes.show()
    fig_current_vs_intercepts.show()








print('done')


