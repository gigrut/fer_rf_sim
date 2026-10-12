# Transmission Plotting And Performance

## Axis Contracts

Current arrays are `(z, V, A)`, LUTs are `(z, V, E)`, and RF-averaged transmission
is `(z, V, A, E)`. Plot surfaces and heatmaps use rows for y and columns for x.
Tests use unequal axis lengths to expose accidental transposes.

Two-dimensional transmission plots require two fixed axes or an explicit
reduction. `requested_transmission_grid` reduces fixed axes before evaluating
RF phases, so a V/A surface at fixed height and energy does not build the full
four-dimensional grid. The full LUT is currently loaded into memory; this is not
an on-disk lazy-slice implementation. Legacy energy-averaged helpers explicitly
average samples, not a physical current integral.

```powershell
fer-sim transmission --lut-path path/to/lut.h5 --config fer_config.yaml --x V --y A --z 0.3 --e 1.0
fer-sim transmission --lut-path path/to/lut.h5 --x E --y V --z 0.3 --a 0 --html transmission.html
```

Fixed values select the nearest in-range configured sample. Omitted fixed values
use the first configured sample. Out-of-range values and instantaneous RF voltages
outside LUT coverage are errors, not zero-filled transmission. The config supplies
plot sampling axes; the explicitly selected LUT supplies transmission physics.
Energy is eV, height nm, bias V and RF amplitude V peak. The legacy `x` energy-axis
alias remains accepted by the Python slice helper; the CLI calls it `E`.

Current figures consume simulated/integrated current, never a sum of transmission
samples. RF current integrates the instantaneous occupation window at each phase
before averaging. Direct and LUT backends share batched current integration, with
bounded amplitude chunks; height rows can run through joblib.

## Benchmark

```powershell
python benchmark_lut_vs_direct.py --repeats 3
```

This runs the actual service on matched small inputs in a temporary directory.
It reports direct time, cold-LUT time, LUT build time, repeated warm-LUT time,
cache reuse, traced peak memory, absolute error and relative error at nonzero
direct-current cells. Tiny physical values are not hidden by an arbitrary error
denominator offset. Benchmark files are removed with their temporary directory.

Measurements are single-process wall time with cProfile and tracemalloc enabled,
including calibration and HDF5 I/O. Traced allocation peaks are **not total RSS**
and exclude untracked native/child-process allocations. These measurements do not
predict production speed or impose hardware-specific CI timing thresholds.

## Profile

```powershell
python profile_simulation.py --n_E 16 --n_V 4 --n_Z 2 --n_A 2 --n_cheb 4 --threads 1 --out profile-results
```

The profiler calls `SimulationRunner` directly and writes a uniquely named profile
under the selected output's `profiles` directory, or `--profile-dir`. It does not
replace the repository's existing profile. SnakeViz is optional and starts only
when explicitly requested with `--snakeviz`; install it separately first.

## Desktop Rendering

The optional GUI uses one local Plotly HTML document shared by both plot views.
Updates use `Plotly.react` without rewriting HTML or loading Plotly from a CDN.
Derivatives are computed once per loaded artifact, and the setpoint cache has a
16-entry limit with invalidation on reload. No new peak/RF GUI feature is implied.