# FER Workflow

Install the package with plotting extras as described in [README](README.md).
Root scripts remain compatibility adapters; the examples use installed commands.

## Run And Reuse An Artifact

```powershell
fer-sim workflow --quick-test --threads 1
fer-sim workflow --quick-test --use-lut --threads 1
fer-sim workflow --skip-simulation --h5-path path/to/result.h5 --steps quick-check constant-current
```

Simulation runs in-process through `SimulationRunner` and returns a `Path`.
Every subsequent step receives that exact path. When skipping simulation,
`--h5-path` selects an artifact explicitly; otherwise the newest parameterized
artifact in `--out` is resolved once (with `current.h5` as a legacy fallback).
Analysis subprocesses use the active interpreter and installed modules, not a
literal `python` command or working-directory-dependent scripts. Downstream
failures produce a nonzero workflow exit status.
`--no-show` builds and validates the analysis figures without opening a browser;
it is also accepted by `check` and `constant-current`. Plotting extras are still
required for figure construction.

The quick preset uses 50 energy, 20 bias, 5 height and 3 RF samples unless an
explicit grid-count option overrides it. For a smaller
headless check without plotting, use explicit counts:

```powershell
fer-sim workflow --steps simulation --n_E 16 --n_V 4 --n_Z 2 --n_A 2 --n_cheb 4 --threads 1 --out temporary-results
```

All simulation options use the shared config loader. Built-ins are 100/100/100/30
samples for energy/bias/height/RF, 32 RF phase nodes, and direct calculation.
The repository YAML overrides some of these values. Use `--no-lut` for an explicit
false override, even when YAML enables LUTs. Grid counts and ranges are validated
before work starts. Avoid using the full default grid as a smoke test.

## Analyze

```powershell
fer-sim check --h5-path path/to/result.h5 --plot-type current
fer-sim check --h5-path path/to/result.h5 --plot-type dlog
fer-sim constant-current --h5-path path/to/result.h5 --I-target 1e-9 --a-val 0
fer-sim constant-current --h5-path path/to/result.h5 --multiple 1e-10 1e-9 --a-idx 0
```

`--I-target` and `--multiple` are alternatives. Multiple-current comparisons
default to the lowest available RF amplitude if none is selected. In workflow
mode only, an omitted target is estimated from the finite positive currents;
this is not a guarantee that every voltage/RF trace reaches that setpoint.

The analysis computes the fixed-height derivative `dI/dV` on the artifact grid,
then interpolates that derivative at the inverted constant-current height. It
does **not** compute a total derivative along the constant-current contour.
Unreachable, nonmonotonic, flat, or nonfinite traces yield NaNs, not extrapolated
heights. Invalid axes and nonfinite target values are errors. Derivatives are
reused across multi-current analysis and cached once per artifact in the GUI.

All current values and derivatives remain in normalized model units. A target
such as `1e-9` must not be interpreted as one nanoampere.

## Desktop

`fer-gui` provides simulated/real-data views, shared peak extraction, and RF
peak-dependence analysis. It uses shared config and analysis, preserves untouched YAML values, and
bounds the setpoint cache to 16 entries. Selecting an artifact invalidates that
cache. Plotly views share one local HTML document and update in place. Closing
during a run waits for completion; it does not terminate numerical workers.
Temporary HTML is removed on close. Without Qt WebEngine, the window retains its
fallback labels, extraction tables and CSV export rather than requiring a browser component.

Choose a source in Peak Extraction, select the simulated signal and current
target or height when applicable, and extract peaks. RF Dependence uses those
same measurements, excluding spectra with unknown RF metadata. Real RF
assignments made with Set RF are session-only. RF matching uses a configurable
maximum bias shift; peak shifts use the lowest observed RF per track as the
reference. Changing extraction settings or loading a source invalidates both
analyses. See [peak workflow](README.md#peak-extraction-and-rf-dependence) for
units, matching limitations, and replicate handling.

## Artifacts

Outputs remain under `fer_output/` unless overridden. Simulation names contain
a configuration identity and a unique run ID, so repeated runs do not replace
one another. Axes, units, effective config, model/schema/calibration versions,
and exact LUT linkage are stored with the legacy `I`, `z`, `V`, `A_rf` datasets.
Legacy range summaries are derived from axes when metadata is absent. See
[LUT management](LUT_MANAGEMENT.md) for cache identity and safe cleanup.