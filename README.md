# FER RF Simulation

Field-emission resonance tunneling with RF modulation, constant-current
post-processing, and optional plotting and desktop interfaces.

## Install

Python 3.10 or newer is required. Dependencies are declared in
[pyproject.toml](pyproject.toml); NumPy 2 or newer supplies `numpy.trapezoid`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[plot,analysis,gui,dev]"
```

For headless simulation, `python -m pip install -e .` is sufficient and does not
install Qt, pandas, Matplotlib, or Plotly. Extras: `plot` for plots, `analysis`
for measured CSV data and plots, `gui` for PySide6, and `dev` for tests/build/lint.
The requirements file is a compatibility shortcut for the full editable install.

## Commands

Commands work outside the repository after installation. `python -m fer_rf_sim`
is equivalent to `fer-sim`.

```powershell
fer-sim simulate --n_E 16 --n_V 4 --n_Z 2 --n_A 2 --n_cheb 4 --threads 1 --out fer_output
fer-sim workflow --quick-test --threads 1
fer-sim check --h5-path path/to/result.h5
fer-sim constant-current --h5-path path/to/result.h5 --multiple 1e-10 1e-9
fer-sim transmission --lut-path path/to/lut.h5 --x V --y A --z 0.3 --e 1.0
fer-sim lut list
fer-gui
```

Direct calculation is the built-in default; `--use-lut` selects interpolation.
Current targets above are examples in **normalized units**, not amperes.
There is no implemented comparison command or recovery/migration command.

## Configuration And Units

Precedence is built-ins < YAML < explicit CLI or GUI edits. The optional
`fer_config.yaml` is discovered in the working directory; `--config PATH` selects
another file. Relative YAML output paths are relative to that file; explicit CLI
paths are relative to the working directory. Explicit default-valued options and
false flags override YAML. Old `Z_min`, `V_min`, and underscore CLI aliases remain
accepted; unknown settings and invalid grids are errors.

Energy and work functions: eV. Height: nm. Bias: V. RF amplitude: V peak.
Power conversion requires impedance (50 ohms by default); RMS and peak-to-peak
conversions are explicit. Current uses the existing normalized integral with
prefactor 1. Image-potential research helpers remain available but disabled in
the active model. Refactoring and regression tests are not scientific certification.

## Low-Field Transmission

The active model uses scaled Airy barrier matching and an analytic rectangular
limit at zero electric field. For tip/sample work functions of 4/5 eV, this limit
is at 1 V bias, not at zero bias. The previous calibrated WKB/Airy transition
could introduce a false peak or dip near this voltage even with zero RF.
Explicit WKB evaluation remains available for comparison; its prefactor
(`wkb_prefactor`) is no longer applied to the active transmission calculation,
and simulations no longer compute or store it.

The numerical model version is `trapezoidal-scaled-airy-5`, so older cached LUTs
are not reused for new runs. Existing HDF5 results remain unchanged and readable;
rerun the simulation to obtain corrected current and derived spectra.

## Layout And Compatibility

| Location | Responsibility |
| --- | --- |
| [src/fer_rf_sim](src/fer_rf_sim) | Config, physics, quadrature, simulation, repositories, analysis, plotting, CLI and optional GUI |
| [tests](tests) | Synthetic HDF5/CSV, numerical, CLI, packaging and Qt regressions |
| `fer_output/` | Default simulation artifacts and LUT cache; unchanged location |
| Root Python adapters | Legacy script/import compatibility during deprecation |
| Root research recipes | Dataset-specific fitting; shared CSV and broadening helpers |
| [archive](archive/README.md) | Verified, unchanged historical source snapshots |

New simulations have unique run IDs, full effective configuration and exact LUT
linkage. Legacy datasets remain readable; missing provenance is marked unknown.
Existing scientific outputs are not migrated or overwritten by cleanup.

## Measured Data

Raw CSVs are separate from fitted-peak tables. Set `FER_REAL_DATA_CSV` or
`real_data.raw_csv_path` for the GUI's raw file/folder. The legacy `csv_path` key
remains an alias. Set `FER_FITTED_PEAKS_CSV` or `real_data.fitted_peaks_csv_path`
for the fitted-peak recipe. `real_data.peak_number` is 1-based and respected.
`FER_CONFIG` selects the real-data configuration file for research/GUI imports.

RF metadata precedence is explicit `real_data.rf_mapping` (filename to V peak),
then actual CSV metadata, then unknown. Filename order never assigns RF.
Unknown RF can be viewed, but RF-dependent analysis requires metadata. Fitted
baselines require `RFEnabled` or a complete `real_data.baseline_files` mapping
(filename to boolean, true means RF-off). The peak-index recipe requires
`FER_PEAK_INDEX_MAPPING`, a YAML filename map containing `drift_reference`,
`drift_offset_nm`, `z_shift_nm`, and optionally `rf_peak_v`. External CSVs are not
included; dataset-specific fit results remain unverified.

## Verification

```powershell
python -m pytest -q
python -m ruff check src/fer_rf_sim tests
python -m build
```

Qt tests skip when the optional dependency is absent. CI checks Linux/Windows
and a separate Windows GUI job. See [workflow](README_WORKFLOW.md),
[LUT management](LUT_MANAGEMENT.md), and
[plotting and performance](TRANSMISSION_PLOTTING_EFFICIENCY.md).
