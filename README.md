# fer_rf_sim

FER RF STM tunneling simulation and analysis.

## Setup

```bash
cd C:\Users\willh\OneDrive\Documents\GitHub\fer_rf_sim
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Layout

| Path | Purpose |
|------|---------|
| `src/fer_rf_sim/` | Installable package (physics, simulation, postprocess, analysis) |
| `src/fer_rf_sim/config.py` | Edit simulation parameters here |
| `outputs/sim/` | HDF5 simulation outputs (was `fer_output/`) |
| `data/raw/` | STM / peak-fit CSV files |
| `scripts/` | Runnable workflow wrappers |
| `archive/` | Old experiments and legacy plotters |

## Commands

```bash
fer-sim simulate          # constant-current simulation
fer-sim check             # quick validation plots
fer-sim workflow          # sim → check → constant-current analysis
fer-sim compare           # real vs simulated peak shifts
fer-sim lut list          # list transmission LUT files
```

## Data paths

Copy `FittedPeaksDF.csv` into `data/raw/` or set:

```bash
set FER_REAL_DATA_CSV=C:\path\to\FittedPeaksDF.csv
```

## Undo / restore

```powershell
git checkout master -- .     # restore tracked files to last commit
git clean -fd                # remove untracked dirs (use with care)
```

Restore post-processing from git: `python _recover_files.py`

**compare_peak_shifts_real_vs_sim.py** was never in git. If only a tiny shim remains, use OneDrive **Version history** on that file. `fer-sim compare` needs the full script (>10 KB).

## Migration helper

```bash
python _migrate_layout.py
```

Creates `data/`, `outputs/`, `archive/`, and moves experimental scripts to `archive/experimental/`.
