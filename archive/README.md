# Legacy Source Archive

Archived on 2026-10-11 from revision `c8109ba8960d94a4191e5c64d2f0a9c853682dd2`.
All eight files were compared with their original Git blob hashes after moving.
These are historical source snapshots, not supported entry points. No scientific
HDF5, NPY, CSV, image, or profile outputs were moved or regenerated.

| Original source | Replacement |
| --- | --- |
| `backup_old_versions/FER_constant_current_simulation_fully_vectorized.py` | `fer_rf_sim.simulation` |
| `backup_old_versions/FER_constant_current_V5.py` | `fer_rf_sim.simulation` |
| `backup_old_versions/FER_constant_current_V6.py` | `fer_rf_sim.simulation` |
| `backup_old_versions/FER_sim_parallel_processing_V2.py` | `fer_rf_sim.simulation` and `fer_rf_sim.service` |
| `backup_old_versions/FER_plot_3D.py` | `fer_rf_sim.plotting` and `fer-sim check` |
| `backup_old_versions/FER_plotter_5D.py` | `fer_rf_sim.plotting` and `fer-sim transmission` |
| `debugAndTesting.py` | Focused tests under `tests/` |
| `FER_basic_current_sim.py` | Static root adapter to `fer-sim simulate` |

The old basic-current shim contained no physics implementation. It depended on
missing `simulation/_legacy_loader.py` and `simulation/basic_current.py` files;
there were no source callers. Its original dynamic loader is preserved here only
for provenance. It is not imported by the package.

Specialized fitting recipes remain at the repository root. Shared helpers were
extracted without replacing their fit models. Dataset-specific fits are not
certified by the synthetic regression suite.