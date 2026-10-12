# LUT Management

Direct calculation is the built-in default. `--use-lut` enables a transmission
lookup table with axes `(z, V, E)`. Bias coverage includes the full instantaneous
RF voltage range. Energy/voltage/height upscaling creates matching data and axes;
`--upscale-z`, `--upscale-v`, and `--upscale-e` override the common `--upscale`.

## Identity And Reuse

New caches are named `transmission_lut_<sha256>.h5`. The identity covers exact
axis values, unrounded work functions, tip and sample Fermi energies,
transition/blend settings, dtype and model/schema version.
The simulator, management utility and plot loaders share the compatibility
implementation. Matching counts or rounded work functions alone are insufficient.

Earlier caches also hashed a WKB calibration factor that the active model never
used. Removing it changed the identity, so older caches are not reused and are
rebuilt on first use.

Both simulations and caches are written through temporary files followed by
atomic publication. A simulation records the exact cache filename and identity.
Repeated identical simulation runs get different run IDs while reusing an
unchanged compatible cache. `--force-rebuild-lut` explicitly rebuilds the matching
cache; it does not modify old, differently identified scientific outputs.

## Commands

```powershell
fer-sim lut list
fer-sim lut list --brief
fer-sim lut --fer-output another-directory list
fer-sim simulate --list-luts
fer-sim lut check --reference-lut path/to/reference-lut.h5
fer-sim lut cleanup --keep 3
```

The compatibility command asks whether the output directory contains the full
identity of a versioned reference LUT. Unknown legacy provenance cannot establish
compatibility. It returns a nonzero status when no compatible cache exists.

Cleanup is **dry-run by default**. Review its candidate list before adding the
explicit `--delete` confirmation. It does not follow simulation references when
choosing candidates, so retain caches needed by archived simulation artifacts.
Negative keep counts are rejected. No cleanup command is run during tests against
the repository's scientific output directory.

## Legacy Files

Legacy LUTs and simulation datasets remain readable. Plotting a legacy LUT warns
that model and calibration provenance are unknown; missing provenance is never
treated as proof of compatibility. Simulation summaries recover missing ranges
from stored axes. New artifact-linked plotting verifies the cache identity;
direct artifacts have no linked LUT and require an explicit LUT selection for
transmission plotting.

Start convergence studies with temporary, small grids. Refine energy, voltage,
height and RF quadrature independently and compare against direct current.
Interpolation error is model/grid dependent; cache availability alone does not
guarantee speed or scientific accuracy. See the
[benchmark guide](TRANSMISSION_PLOTTING_EFFICIENCY.md).