# LUT Management Best Practices

## Overview

The Look-Up Table (LUT) for transmission calculations is computationally expensive to generate but only needs to be computed once for a given set of parameters. This document explains how to manage LUT files efficiently.

## LUT File Naming

LUT files are now automatically named based on their parameters:
```
transmission_lut_phi{phi_tip:.1f}_{phi_samp:.1f}_nE{n_E}_nV{n_V}_nZ{n_Z}.h5
```

For example: `transmission_lut_phi4.0_4.0_nE100_nV150_nZ100.h5`

This ensures that:
- Different parameter sets get different LUT files
- Files are never overwritten accidentally
- You can easily identify which parameters each LUT corresponds to

## Automatic LUT Reuse

The simulation automatically:
1. Checks if a compatible LUT exists for your parameters
2. Loads the existing LUT if found (saving hours of computation)
3. Only rebuilds the LUT if parameters have changed

## Command Line Options

### List Existing LUTs
```bash
python FER_constant_current_simulation.py --list-luts
```

### Force LUT Rebuild
```bash
python FER_constant_current_simulation.py --force-rebuild-lut [other options...]
```

## LUT Management Utility

Use the `manage_luts.py` script for advanced LUT management:

### List All LUTs
```bash
python manage_luts.py list
python manage_luts.py list --brief  # Show only file names and sizes
```

### Check Compatibility
```bash
python manage_luts.py check --phi-tip 4.0 --phi-samp 4.0 --n-E 100 --n-V 150 --n-Z 100
```

### Clean Up Old LUTs
```bash
python manage_luts.py cleanup --keep 3  # Keep only 3 most recent LUTs
```

## When LUTs Are Rebuilt

A LUT will be rebuilt if:
- No compatible LUT exists
- Work functions (phi_tip, phi_samp) have changed
- Grid dimensions (n_E, n_V, n_Z) have changed
- Grid ranges have changed
- The `--force-rebuild-lut` flag is used

## Storage Considerations

- LUT files can be large (100MB+ for high-resolution grids)
- Consider using `manage_luts.py cleanup` periodically to free disk space
- LUT files are stored in the `fer_output/` directory
- You can safely move/copy LUT files between machines with the same parameters

## Best Practices

1. **Start with coarse grids** for initial testing, then increase resolution
2. **Use the same work functions** across related simulations to reuse LUTs
3. **Check existing LUTs** before running new simulations
4. **Back up important LUTs** if they took a long time to compute
5. **Use `--force-rebuild-lut` sparingly** - only when you need to regenerate

## Example Workflow

```bash
# 1. Check what LUTs you already have
python manage_luts.py list

# 2. Run a quick test with coarse grids
python FER_constant_current_simulation.py --n-E 50 --n-V 50 --n-Z 50

# 3. Run production simulation with fine grids (will reuse LUT if compatible)
python FER_constant_current_simulation.py --n-E 200 --n-V 200 --n-Z 200

# 4. Clean up old LUTs periodically
python manage_luts.py cleanup --keep 5
```

## Troubleshooting

**Problem**: LUT is being rebuilt unnecessarily
**Solution**: Check that all parameters match exactly, including work functions

**Problem**: LUT files taking too much disk space
**Solution**: Use `manage_luts.py cleanup` or delete old LUT files manually

**Problem**: LUT compatibility check fails
**Solution**: Use `--force-rebuild-lut` to regenerate the LUT 