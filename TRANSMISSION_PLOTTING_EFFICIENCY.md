# Transmission Plotting Efficiency Improvements

## Overview

The `plot_transmission.py` file has been optimized to address efficiency issues with the massive T_array generation. The original approach was creating a 50-million point array (100×100×50×100) which was slow and memory-intensive.

## Key Improvements

### 1. **Direct LUT Plotting** (`plot_transmission_direct_from_lut`)
- **What it does**: Plots transmission directly from the LUT without building the full T_array
- **Memory usage**: Only ~10-50 MB (vs 0.6 GB for full T_array)
- **Speed**: Much faster for single plots
- **Use case**: Perfect for exploration and single plots

### 2. **Optimized T_array Building** (`build_T_array_from_lut`)
- **Multiple parallelization strategies**: Parallelize over A or Z slices based on grid size
- **A-parallelization**: Better load balancing for smaller grids (50 tasks vs 100)
- **Z-parallelization**: Better for large grids with chunked processing
- **Chunked processing**: Processes data in smaller chunks to reduce memory usage
- **Better compression**: Uses higher compression (level 6) for HDF5 files
- **Memory estimation**: Shows memory requirements before building
- **Adaptive processing**: Automatically chooses best strategy based on grid dimensions

### 3. **Smart Default Behavior**
- **No T_array exists**: Uses direct plotting (fast)
- **T_array exists**: Uses pre-computed data (fast for multiple plots)
- **Force direct**: `--direct` flag forces direct plotting even if T_array exists

## Memory Usage Analysis

Current grid (100×100×50×100):
- **T_array size**: 0.37 GB
- **RF averaging per slice**: 0.12 GB  
- **Total estimated**: 0.61 GB
- **Direct plotting**: ~0.01 GB

## Usage Examples

### Quick Plot (Recommended for exploration)
```bash
python plot_transmission.py
# Automatically uses direct plotting if no T_array exists
```

### Force Direct Plotting
```bash
python plot_transmission.py --direct
# Always uses direct plotting, even if T_array exists
```

### Build T_array (for repeated analysis)

**From IDE (Recommended):**
1. Edit `fer_sim_config.py` to set `parallel_strategy = 'A'` (or 'Z', 'auto')
2. In `plot_transmission.py`, uncomment the line: `# build_tarray_from_config()`
3. Run with IDE play button

**From command line:**
```bash
python plot_transmission.py --build
# Builds and saves T_array for future fast plotting

# Choose parallelization strategy:
python plot_transmission.py --build --parallel-strategy A
# Force parallelization over A slices (better for small grids)

python plot_transmission.py --build --parallel-strategy Z  
# Force parallelization over Z slices (better for large grids)
```

### Check Memory Requirements
```bash
python plot_transmission.py --memory-info
# Shows memory usage for current grid
```

### Compare Methods
```bash
python plot_transmission.py --compare
# Shows pros/cons of each method
```

## When to Use Each Method

### Direct LUT Plotting
- ✅ Single plots or exploration
- ✅ Limited memory available
- ✅ Quick visualization needs
- ❌ Multiple plots with same data
- ❌ Repeated analysis

### T_array Building
- ✅ Multiple plots with same data
- ✅ Repeated analysis
- ✅ Batch processing
- ❌ Limited memory
- ❌ Single quick plots

## Performance Comparison

| Method | Memory | First Plot | Subsequent Plots | Use Case |
|--------|--------|------------|------------------|----------|
| Direct | ~10MB | Fast | Slow | Exploration |
| T_array | ~600MB | Slow | Very Fast | Analysis |

## Parallelization Strategies

### A-Parallelization (Recommended for current grid)
- **Tasks**: 50 (one per A value)
- **Load balancing**: Excellent (all tasks similar size)
- **Memory per task**: ~12MB per A slice
- **Best for**: Grids with n_A ≤ 100 and total points ≤ 50M

### Z-Parallelization (Original method)
- **Tasks**: 100 (one per Z value)  
- **Load balancing**: Good (all tasks similar size)
- **Memory per task**: ~6MB per Z slice
- **Best for**: Large grids (>50M points) or when n_A > 100

## Configuration

The efficiency is controlled by parameters in `fer_sim_config.py`:
- `n_Z`, `n_V`, `n_A`, `n_E`: Grid dimensions
- `n_cheb`: Number of Chebyshev nodes for RF averaging
- `parallel_strategy`: Parallelization strategy ('auto', 'A', 'Z')

### Parallelization Strategy Options:
- **'auto'**: Automatically choose best strategy based on grid size
- **'A'**: Force parallelization over A slices (50 tasks, better for small grids)
- **'Z'**: Force parallelization over Z slices (100 tasks, better for large grids)

Reducing grid dimensions will significantly improve performance and reduce memory usage.

## Future Improvements

1. **Lazy loading**: Load T_array slices on demand
2. **Caching**: Cache frequently used slices
3. **Progressive loading**: Load data progressively for large grids
4. **GPU acceleration**: Use GPU for RF averaging calculations 