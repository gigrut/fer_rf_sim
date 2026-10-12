"""Matched batched-current benchmark; all artifacts live in a temporary directory."""

import argparse
from contextlib import redirect_stdout
import cProfile
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import pstats
from tempfile import TemporaryDirectory
import time
import tracemalloc

import numpy as np

from fer_rf_sim.repository import load_simulation
from fer_rf_sim.service import SimulationConfig, SimulationRunner


def measure(config):
    profiler = cProfile.Profile()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        with redirect_stdout(io.StringIO()):
            artifact = profiler.runcall(SimulationRunner().run, config)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    stats = pstats.Stats(profiler).stats
    build_time = sum(values[3] for key, values in stats.items() if key[2] == "build_lut")
    return load_simulation(artifact), {"seconds": elapsed, "peak_traced_bytes": peak,
                                       "lut_build_seconds": build_time}


def benchmark_transmission_vs_interpolation(config=None, repeats=3):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be positive")
    config = config or SimulationConfig(n_E=32, n_V=12, n_Z=4, n_A=3, n_cheb=8,
                                        v_min=.1, v_max=2., z_min=.3, z_max=1., A_max=.2,
                                        phi_samp=4., threads=1)
    with TemporaryDirectory(prefix="fer_benchmark_") as directory:
        config = replace(config, out=directory, threads=1, force_rebuild_lut=False)
        direct, direct_stats = measure(replace(config, use_lut=False))
        cold, cold_stats = measure(replace(config, use_lut=True))
        cache = Path(directory) / cold.metadata["lut_file"]
        cache_time = cache.stat().st_mtime_ns
        warm_stats = []
        for _ in range(repeats):
            warm, measurement = measure(replace(config, use_lut=True))
            warm_stats.append(measurement)
            np.testing.assert_array_equal(warm.current, cold.current)
        for actual, expected in ((cold.z, direct.z), (cold.voltage, direct.voltage),
                                 (cold.rf_amplitude, direct.rf_amplitude)):
            np.testing.assert_array_equal(actual, expected)
        difference = np.abs(cold.current - direct.current)
        nonzero = direct.current != 0
        relative = difference[nonzero] / np.abs(direct.current[nonzero])
        return {"config": {**asdict(config), "out": "<temporary>"}, "direct": direct_stats,
                "lut_cold": cold_stats, "lut_warm": warm_stats,
                "cache_reused": cache.stat().st_mtime_ns == cache_time,
                "max_absolute_error": float(difference.max()),
                "max_relative_error_nonzero": float(relative.max()) if relative.size else None,
                "zero_reference_cells": int(np.count_nonzero(~nonzero)),
                "measurement_scope": "Single-process service wall time with cProfile/tracemalloc, including calibration and HDF5; traced memory is not total RSS"}


def benchmark_speedups(config=None, repeats=3):
    """Time numerical kernels at one height, including DC-table construction."""
    from fer_rf_sim.lut_build import interp_lut
    from fer_rf_sim.physics import transmission
    from fer_rf_sim.quadrature import cosine_nodes
    from fer_rf_sim.simulation import (direct_rf_current_grid, lut_rf_current_grid,
                                       dc_table_rf_current_grid)

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be positive")
    config = config or SimulationConfig(n_E=48, n_V=40, n_Z=2, n_A=10, n_cheb=24,
                                        z_min=0.3, z_max=1.1, A_max=3, phi_samp=4, threads=1)
    config.validate()
    energy = np.linspace(config.E_min, config.E_max + config.e_extra, config.n_E)
    voltage = np.linspace(config.v_min, config.v_max, config.n_V)
    amplitudes = np.linspace(config.A_min, config.A_max, config.n_A)
    nodes = cosine_nodes(config.n_cheb)
    height = (config.z_min + config.z_max) / 2
    heights = np.array([config.z_min, height, config.z_max])
    spacing = (config.v_max - config.v_min) / (config.n_V - 1)
    lut_voltage = np.linspace(config.v_min - config.A_max, config.v_max + config.A_max,
                              config.n_V + int(np.ceil(2 * config.A_max / spacing)))
    started = time.perf_counter()
    spectra = transmission(energy[None, None, :], lut_voltage[None, :, None],
                           heights[:, None, None], config.phi_tip, config.phi_samp)
    build_seconds = time.perf_counter() - started
    interpolator = interp_lut(spectra, energy, lut_voltage, heights)

    def measure_kernel(function):
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = function()
            durations.append(time.perf_counter() - started)
        return result, {"seconds": float(np.median(durations)), "samples_seconds": durations}

    direct, direct_stats = measure_kernel(lambda: direct_rf_current_grid(
        energy, voltage, amplitudes, height, config.phi_tip, config.phi_samp,
        1, nodes, config.amplitude_chunk,
    ))
    general, general_stats = measure_kernel(lambda: lut_rf_current_grid(
        lambda points: interpolator(points), energy, voltage, amplitudes, height,
        nodes, config.amplitude_chunk,
    ))
    aligned, aligned_stats = measure_kernel(lambda: lut_rf_current_grid(
        interpolator, energy, voltage, amplitudes, height, nodes, config.amplitude_chunk,
    ))
    np.testing.assert_allclose(aligned, general, rtol=1e-12, atol=1e-15)

    def evaluate(energy, voltage):
        return transmission(energy, voltage, height, config.phi_tip, config.phi_samp)

    (table, diagnostics), table_stats = measure_kernel(lambda: dc_table_rf_current_grid(
        evaluate, energy, voltage, amplitudes, nodes,
        initial_points=config.dc_table_points, max_points=config.dc_table_max_points,
        rtol=config.dc_table_rtol, atol=config.dc_table_atol, amplitude_chunk=config.amplitude_chunk,
    ))
    significant = np.abs(direct) > np.max(np.abs(direct)) * 1e-6
    error = np.abs(table - direct)
    direct_gradient = np.gradient(direct, voltage, axis=0)
    gradient_error = np.max(np.abs(np.gradient(table, voltage, axis=0) - direct_gradient))
    gradient_peak = np.max(np.abs(direct_gradient))
    return {"grid": {"n_E": config.n_E, "n_V": config.n_V, "n_A": config.n_A,
                     "n_cheb": config.n_cheb, "z_nm": height},
            "direct": direct_stats, "lut_general": general_stats, "lut_aligned": aligned_stats,
            "lut_build_seconds_three_heights": build_seconds,
            "aligned_lut_speedup": general_stats["seconds"] / aligned_stats["seconds"],
            "aligned_lut_max_absolute_difference": float(np.max(np.abs(aligned - general))),
            "dc_table": {**table_stats, "diagnostics": diagnostics,
                         "speedup_vs_direct": direct_stats["seconds"] / table_stats["seconds"],
                         "max_absolute_error": float(error.max()),
                         "max_relative_error_above_1e-6_peak": float(np.max(error[significant] / np.abs(direct[significant]))) if significant.any() else None,
                         "max_gradient_error_fraction_of_peak": float(gradient_error / gradient_peak) if gradient_peak else None},
            "measurement_scope": "Single-height numerical kernels, median wall time; DC-table build and checks included; imports, transmission-LUT build, calibration, workers and HDF5 excluded"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--speedups", action="store_true", help="Measure aligned LUT and checked DC-current-table kernels")
    args = parser.parse_args(argv)
    benchmark = benchmark_speedups if args.speedups else benchmark_transmission_vs_interpolation
    print(json.dumps(benchmark(repeats=args.repeats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())