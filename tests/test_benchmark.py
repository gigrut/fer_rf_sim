import benchmark_lut_vs_direct
import profile_simulation
from fer_rf_sim.config import SimulationConfig


def test_benchmark_measures_current_paths_and_cache():
    config = SimulationConfig(n_E=8, n_V=3, n_Z=2, n_A=2, n_cheb=3,
                              v_min=.01, v_max=.04, A_max=.003, z_min=.3, z_max=.5,
                              phi_samp=4, threads=1)
    report = benchmark_lut_vs_direct.benchmark_transmission_vs_interpolation(config, repeats=1)
    assert report["cache_reused"]
    assert report["lut_cold"]["lut_build_seconds"] > 0
    assert report["lut_warm"][0]["lut_build_seconds"] == 0
    assert report["direct"]["peak_traced_bytes"] > 0
    assert report["max_relative_error_nonzero"] >= 0


def test_speedup_benchmark_compares_matching_numerical_kernels():
    config = SimulationConfig(n_E=8, n_V=3, n_Z=2, n_A=2, n_cheb=3,
                              v_min=0.01, v_max=0.04, A_max=0.003, z_min=0.3, z_max=0.5,
                              phi_samp=4, threads=1, dc_table_points=9)
    report = benchmark_lut_vs_direct.benchmark_speedups(config, repeats=1)
    assert report["direct"]["seconds"] > 0
    assert report["aligned_lut_max_absolute_difference"] < 1e-15
    assert report["dc_table"]["diagnostics"]["max_scaled_error"] <= 1
    assert report["dc_table"]["max_relative_error_above_1e-6_peak"] < 0.002


def test_profiler_uses_unique_outputs(tmp_path):
    arguments = ["--n_E", "8", "--n_V", "3", "--n_Z", "2", "--n_A", "1",
                 "--n_cheb", "2", "--threads", "1", "--out", str(tmp_path),
                 "--profile-dir", str(tmp_path)]
    assert profile_simulation.main(arguments) == 0
    assert profile_simulation.main(arguments) == 0
    assert len(list(tmp_path.glob("simulation_*.prof"))) == 2