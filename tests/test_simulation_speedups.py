import numpy as np
import pytest

from fer_rf_sim.lut_build import interp_lut
from fer_rf_sim.quadrature import cosine_nodes
from fer_rf_sim.simulation import lut_rf_current_grid


@pytest.mark.parametrize("height,aligned_energy", [(0.3, True), (0.4, True), (0.3, False)])
def test_lut_lookup_matches_general_interpolation(height, aligned_energy):
    energies = np.linspace(0.01, 5.5, 17)
    voltages = np.linspace(-1, 3, 21)
    heights = np.array([0.3, 0.5])
    random = np.random.default_rng(42)
    interpolator = interp_lut(random.random((2, 21, 17)), energies, voltages, heights)
    requested = energies[::2] if aligned_energy else np.linspace(0.02, 5.4, 11)
    bias = np.array([-0.5, 0, 0.5, 2.5])
    amplitudes = np.array([0, 0.2, 0.5])
    nodes = cosine_nodes(7)
    expected = lut_rf_current_grid(lambda points: interpolator(points), requested,
                                   bias, amplitudes, height, nodes, amplitude_chunk=1)
    actual = lut_rf_current_grid(interpolator, requested, bias, amplitudes,
                                height, nodes, amplitude_chunk=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_aligned_lut_does_not_call_general_interpolator(monkeypatch):
    from scipy.interpolate import RegularGridInterpolator

    energy = np.linspace(0.01, 5.5, 8)
    interpolator = interp_lut(np.ones((2, 3, 8)), energy, [-1, 0, 1], [0.3, 0.5])

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("Aligned lookup should not call the 3-D interpolator")

    monkeypatch.setattr(RegularGridInterpolator, "__call__", unexpected_lookup)
    result = lut_rf_current_grid(interpolator, energy, [-0.5, 0.5], [0, 0.2],
                                 0.3, cosine_nodes(5))
    assert result.shape == (2, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        lut_rf_current_grid(interpolator, energy, [-2], [0], 0.3, cosine_nodes(5))


@pytest.mark.parametrize("energy", [np.linspace(0.01, 8, 24), np.linspace(6, 8, 5)])
def test_direct_energy_filter_and_zero_rf_match_scalar_reference(monkeypatch, energy):
    from fer_rf_sim.physics import E_F, transmission
    from fer_rf_sim.quadrature import rf_averaged_current
    from fer_rf_sim.simulation import direct_rf_current_grid

    voltages = np.array([-0.2, 0, 0.3])
    amplitudes = np.array([0, 0.1])
    nodes = cosine_nodes(5)
    evaluations = []

    def counted_transmission(energy, voltage, *args, **kwargs):
        assert np.all(energy <= E_F)
        evaluations.append(np.broadcast_shapes(energy.shape, voltage.shape))
        return transmission(energy, voltage, *args, **kwargs)

    monkeypatch.setattr("fer_rf_sim.simulation.transmission", counted_transmission)
    actual = direct_rf_current_grid(energy, voltages, amplitudes, 0.3,
                                    4, 5, nodes, amplitude_chunk=1)
    expected = np.array([
        [rf_averaged_current(lambda bias: transmission(energy, bias, 0.3, 4, 5),
                             energy, voltage, amplitude, nodes)
         for amplitude in amplitudes] for voltage in voltages
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    if np.any(energy <= E_F):
        assert evaluations[0] == (voltages.size, np.count_nonzero(energy <= E_F))
        assert len(evaluations) == 2
    else:
        assert not evaluations


def test_dc_table_refines_against_independent_rf_reference():
    from fer_rf_sim.quadrature import rf_averaged_current
    from fer_rf_sim.simulation import dc_table_rf_current_grid

    energy = np.linspace(0.01, 5.5, 24)
    voltage = np.linspace(-0.4, 1.5, 51)
    amplitudes = np.array([0, 0.1, 0.3])
    nodes = cosine_nodes(17)
    queried_voltages = []

    def evaluate(energy, voltage):
        queried_voltages.extend(np.asarray(voltage).ravel().tolist())
        return np.exp(-energy) * (1 + 0.4 * np.exp(-((voltage - 0.7) / 0.08)**2))

    actual, diagnostics = dc_table_rf_current_grid(
        evaluate, energy, voltage, amplitudes, nodes,
        initial_points=9, max_points=5000, rtol=1e-5, atol=1e-12,
    )
    table_queries = queried_voltages[:-voltage.size]
    assert len(table_queries) == 9 + diagnostics["validation_points"]
    assert np.unique(table_queries).size == len(table_queries)
    expected = np.array([
        [rf_averaged_current(lambda bias: evaluate(energy, bias), energy, bias, amplitude, nodes)
         for amplitude in amplitudes] for bias in voltage
    ])
    assert diagnostics["points"] > 9
    assert diagnostics["max_scaled_error"] <= 1
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-10)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.gradient(actual, voltage, axis=0),
                               np.gradient(expected, voltage, axis=0), rtol=0, atol=0.002)
    from scipy.signal import find_peaks

    expected_peaks, _ = find_peaks(np.gradient(expected[:, 2], voltage), prominence=0.1)
    actual_peaks, _ = find_peaks(np.gradient(actual[:, 2], voltage), prominence=0.1)
    assert expected_peaks.size > 0
    np.testing.assert_array_equal(actual_peaks, expected_peaks)


def test_dc_table_failure_does_not_write_current_artifact(tmp_path):
    from fer_rf_sim.service import SimulationConfig, SimulationRunner

    config = SimulationConfig(n_E=8, n_V=3, n_Z=2, n_A=2, n_cheb=3, threads=1,
                              v_min=0.1, v_max=2, A_max=0.2, z_max=0.5,
                              dc_table=True, dc_table_points=2, dc_table_max_points=2,
                              dc_table_rtol=1e-12, dc_table_atol=1e-30, out=str(tmp_path))
    with pytest.raises(RuntimeError, match="did not converge"):
        SimulationRunner().run(config)
    assert not list(tmp_path.glob("current*.h5"))


@pytest.mark.parametrize("use_lut,threads", [(False, 1), (True, 1), (False, 2), (True, 2)])
def test_dc_table_service_metadata_and_current(tmp_path, use_lut, threads):
    from dataclasses import replace
    import json
    from fer_rf_sim.repository import load_simulation
    from fer_rf_sim.service import SimulationConfig, SimulationRunner

    config = SimulationConfig(n_E=16, n_V=5, n_Z=2, n_A=2, n_cheb=7, threads=threads,
                              v_min=0.1, v_max=0.4, A_max=0.05, z_max=0.5,
                              use_lut=use_lut, phi_samp=4, out=str(tmp_path))
    runner = SimulationRunner()
    reference = load_simulation(runner.run(config))
    accelerated = load_simulation(runner.run(replace(config, dc_table=True, dc_table_points=17)))
    diagnostics = json.loads(accelerated.metadata["dc_table_diagnostics_json"])
    assert len(diagnostics) == config.n_Z
    assert all(item["max_scaled_error"] <= 1 for item in diagnostics)
    assert accelerated.metadata["dc_table"]
    assert accelerated.metadata["config_identity"] != reference.metadata["config_identity"]
    np.testing.assert_allclose(accelerated.current, reference.current, rtol=0.002, atol=1e-15)


def test_fermi_energy_reaches_simulation_and_artifact(tmp_path):
    from dataclasses import replace
    from fer_rf_sim.repository import load_simulation
    from fer_rf_sim.service import SimulationConfig, SimulationRunner

    config = SimulationConfig(n_E=16, n_V=5, n_Z=2, n_A=2, n_cheb=7, threads=1,
                              v_min=0.1, v_max=0.4, A_max=0.05, z_max=0.5,
                              phi_samp=4, out=str(tmp_path))
    runner = SimulationRunner()
    default = load_simulation(runner.run(config))
    lowered = load_simulation(runner.run(replace(config, E_max=4.5, E_F_tip=4.5, E_F_samp=4.0)))
    assert (default.metadata["E_F_tip"], default.metadata["E_F_samp"]) == (5.5, 5.5)
    assert (lowered.metadata["E_F_tip"], lowered.metadata["E_F_samp"]) == (4.5, 4.0)
    assert not np.allclose(lowered.current, default.current, rtol=0.01)
    swapped = load_simulation(runner.run(replace(config, E_max=4.5, E_F_tip=4.0, E_F_samp=4.5)))
    assert not np.allclose(swapped.current, lowered.current, rtol=0.01)


def test_service_reports_z_progress_and_cancels_before_current_artifact(tmp_path):
    from threading import Event
    from fer_rf_sim.service import SimulationConfig, SimulationRunner

    config = SimulationConfig(n_E=8, n_V=3, n_Z=3, n_A=2, n_cheb=3, threads=1,
                              v_min=0.1, v_max=0.4, A_max=0.05, z_max=0.5,
                              phi_samp=4, out=str(tmp_path))
    cancel_event = Event()
    reports = []

    def report_progress(stage, completed, total):
        reports.append((stage, completed, total))
        if stage == "Current grid" and completed == 1:
            cancel_event.set()

    with pytest.raises(InterruptedError, match="cancelled by user"):
        SimulationRunner().run(
            config, progress_callback=report_progress, cancel_event=cancel_event,
        )

    assert reports[0] == ("Current grid", 1, config.n_Z)
    assert not list(tmp_path.glob("current*.h5"))


@pytest.mark.parametrize("values", [
    {"amplitude_chunk": 0}, {"dc_table": "true"}, {"dc_table_points": 1},
    {"dc_table_max_points": 2}, {"dc_table_rtol": -1}, {"dc_table_atol": float("nan")},
    {"dc_table_rtol": 0, "dc_table_atol": 0},
])
def test_speedup_config_validation(values):
    from fer_rf_sim.config import SimulationConfig

    with pytest.raises(ValueError):
        SimulationConfig(**values).validate()


def test_dc_table_flags_and_yaml_precedence(tmp_path):
    from fer_rf_sim.config import parse_config

    path = tmp_path / "settings.yaml"
    path.write_text("simulation:\n  dc_table: true\n  amplitude_chunk: 2\n")
    config = parse_config(["--config", str(path), "--no-dc-table", "--amplitude-chunk", "4"])
    assert not config.dc_table
    assert config.amplitude_chunk == 4
    enabled = parse_config(["--config", str(path), "--dc-table", "--dc-table-rtol", "0.0001"])
    assert enabled.dc_table
    assert enabled.dc_table_rtol == 0.0001


def test_dc_table_matches_physics_across_bias_and_contact_potential():
    from fer_rf_sim.physics import transmission
    from fer_rf_sim.simulation import dc_table_rf_current_grid, direct_rf_current_grid

    energy = np.linspace(0.01, 5.5, 24)
    voltage = np.linspace(-0.3, 10, 45)
    amplitudes = np.array([0, 0.2])
    nodes = cosine_nodes(11)

    def evaluate(energy, voltage):
        return transmission(energy, voltage, 0.7, 4, 5)

    reference = direct_rf_current_grid(energy, voltage, amplitudes, 0.7, 4, 5, nodes)
    actual, diagnostics = dc_table_rf_current_grid(
        evaluate, energy, voltage, amplitudes, nodes, rtol=1e-4, max_points=12000,
    )
    assert diagnostics["max_scaled_error"] <= 1
    np.testing.assert_allclose(actual, reference, rtol=0.001, atol=1e-14)


def test_default_dc_table_tolerances_hold_where_current_is_tiny():
    from fer_rf_sim.config import SimulationConfig
    from fer_rf_sim.physics import transmission
    from fer_rf_sim.simulation import dc_table_rf_current_grid, direct_rf_current_grid

    # At z=2.8 nm the current spans ~1e-27 to 1e-9, mostly below any fixed absolute tolerance.
    defaults = SimulationConfig()
    energy = np.linspace(0.01, 5.5, 60)
    voltage = np.linspace(0.5, 10, 200)
    amplitudes = np.array([0.0, 1.0])
    nodes = cosine_nodes(16)

    def evaluate(energy, voltage):
        return transmission(energy, voltage, 2.8, 4, 5)

    reference = direct_rf_current_grid(energy, voltage, amplitudes, 2.8, 4, 5, nodes)
    actual, _ = dc_table_rf_current_grid(
        evaluate, energy, voltage, amplitudes, nodes,
        rtol=defaults.dc_table_rtol, atol=defaults.dc_table_atol, max_points=defaults.dc_table_max_points,
    )
    np.testing.assert_allclose(actual, reference, rtol=0.005, atol=0)


def test_single_voltage_lut_keeps_general_fallback():
    energy = np.linspace(0.01, 5.5, 8)
    interpolator = interp_lut(np.ones((2, 1, 8)), energy, [0.5], [0.3, 0.5])
    result = lut_rf_current_grid(interpolator, energy, [0.5], [0], 0.3, cosine_nodes(3))
    expected = lut_rf_current_grid(lambda points: interpolator(points), energy, [0.5],
                                   [0], 0.3, cosine_nodes(3))
    np.testing.assert_array_equal(result, expected)