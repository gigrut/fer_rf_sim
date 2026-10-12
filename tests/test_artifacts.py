from dataclasses import replace
import json

import h5py
import numpy as np
import pytest

from fer_rf_sim.provenance import atomic_h5, numerical_identity
from fer_rf_sim.lut_repository import lut_matches
from fer_rf_sim.repository import load_simulation
from fer_rf_sim.service import SimulationConfig, SimulationRunner


def tiny_config(tmp_path, **changes):
    return replace(SimulationConfig(n_E=16, n_V=4, n_Z=2, n_A=2, n_cheb=4, threads=1,
                                   v_min=0.01, v_max=0.04, z_min=0.3, z_max=0.5,
                                   A_max=0.003, phi_samp=4, out=str(tmp_path)), **changes)


def test_unique_atomic_artifacts_and_lut_reuse(tmp_path):
    config = tiny_config(tmp_path, use_lut=True)
    runner = SimulationRunner()
    first = runner.run(config)
    cache = next(tmp_path.glob("transmission_lut*.h5"))
    original_time = cache.stat().st_mtime_ns
    second = runner.run(config)
    assert first != second and first.exists() and second.exists()
    assert cache.stat().st_mtime_ns == original_time
    data = load_simulation(first)
    assert data.metadata["lut_file"] == cache.name
    assert json.loads(data.metadata["effective_config_json"])["v_min"] == config.v_min
    with h5py.File(cache) as file:
        spec = json.loads(file.attrs["lut_spec_json"])
        assert lut_matches(file, spec)
        assert file.attrs["lut_identity"] == numerical_identity(spec)
        old_model = {**spec, "model_version": "trapezoidal-hybrid-2"}
        assert not lut_matches(file, old_model)
        assert numerical_identity(old_model) != file.attrs["lut_identity"]
        spec["phi_tip"] *= 1.00001
        assert not lut_matches(file, spec)
    assert not list(tmp_path.glob("*.tmp"))


def test_direct_lut_convergence(tmp_path):
    runner = SimulationRunner()
    config = tiny_config(tmp_path)
    direct = load_simulation(runner.run(config)).current
    coarse = load_simulation(runner.run(replace(config, use_lut=True))).current
    fine = load_simulation(runner.run(replace(config, use_lut=True, upscale_v=8))).current
    coarse_error = np.max(np.abs((coarse - direct) / direct))
    fine_error = np.max(np.abs((fine - direct) / direct))
    assert np.all(direct > 0)
    assert fine_error < coarse_error
    np.testing.assert_allclose(fine, direct, rtol=0.01, atol=0)


def test_failed_atomic_write_preserves_existing_target(tmp_path):
    artifact = tmp_path / "protected.h5"
    artifact.write_bytes(b"existing")
    with pytest.raises(RuntimeError):
        with atomic_h5(artifact) as file:
            file["values"] = [1]
            raise RuntimeError("interrupted")
    assert artifact.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [artifact]


def test_cleanup_defaults_to_dry_run(tmp_path):
    from fer_rf_sim.lut_cli import cleanup_old_luts, check_compatibility
    paths = [tmp_path / f"transmission_lut_{index}.h5" for index in range(3)]
    for path in paths:
        path.touch()
    assert len(cleanup_old_luts(tmp_path, 1)) == 2
    assert all(path.exists() for path in paths)
    assert not check_compatibility(4, 4, 10, 10, 10, tmp_path)
    with pytest.raises(ValueError):
        cleanup_old_luts(tmp_path, -1)


def test_legacy_ranges_come_from_axes(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": np.zeros((2, 3, 1)), "z": [1, 2], "V": [0, 1, 2], "A_rf": [0]}.items():
            file[name] = values
    data = load_simulation(path)
    assert data.metadata["provenance"] == "unknown"
    assert data.metadata["v_max"] == 2
    assert data.metadata["z_min"] == 1


def test_batched_lut_current_matches_scalar_reference():
    from fer_rf_sim.simulation import lut_rf_current_grid
    from fer_rf_sim.quadrature import rf_averaged_current, current, cosine_nodes
    energy = np.linspace(.01, 5.5, 17)
    voltage = np.array([.1, .2, .3])
    amplitudes = np.array([0., .02])
    nodes = cosine_nodes(5)

    def interpolator(points):
        return np.exp(-points[:, 2]) * (1 + points[:, 1]) * 1e-15

    result = lut_rf_current_grid(interpolator, energy, voltage, amplitudes, 1., nodes, amplitude_chunk=1)
    expected = np.empty((3, 2))
    for voltage_index, bias in enumerate(voltage):
        def at_voltage(instantaneous):
            return interpolator(np.column_stack((np.ones_like(energy), np.full_like(energy, instantaneous), energy)))
        for amplitude_index, amplitude in enumerate(amplitudes):
            expected[voltage_index, amplitude_index] = rf_averaged_current(
                at_voltage, energy, bias, amplitude, nodes,
            )
        assert result[voltage_index, 0] == pytest.approx(current(at_voltage(bias), energy, bias), rel=1e-12, abs=0)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_zero_rf_lut_has_no_contact_potential_peak():
    from fer_rf_sim.lut_build import build_lut, interp_lut
    from fer_rf_sim.physics import E_F
    from fer_rf_sim.quadrature import cosine_nodes
    from fer_rf_sim.simulation import direct_rf_current_grid, lut_rf_current_grid

    energy = np.linspace(0.01, E_F, 200)
    voltage = np.linspace(0.9, 1.1, 41)
    heights = np.array([0.3, 0.5])
    amplitudes = np.array([0.0])
    nodes = cosine_nodes(5)
    lut = build_lut(energy, voltage, heights, 4.0, 5.0, n_jobs=1)
    interpolator = interp_lut(lut, energy, voltage, heights)
    direct = direct_rf_current_grid(energy, voltage, amplitudes, 0.3, 4.0, 5.0, nodes)
    interpolated = lut_rf_current_grid(interpolator, energy, voltage, amplitudes, 0.3, nodes)

    assert np.all(np.diff(interpolated[:, 0]) > 0)
    np.testing.assert_allclose(interpolated, direct, rtol=1e-12, atol=0)