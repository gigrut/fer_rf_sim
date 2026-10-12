import h5py
import numpy as np
import pytest

from fer_rf_sim import plotting as FER_plotter
from fer_rf_sim.transmission_plotting import rf_transmission_grid, transmission_slice, requested_transmission_grid


def synthetic_transmission(points):
    return 100 * points[:, 0] + 10 * points[:, 1] + points[:, 2]


def test_asymmetric_rf_transmission_axes():
    axes = (np.arange(2.), np.arange(3.), np.arange(4.), np.arange(5.))
    nodes = np.array([np.pi / 3, 2 * np.pi / 3])
    result = rf_transmission_grid(synthetic_transmission, *axes, nodes)
    assert result.shape == (2, 3, 4, 5)
    assert result[1, 2, 3, 4] == pytest.approx(124)
    x_values, y_values, selected = transmission_slice(result, axes, "V", "E", {"Z": 1, "A": 2})
    assert selected.shape == (5, 3)
    assert selected[4, 2] == pytest.approx(124)
    with pytest.raises(ValueError, match="two fixed"):
        transmission_slice(result, axes, "V", "A", {"Z": 1})


def test_only_requested_slice_is_computed():
    axes = (np.arange(7.), np.arange(3.), np.arange(4.), np.arange(9.))
    sizes = []

    def record(points):
        sizes.append(len(points))
        return synthetic_transmission(points)

    result = requested_transmission_grid(record, axes, "V", "A", {"Z": 2, "E": 3}, [0, np.pi])
    assert sizes == [3 * 4 * 2]
    assert result[2].shape == (4, 3)


def test_direct_artifact_transmission_slice(tmp_path):
    from fer_rf_sim.repository import load_simulation
    from fer_rf_sim.physics import transmission
    from fer_rf_sim.transmission_plotting import simulation_transmission_source

    path = tmp_path / "direct.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": np.ones((2, 3, 2)), "z": [.5, 1.],
                             "V": [1., 2., 3.], "A_rf": [0., .4], "E": [.1, .2]}.items():
            file[name] = values
        file.attrs.update(phi_tip=4., phi_samp=5.)
    data = load_simulation(path)
    np.testing.assert_array_equal(data.energy, [.1, .2])
    source, description = simulation_transmission_source(data)
    assert "Recomputed" in description
    axes = (data.z, data.voltage, data.rf_amplitude, data.energy)
    nodes = np.array([np.pi / 3, 2 * np.pi / 3])
    for amplitude in data.rf_amplitude:
        _, _, values = requested_transmission_grid(source, axes, "V", "Z", {"A": amplitude, "E": .2}, nodes)
        expected = np.array([
            [np.mean(transmission(.2, bias + amplitude * np.cos(nodes), height, 4., 5.))
             for bias in data.voltage] for height in data.z
        ])
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0)
    with h5py.File(path, "a") as file:
        del file.attrs["phi_tip"]
    with pytest.raises(ValueError, match="metadata"):
        simulation_transmission_source(load_simulation(path))


def test_current_surface_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "asymmetric.h5"
    current = np.arange(6.).reshape(2, 3, 1)
    with h5py.File(path, "w") as file:
        for name, values in {"I": current, "z": [1, 2], "V": [3, 4, 5], "A_rf": [0]}.items():
            file[name] = values
    monkeypatch.setattr(FER_plotter.go.Figure, "show", lambda self: None)
    figure = FER_plotter.surface(path, a_idx=0)
    assert np.shape(figure.data[0].z) == (2, 3)
    np.testing.assert_array_equal(figure.data[0].z, current[:, :, 0])
    assert figure.data[0].x[1, 2] == 5
    assert figure.data[0].y[1, 2] == 2


def test_artifact_transmission_uses_exact_linked_lut(tmp_path):
    from fer_rf_sim.repository import load_simulation
    from fer_rf_sim.transmission_plotting import simulation_transmission_source

    path = tmp_path / "current.h5"
    lut_path = tmp_path / "linked.h5"
    with h5py.File(path, "w") as file:
        for name, values in {"I": np.ones((2, 3, 1)), "z": [.5, 1.],
                             "V": [1., 2., 3.], "A_rf": [0.], "E": [.1, .2]}.items():
            file[name] = values
        file.attrs.update(lut_file=lut_path.name, lut_identity="expected")
    with h5py.File(lut_path, "w") as file:
        for name, values in {"D": np.arange(12.).reshape(2, 3, 2) / 12,
                             "z": [.5, 1.], "V": [1., 2., 3.], "E": [.1, .2]}.items():
            file[name] = values
        file.attrs["lut_identity"] = "expected"
    data = load_simulation(path)
    with pytest.warns(UserWarning, match="Legacy LUT"):
        source, label = simulation_transmission_source(data)
    assert label == "Linked simulation LUT"
    _, _, values = requested_transmission_grid(
        source, (data.z, data.voltage, data.rf_amplitude, data.energy),
        "V", "Z", {"A": 0., "E": .2}, [0.],
    )
    np.testing.assert_allclose(values, np.arange(12.).reshape(2, 3, 2)[:, :, 1] / 12)
    with h5py.File(lut_path, "a") as file:
        file.attrs["lut_identity"] = "different"
    with pytest.warns(UserWarning, match="Legacy LUT"), pytest.raises(ValueError, match="identity differs"):
        simulation_transmission_source(data)
    lut_path.unlink()
    with pytest.raises(FileNotFoundError):
        simulation_transmission_source(data)


def test_current_plot_rejects_transmission():
    with pytest.raises(ValueError, match="not transmission"):
        FER_plotter._log_current(np.ones((2, 3, 4, 5)))
    np.testing.assert_allclose(FER_plotter._log_current(np.full((2, 3, 4), np.e)), 1)


def test_legacy_transmission_views_use_shared_axes(monkeypatch):
    from types import SimpleNamespace
    data = SimpleNamespace(z=np.arange(2.), voltage=np.arange(3.), rf_amplitude=np.arange(4.))
    energy = np.arange(5.)
    monkeypatch.setattr(FER_plotter, "_transmission_source", lambda folder: (data, synthetic_transmission, energy))
    figures = []
    monkeypatch.setattr(FER_plotter.go.Figure, "show", lambda self: figures.append(self))
    FER_plotter.trans_map(x="E", y="A", z_val=1, V_val=2)
    assert figures[-1].data[0].z.shape == (4, 5)
    assert figures[-1].data[0].z[3, 4] == pytest.approx(124)
    FER_plotter.trans_heat_constV_slider(V_val=2)
    assert len(figures[-1].data) == 2
    assert figures[-1].data[1].z.shape == (4, 5)
    assert figures[-1].data[1].z[3, 4] == pytest.approx(124)
    FER_plotter.trans_heat_constE_slider(E_val=4)
    assert figures[-1].data[1].z.shape == (4, 3)
    assert figures[-1].data[1].z[3, 2] == pytest.approx(124)