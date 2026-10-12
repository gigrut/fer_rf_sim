import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import airy

from fer_rf_sim import physics


def _unscaled_airy_reference(energy, voltage, distance, phi_tip, phi_sample):
    field = (voltage + phi_tip - phi_sample) / distance
    factor = (physics.a / np.abs(field))**(2 / 3)
    derivative_scale = np.sign(field) * (physics.a**2 * np.abs(field))**(1 / 3)
    Ai0, Aip0, Bi0, Bip0 = airy(factor * (phi_tip + physics.E_F - energy))
    Ais, Aips, Bis, Bips = airy(factor * (phi_sample + physics.E_F - energy - voltage))
    left = physics.a * np.sqrt(np.maximum(energy, 1e-3))
    right = physics.a * np.sqrt(np.maximum(energy + voltage, 1e-12))
    t1 = Aip0 * Bips - Aips * Bip0
    t2 = Ai0 * Bis - Ais * Bi0
    t3 = Ais * Bip0 - Aip0 * Bis
    t4 = Ai0 * Bips - Aips * Bi0
    denominator = (
        ((derivative_scale / left) * t1 + (right / derivative_scale) * t2)**2
        + ((right / left) * t3 + t4)**2
    )
    return (right / left) * (4 / np.pi**2) / denominator


def test_explicit_wkb_matches_barrier_integral():
    energy, voltage, distance = 1.0, 0.2, 1.0
    integral = quad(lambda position: np.sqrt(4 + 5.5 - energy - voltage * position / distance),
                    0, distance)[0]
    expected = np.exp(-2 * 5.12 * integral)
    actual = physics.transmission_wkb(energy, voltage, distance, 4, 4)
    assert actual > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    airy = physics.transmission_airy(energy, voltage, distance, 4, 4)
    assert abs(airy / expected - 1) > 0.1


def test_wkb_prefactor_fit_preserves_unit_ratios(monkeypatch):
    def unit(energy, voltage, distance, *args, **kwargs):
        return np.ones_like(energy)

    monkeypatch.setattr(physics, "transmission_wkb", unit)
    monkeypatch.setattr(physics, "transmission_airy", unit)
    average, samples = physics.fit_wkb_prefactor(4, 4, 0.2, 0.3, 1, n_D=3, n_E=5)
    assert average == 1
    np.testing.assert_array_equal(samples, np.ones(3))


def test_explicit_branches_preserve_electrode_swap():
    for evaluator in (physics.transmission_wkb, physics.transmission_airy):
        actual = evaluator(np.array([1., 2.]), -0.5, 0.8, 4., 5.)
        expected = evaluator(np.array([1., 2.]), 0.5, 0.8, 5., 4.)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_airy_values_against_scipy_reference():
    values = np.array([-5., 0., 2., 7.])
    np.testing.assert_allclose(physics.airy_all(values), np.asarray(airy(values)), rtol=1e-12, atol=0)


def test_open_barrier_airy_branch_matches_unscaled_reference_past_asymptotic_switch():
    from scipy.special import airy as unscaled_airy

    # Ws < 0 selects the unscaled-formula branch; z=4 drives eta past 8.
    energy, voltage, height, phi_t, phi_s = 5.4, np.linspace(6, 9, 301), 4.0, 4., 5.
    field = (voltage + phi_t - phi_s) / height
    factor = (physics.a / field) ** (2 / 3)
    z0 = factor * (phi_t + physics.E_F - energy)
    zs = factor * (phi_s + physics.E_F - energy - voltage)
    assert z0.max() > 8 and zs.max() < 0
    zp = (physics.a**2 * field) ** (1 / 3)
    Ai0, Aip0, Bi0, Bip0 = unscaled_airy(z0)
    Ais, Aips, Bis, Bips = unscaled_airy(zs)
    k1 = physics.a * np.sqrt(energy)
    k3 = physics.a * np.sqrt(energy + voltage)
    denom = ((zp / k1) * (Aip0 * Bips - Aips * Bip0) + (k3 / zp) * (Ai0 * Bis - Ais * Bi0)) ** 2 \
        + ((k3 / k1) * (Ais * Bip0 - Aip0 * Bis) + (Ai0 * Bips - Aips * Bi0)) ** 2
    expected = (k3 / k1) * (4 / np.pi**2) / denom
    np.testing.assert_allclose(physics.transmission(energy, voltage, height, phi_t, phi_s), expected, rtol=1e-9)


def test_equal_fermi_energies_match_the_default_and_shift_the_barrier_like_work_functions():
    energy = np.linspace(0.5, 3.0, 6)
    voltage, height, shift = 0.8, 0.7, 0.4
    shifted = physics.transmission(energy, voltage, height, 4. + shift, 5. + shift)
    raised = physics.transmission(energy, voltage, height, 4., 5.,
                                  fermi_tip=physics.E_F + shift, fermi_samp=physics.E_F + shift)
    np.testing.assert_allclose(raised, shifted, rtol=1e-12, atol=0)
    default = physics.transmission(energy, voltage, height, 4., 5.)
    explicit = physics.transmission(energy, voltage, height, 4., 5.,
                                    fermi_tip=physics.E_F, fermi_samp=physics.E_F)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("voltage", [0.8, -0.8])
def test_emitter_fermi_sets_barrier_and_collector_fermi_sets_final_states(voltage):
    energy = np.linspace(0.5, 3.0, 6)
    height, phi_t, phi_s = 0.7, 4., 5.
    tip_emits = voltage > 0

    def transmit(emitter, collector):
        fermi = dict(fermi_tip=emitter, fermi_samp=collector) if tip_emits else \
            dict(fermi_tip=collector, fermi_samp=emitter)
        return physics.transmission(energy, voltage, height, phi_t, phi_s, **fermi)

    # Moving only the collector's Fermi energy leaves the barrier alone and changes only k3.
    assert not np.allclose(transmit(5.5, 5.0), transmit(5.5, 7.0), rtol=1e-3, atol=0)
    # A higher emitter Fermi energy raises the barrier above a fixed E, so transmission falls.
    assert np.all(transmit(6.5, 5.5) < transmit(5.5, 5.5))


def test_transmission_vanishes_without_propagating_collector_states():
    # Collector band bottom lies above the electron energy: E + |V| - E_F(emitter) + E_F(collector) <= 0.
    energy = np.array([0.5, 1.0])
    result = physics.transmission(energy, 0.2, 0.7, 4., 5., fermi_tip=5.5, fermi_samp=4.0)
    np.testing.assert_array_equal(result, 0.0)


def test_current_integral_window_follows_the_emitting_electrodes_fermi_energy():
    from fer_rf_sim.quadrature import current, current_batch

    energy = np.linspace(0.01, 5.5, 50)
    transmission = np.exp(-energy)
    tip, samp = 3.0, 4.5
    for voltage, fermi in ((2.0, tip), (-2.0, samp)):
        inside = energy <= fermi
        expected = np.sign(voltage) * np.trapezoid(
            np.minimum(abs(voltage), fermi - energy[inside]) * transmission[inside], energy[inside],
        )
        batch = current_batch(transmission[None, :], energy, np.array([voltage]), tip, samp)
        np.testing.assert_allclose(batch, [expected], rtol=1e-12)
        np.testing.assert_allclose(current(transmission, energy, voltage, tip, samp), expected, rtol=1e-12)
    equal = current_batch(transmission[None, :], energy, np.array([2.0]), tip, tip)
    np.testing.assert_allclose(equal, current_batch(transmission[None, :], energy, np.array([2.0]), tip))


def test_zero_rf_current_has_no_contact_potential_peak():
    from fer_rf_sim.quadrature import cosine_nodes
    from fer_rf_sim.simulation import direct_rf_current_grid

    energy = np.linspace(0.01, physics.E_F, 1000)
    voltage = np.linspace(0.9, 1.1, 201)
    current = direct_rf_current_grid(
        energy, voltage, np.array([0.0]), 0.3, 4.0, 5.0,
        cosine_nodes(1),
    )[:, 0]

    assert np.all(np.diff(current) > 0)


def test_low_field_transmission_matches_exact_airy():
    energy = np.linspace(0.01, physics.E_F, 100)
    voltage = np.array([0.92, 0.94, 0.95, 1.05, 1.06, 1.08])[:, None]
    expected = _unscaled_airy_reference(energy, voltage, 0.3, 4.0, 5.0)
    actual = physics.transmission(energy, voltage, 0.3, 4.0, 5.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("distance", [0.3, 1.0, 5.0])
def test_contact_potential_matches_rectangular_barrier(distance):
    energy = np.linspace(0.01, physics.E_F, 100)
    barrier = 4.0 + physics.E_F - energy
    left = physics.a * np.sqrt(energy)
    right = physics.a * np.sqrt(energy + 1.0)
    kappa = physics.a * np.sqrt(barrier)
    expected = 4 * left * right / (
        (left + right)**2 * np.cosh(kappa * distance)**2
        + ((kappa**2 - left * right) / kappa)**2 * np.sinh(kappa * distance)**2
    )
    voltage = np.array([1.0 - 1e-9, 1.0, 1.0 + 1e-9])[:, None]
    actual = physics.transmission(energy, voltage, distance, 4.0, 5.0)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=1e-7, atol=0)


def test_default_transmission_ignores_the_wkb_prefactor():
    voltage = np.linspace(0.9, 1.1, 21)
    actual = physics.transmission(physics.E_F, voltage, 0.3, 4.0, 5.0, wkb_prefactor=1.0)
    calibrated = physics.transmission(physics.E_F, voltage, 0.3, 4.0, 5.0, wkb_prefactor=10.0)

    np.testing.assert_array_equal(actual, calibrated)