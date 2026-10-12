import numpy as np
from .physics import E_F


def chebyshev_phases(count):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Quadrature count must be a positive integer")
    return np.pi * (2 * np.arange(1, count + 1) - 1) / (2 * count)


def cosine_nodes(count):
    return np.cos(chebyshev_phases(count))


def CGQ(function, dc_voltage, rf_amplitude, n=100):
    """RF average for scalar or array DC voltages in the same units as amplitude."""
    amplitude = np.asarray(rf_amplitude, dtype=float)
    if amplitude.size != 1 or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError("RF amplitude must be a finite non-negative scalar")
    voltage = np.asarray(dc_voltage, dtype=float)
    if not np.isfinite(voltage).all():
        raise ValueError("DC voltage must be finite")
    arguments = voltage[..., None] + float(amplitude.item()) * cosine_nodes(n)
    return np.mean(function(arguments), axis=-1)


def D_avg(interp, Eg, Vdc, Arf, z, cos_nodes):
    """RF‑averaged D(E) using precomputed LUT interpolator."""
    V_inst = Vdc + Arf * cos_nodes                  # (N,)
    nE, nN = Eg.size, V_inst.size
    # repeat grids so shapes match
    E_rep = np.repeat(Eg, nN)                      # (nE*nN,)
    V_rep = np.tile(V_inst, nE)                    # (nE*nN,)
    z_rep = np.full(E_rep.shape, z)

    pts = np.stack([z_rep, V_rep, E_rep], axis=1)  # (nE*nN, 3)
    vals = interp(pts).reshape(nE, nN)             # (nE, N)
    return vals.mean(axis=1)


def current(D_E, Eg, Vdc, fermi_tip=E_F, fermi_samp=None):
    D_E = np.asarray(D_E, dtype=np.float64)
    Eg = np.asarray(Eg, dtype=np.float64)
    if D_E.ndim != 1 or Eg.ndim != 1 or D_E.shape != Eg.shape:
        raise ValueError("D_E and Eg must be one-dimensional arrays with matching shapes")
    if not (np.all(np.isfinite(D_E)) and np.all(np.isfinite(Eg)) and np.isfinite(Vdc)):
        raise ValueError("D_E, Eg, and Vdc must contain only finite values")
    if np.any(np.diff(Eg) <= 0):
        raise ValueError("Eg must be strictly increasing")

    bias_sign = np.sign(Vdc)
    bias = abs(Vdc)
    # The emitting electrode sets the occupied-energy limit: tip for V >= 0, sample for V < 0.
    emitter_fermi = fermi_tip if fermi_samp is None or Vdc >= 0 else fermi_samp
    domain = (Eg >= 0.0) & (Eg <= emitter_fermi)
    energy = Eg[domain]
    transmission_values = D_E[domain]
    occupation_weight = np.minimum(bias, emitter_fermi - energy)
    return bias_sign * np.trapezoid(
        occupation_weight * transmission_values, energy,
    )


def current_batch(D_E, Eg, Vdc, fermi_tip=E_F, fermi_samp=None):
    """Evaluate ``current`` for a batch of transmission spectra.

    ``D_E`` has shape ``(..., n_E)`` and ``Vdc`` has the matching leading
    shape. The interval masks reproduce the point-selection behavior of
    ``current`` without constructing an interpolator or Python loop per phase.
    """
    D_E = np.asarray(D_E, dtype=np.float64)
    Eg = np.asarray(Eg, dtype=np.float64)
    Vdc = np.asarray(Vdc, dtype=np.float64)
    if D_E.ndim < 2 or Eg.ndim != 1 or D_E.shape[-1] != Eg.size:
        raise ValueError("D_E must end with an energy axis matching Eg")
    if Vdc.shape != D_E.shape[:-1]:
        raise ValueError("Vdc must match the leading D_E dimensions")
    if not (np.all(np.isfinite(D_E)) and np.all(np.isfinite(Eg)) and np.all(np.isfinite(Vdc))):
        raise ValueError("D_E, Eg, and Vdc must contain only finite values")
    if np.any(np.diff(Eg) <= 0):
        raise ValueError("Eg must be strictly increasing")

    domain = Eg >= 0.0
    if not np.all(domain):
        D_E = D_E[..., domain]
        Eg = Eg[domain]

    fermi_samp = fermi_tip if fermi_samp is None else fermi_samp
    # The emitting electrode sets the occupied-energy limit: tip for V >= 0, sample for V < 0.
    emitter_fermi = np.where(Vdc < 0, fermi_samp, fermi_tip)[..., None]
    bias = np.abs(Vdc)
    occupation_weight = np.minimum(bias[..., None], emitter_fermi - Eg)
    weighted = occupation_weight * D_E
    intervals = np.diff(Eg) * (weighted[..., 1:] + weighted[..., :-1]) / 2.0
    integral = np.where(Eg[1:] <= emitter_fermi, intervals, 0.0).sum(axis=-1)
    return np.sign(Vdc) * integral


def rf_averaged_current(transmission_at_voltage, Eg, Vdc, Arf, cos_nodes,
                        fermi_tip=E_F, fermi_samp=None):
    """Average the full instantaneous current over RF phase nodes.

    ``transmission_at_voltage`` must return transmission over ``Eg`` for one
    instantaneous voltage. Averaging after current integration preserves the
    voltage-dependent occupation window for each RF phase.
    """
    cos_nodes = np.asarray(cos_nodes, dtype=np.float64)
    if cos_nodes.ndim != 1 or cos_nodes.size == 0:
        raise ValueError("cos_nodes must be a non-empty one-dimensional array")
    if not (np.isfinite(Vdc) and np.isfinite(Arf)):
        raise ValueError("Vdc and Arf must be finite")

    phase_currents = [
        current(transmission_at_voltage(Vdc + Arf * cosine), Eg,
                Vdc + Arf * cosine, fermi_tip, fermi_samp)
        for cosine in cos_nodes
    ]
    return float(np.mean(phase_currents))
