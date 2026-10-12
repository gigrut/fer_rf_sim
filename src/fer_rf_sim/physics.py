import numpy as np
from scipy.special import airy as sp_airy
from scipy.special import airye as sp_airye
E_F = 5.5
a = 5.12
F_SWITCH = 0.2
F_BLEND = 0.05


def _cardano_two_smallest(a, b, c, d, tol=1e-9):
    """
    Solve a x^3 + b x^2 + c x + d = 0 for broadcast-compatible coefficients.
    Returns two real roots per equation: the two smallest (by value) of all real solutions.
    """
    coefficient_arrays = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
    )
    coefficient_rows = np.stack([values.ravel() for values in coefficient_arrays], axis=1)
    result = np.full((coefficient_rows.shape[0], 2), np.nan, dtype=np.float64)
    root_tolerance = max(tol, np.sqrt(np.finfo(np.float64).eps))

    for index, coefficients in enumerate(coefficient_rows):
        scale = np.max(np.abs(coefficients))
        if not np.isfinite(scale) or scale == 0:
            continue

        nonzero = np.flatnonzero(np.abs(coefficients) > tol * scale)
        roots = np.roots(coefficients[nonzero[0]:])
        real_roots = np.sort(roots.real[np.abs(roots.imag) <= root_tolerance])
        count = min(2, real_roots.size)
        result[index, :count] = real_roots[:count]

    return result


def image_potential(D, V, phi_t, rel_perm=1.0):
    """
    Vectorized Simmons image‐charge correction using Cardano’s formula.
    D : float
    V : array_like of any shape
    phi_t : float
    rel_perm : barrier relative permittivity
    """
    V = np.asarray(V, np.float64)
    orig_shape = V.shape
    Vf = V.ravel()

    # build cubic coefficients for each Vf
    a = Vf / D
    b = phi_t + Vf
    c = phi_t * D
    d = -4.255 * D / rel_perm

    # solve for two smallest real roots per Vf
    roots2 = _cardano_two_smallest(a, b, c, d)   # shape (n,2)
    s1 = roots2[:,0]
    s2 = roots2[:,1]

    # avoid division by zero
    ds = s2 - s1
    ds[ds==0] = np.finfo(np.float64).eps

    # Simmons’s log‐argument
    log_arg = (s2 * (D - s1)) / (s1 * (D - s2))
    L = 3.7 / D

    phi_im_flat = -(
        Vf * (s1 + s2) / (2*D)
        - (1.15 * L * D / ds) * np.log(np.abs(log_arg))
    )

    return phi_im_flat.reshape(orig_shape)


def airy_all(z, z_switch=8.0):
    """
    Evaluate Ai(z), Ai'(z), Bi(z), Bi'(z) with:
      - Exact scipy.special.airy for |z| <= z_switch
      - Two-term asymptotic expansions for |z| > z_switch,
        but with exp(t) clamped to avoid overflow.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty((4,) + z.shape, dtype=np.float64)

    # region 1: exact
    mask_mid = np.abs(z) <= z_switch
    if np.any(mask_mid):
        Ai_m, Aip_m, Bi_m, Bip_m = sp_airy(z[mask_mid])
        out[0][mask_mid] = Ai_m
        out[1][mask_mid] = Aip_m
        out[2][mask_mid] = Bi_m
        out[3][mask_mid] = Bip_m

    # region 2: large positive z
    mask_pos = z > z_switch
    if np.any(mask_pos):
        zp = z[mask_pos]
        t  = (2.0/3.0) * zp**1.5

        exp_neg = np.exp(-t)  # this never overflows

        preA   = 1.0/(2.0*np.sqrt(np.pi)*zp**0.25)
        preAp  = -zp**0.25/(2.0*np.sqrt(np.pi))
        preB   = 1.0/(np.sqrt(np.pi)*zp**0.25)
        preBp  = zp**0.25/(np.sqrt(np.pi))

        corrA  = 1.0 - 5.0/(72.0*t)
        corrAp = 1.0 - 7.0/(72.0*t)
        corrB  = 1.0 + 5.0/(72.0*t)
        corrBp = 1.0 + 7.0/(72.0*t)

        out[0][mask_pos] = preA  * exp_neg * corrA
        out[1][mask_pos] = preAp * exp_neg * corrAp
        log_cap = np.log(np.finfo(np.float64).max) - 1.0
        out[2][mask_pos] = np.exp(np.minimum(t + np.log(preB * corrB), log_cap))
        out[3][mask_pos] = np.exp(np.minimum(t + np.log(preBp * corrBp), log_cap))

    # region 3: large negative z
    mask_neg = z < -z_switch
    if np.any(mask_neg):
        zn   = -z[mask_neg]
        xi   = (2.0/3.0)*zn**1.5 + np.pi/4.0
        amp  = 1.0/(np.sqrt(np.pi)*zn**0.25)
        corr = 1.0 + 5.0/(72.0*xi)
        corrp= 1.0 + 7.0/(72.0*xi)
        out[0][mask_neg] =  amp * np.sin(xi) * corr
        out[2][mask_neg] =  amp * np.cos(xi) / corr
        out[1][mask_neg] = -amp * (zn**0.5 * np.cos(xi) * corrp)
        out[3][mask_neg] =  amp * (zn**0.5 * np.sin(xi) / corrp)

    return out


def _scaled_airy(eta):
    """Return Ai, Ai', Bi, Bi' with Ai·exp(s), Bi·exp(-s) scaling, plus s = (2/3)·max(eta, 0)^(3/2)."""
    eta = np.asarray(eta, dtype=np.float64)
    positive = eta > 0
    values = np.empty((4,) + eta.shape, dtype=np.float64)
    if np.any(positive):
        values[:, positive] = sp_airye(eta[positive])
    if np.any(~positive):
        values[:, ~positive] = sp_airy(eta[~positive])
    return (*values, np.where(positive, (2.0 / 3.0) * np.maximum(eta, 0) ** 1.5, 0.0))


def _forbidden_barrier_transmission(E, collector_energy, d, W0, Ws, field):
    """Match scaled Airy waves, with a midpoint rectangular limit at zero tilt."""
    root0 = np.sqrt(W0)
    roots = np.sqrt(Ws)
    action = (2 * a * d / 3) * (W0 + root0 * roots + Ws) / (root0 + roots)
    attenuation = np.exp(-2 * action)
    k1 = a * np.sqrt(np.maximum(E, 1e-3))
    k3 = a * np.sqrt(np.maximum(collector_energy, 1e-12))
    result = np.empty_like(E)
    rectangular = np.abs(W0 - Ws) <= 1e-6 * np.maximum(W0, Ws)

    if np.any(rectangular):
        kappa = a * np.sqrt((W0[rectangular] + Ws[rectangular]) / 2)
        decay = attenuation[rectangular]
        cosh_scaled = (1 + decay) / 2
        sinh_scaled = -np.expm1(-2 * action[rectangular]) / 2
        left = k1[rectangular]
        right = k3[rectangular]
        denominator = (
            kappa**2 * (left + right)**2 * cosh_scaled**2
            + (kappa**2 - left * right)**2 * sinh_scaled**2
        )
        result[rectangular] = 4 * left * right * kappa**2 * decay / denominator

    tilted = ~rectangular
    if np.any(tilted):
        magnitude = np.abs(field[tilted])
        direction = np.sign(field[tilted])
        factor = (a / magnitude)**(2 / 3)
        Ai0, Aip0, Bi0, Bip0 = sp_airye(factor * W0[tilted])
        Ais, Aips, Bis, Bips = sp_airye(factor * Ws[tilted])
        forward = np.exp(-action[tilted] * (1 + direction))
        backward = np.exp(-action[tilted] * (1 - direction))
        t1 = Aip0 * Bips * forward - Aips * Bip0 * backward
        t2 = Ai0 * Bis * forward - Ais * Bi0 * backward
        t3 = Ais * Bip0 * backward - Aip0 * Bis * forward
        t4 = Ai0 * Bips * forward - Aips * Bi0 * backward
        derivative_scale = direction * (a**2 * magnitude)**(1 / 3)
        left = k1[tilted]
        right = k3[tilted]
        denominator = (
            ((derivative_scale / left) * t1 + (right / derivative_scale) * t2)**2
            + ((right / left) * t3 + t4)**2
        )
        result[tilted] = (right / left) * (4 / np.pi**2) * attenuation[tilted] / denominator

    return result


def transmission(E, V, d, phi_t, phi_s,
                 F1=F_SWITCH, z_switch=8.0, wkb_prefactor=3.69, F_blend=F_BLEND,
                 *, approximation="hybrid", fermi_tip=E_F, fermi_samp=None):
    """
    Transmission probability through a trapezoidal vacuum barrier.

    E is measured from the emitting electrode's band bottom: the tip for V >= 0 and
    the sample for V < 0. Fermi levels align at zero bias, so the barrier depends on
    the emitter's Fermi energy and the collector's band bottom sits at
    E_F(emitter) - |V| - E_F(collector). fermi_samp defaults to fermi_tip.

    Works for any broadcast-compatible mix of scalars/arrays in E, V, d.
    The default uses scaled Airy matching below the barrier, not a calibrated
    WKB/Airy switch. F1 and F_blend remain accepted for caller compatibility;
    wkb_prefactor applies only to the explicit WKB approximation.
    """

    # 1) broadcast all inputs to common shape
    E, V, d = np.broadcast_arrays(
        np.asarray(E, dtype=np.float64),
        np.asarray(V, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
    )
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(V)) and np.all(np.isfinite(d))):
        raise ValueError("E, V, and d must contain only finite values")
    if np.any(d <= 0):
        raise ValueError("Barrier thickness d must be positive")

    # 2) relabel electrodes so V_eff ≥ 0
    swap   = V < 0
    V_eff  = np.abs(V)
    phi1   = np.where(swap, phi_s, phi_t)
    phi2   = np.where(swap, phi_t, phi_s)
    fermi_samp = fermi_tip if fermi_samp is None else fermi_samp
    fermi_emitter   = np.where(swap, fermi_samp, fermi_tip)
    fermi_collector = np.where(swap, fermi_tip, fermi_samp)
    # (optional image-charge term would be added here)

    # 3) barrier tops and electric field
    W0         = phi1 + fermi_emitter - E
    Ws_minus_V = phi2 + fermi_emitter - E - V_eff
    collector_energy = E + V_eff - fermi_emitter + fermi_collector
    F_s        = (V_eff + (phi1 - phi2)) / d

    T_wkb = np.zeros_like(E, dtype=np.float64)
    T_air = np.zeros_like(E, dtype=np.float64)
    abs_F = np.abs(F_s)
    wkb_valid = (W0 > 0) & (Ws_minus_V > 0)

    if approximation not in ("hybrid", "wkb", "airy"):
        raise ValueError("approximation must be hybrid, wkb, or airy")
    m_wkb = wkb_valid if approximation == "wkb" else np.zeros_like(wkb_valid)
    if np.any(m_wkb):
        W0w   = W0[m_wkb]
        Wsw   = Ws_minus_V[m_wkb]
        root0 = np.sqrt(W0w)
        roots = np.sqrt(Wsw)
        logT = -(4 * a * d[m_wkb] / 3) * (
            (W0w + root0 * roots + Wsw) / (root0 + roots)
        )
        T_wkb[m_wkb] = wkb_prefactor * np.exp(logT)

    if approximation != "wkb" and np.any(wkb_valid):
        T_air[wkb_valid] = _forbidden_barrier_transmission(
            E[wkb_valid], collector_energy[wkb_valid], d[wkb_valid],
            W0[wkb_valid], Ws_minus_V[wkb_valid], F_s[wkb_valid],
        )

    m_air = (~wkb_valid) & (abs_F > 0) & (approximation != "wkb")
    if np.any(m_air):
        F_air = F_s[m_air]
        factor = (a / np.abs(F_air))**(2/3)
        z0  = factor * W0[m_air]
        zs  = factor * Ws_minus_V[m_air]
        zp  = np.sign(F_air) * (a**2 * np.abs(F_air))**(1/3)

        # Exact scaled Airy values; truncated asymptotics past |eta| > 8 made ln T(V) jagged.
        Ai0, Aip0, Bi0, Bip0, s0 = _scaled_airy(z0)
        Ais, Aips, Bis, Bips, ss = _scaled_airy(zs)
        delta = s0 - ss
        forward = np.exp(-delta - np.abs(delta))
        backward = np.exp(delta - np.abs(delta))

        E_a = E[m_air]
        k1  = a * np.sqrt(np.maximum(E_a, 1e-3))
        k3  = a * np.sqrt(np.maximum(collector_energy[m_air], 1e-12))
        num = (k3 / k1) * (4 / np.pi**2)

        t1 = Aip0*Bips*forward - Aips*Bip0*backward
        t2 = Ai0*Bis*forward   - Ais*Bi0*backward
        t3 = Ais*Bip0*backward - Aip0*Bis*forward
        t4 = Ai0*Bips*forward  - Aips*Bi0*backward

        denom = ((zp/k1)*t1 + (k3/zp)*t2)**2 + ((k3/k1)*t3 + t4)**2
        denom = np.where(np.abs(denom) < 1e-12, np.inf, denom)

        T_air[m_air] = num * np.exp(-2 * np.abs(delta)) / denom

    clipped = np.clip(T_wkb if approximation == "wkb" else T_air, 0.0, 1.0)
    return np.where(collector_energy > 0, clipped, 0.0)  # no propagating collector states


def transmission_wkb(E, V, d, phi_t, phi_s, *, wkb_prefactor=1.0, z_switch=8.0,
                     fermi_tip=E_F, fermi_samp=None):
    """Evaluate WKB explicitly, independent of hybrid transition masks."""
    return transmission(E, V, d, phi_t, phi_s, wkb_prefactor=wkb_prefactor, z_switch=z_switch,
                        approximation="wkb", fermi_tip=fermi_tip, fermi_samp=fermi_samp)


def transmission_airy(E, V, d, phi_t, phi_s, *, z_switch=8.0, fermi_tip=E_F, fermi_samp=None):
    """Evaluate Airy matching, including the analytic zero-field limit."""
    return transmission(E, V, d, phi_t, phi_s, wkb_prefactor=1.0, z_switch=z_switch,
                        approximation="airy", fermi_tip=fermi_tip, fermi_samp=fermi_samp)


def fit_wkb_prefactor(phi_t, phi_s, F1,
                      D_min, D_max, n_D=20,
                      n_E=200,    E_min=0.01,
                      z_switch=8.0, fermi_tip=E_F, fermi_samp=None):
    """
    Fit the WKB prefactor so that
      prefactor * T_WKB(E)  ≈  T_Airy(E)
    at F=F1, over n_D sample distances. A standalone WKB-versus-Airy comparison
    tool; the simulation does not use or record it.

    Returns
    -------
    f_avg : float
      The average prefactor across all valid samples.
    f_i   : ndarray, shape (M,)
      The per-distance fitted prefactors.
    """
    if not 0 < D_min < D_max:
        raise ValueError("Calibration distances must satisfy 0 < D_min < D_max")

    # 1) build sample voltages and corresponding Ds
    Ds = np.linspace(D_min, D_max, n_D)
    Vs = Ds * F1 - phi_t + phi_s

    # only keep physical points
    if Vs.size == 0:
        raise ValueError("No positive D values; check your (V_min,V_max,phi_t,phi_s,F1).")

    M = Vs.size

    # 2) build energy grid
    fermi_samp = fermi_tip if fermi_samp is None else fermi_samp
    Es = np.linspace(E_min, min(fermi_tip, fermi_samp) - 1e-8, n_E)

    # 3) broadcast into M × n_E arrays
    V_grid = Vs[:, None]          # shape (M,1)
    E_grid = Es[None, :]          # shape (1,N)
    D_grid = Ds[:, None]          # shape (M,1)

    # now shape-broadcast to (M,N)
    V_grid = np.broadcast_to(V_grid, (M, n_E))
    E_grid = np.broadcast_to(E_grid, (M, n_E))
    D_grid = np.broadcast_to(D_grid, (M, n_E))

    # 4) call your transmission() twice
    #    a) WKB‐only  (prefactor=1, cutoff=F1)
    T_wkb = transmission_wkb(E_grid, V_grid, D_grid, phi_t, phi_s, z_switch=z_switch,
                             fermi_tip=fermi_tip, fermi_samp=fermi_samp)

    #    b) Airy‐only (prefactor=1, cutoff=0 → forces Airy everywhere)
    T_airy = transmission_airy(E_grid, V_grid, D_grid, phi_t, phi_s, z_switch=z_switch,
                               fermi_tip=fermi_tip, fermi_samp=fermi_samp)

    # 5) build mask for energies where WKB would have applied
    fermi_emitter = np.where(V_grid < 0, fermi_samp, fermi_tip)
    W0 = np.where(V_grid < 0, phi_s, phi_t) + fermi_emitter - E_grid
    Ws = np.where(V_grid < 0, phi_t, phi_s) + fermi_emitter - E_grid - np.abs(V_grid)
    valid = (W0 > 0) & (Ws > 0)

    # 6) compute per-sample numerator and denominator
    #    numerator_i = ∑_j T_airy[i,j] * T_wkb[i,j]  over valid j
    #    denom_i     = ∑_j T_wkb[i,j]**2            over valid j
    Tw = T_wkb  * valid
    Ta = T_airy * valid

    numer = np.sum(Ta * Tw, axis=1)   # shape (M,)
    denom = np.sum(Tw * Tw, axis=1)  # shape (M,)

    # avoid divide‐by‐zero
    good = denom > 0
    f_i  = np.zeros_like(numer)
    f_i[good] = numer[good] / denom[good]

    samples = f_i[good & np.isfinite(f_i) & (f_i > 0)]
    if samples.size == 0:
        raise ValueError("WKB prefactor fit produced no valid samples")
    f_avg = float(np.mean(samples))

    return f_avg, samples
