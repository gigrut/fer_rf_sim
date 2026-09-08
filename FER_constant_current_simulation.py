import argparse
import os
from pathlib import Path
import time
import yaml

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.special import airy as sp_airy

# To align the two pieces of the data (Airy vs WKB), you may have to tune the fudge factor.

###############################################################################
# constants
###############################################################################
E_F = 5.5  # Fermi energy (eV)
a = 5.12  # sqrt(2m)/hbar with units of 1/(nm*sqrt(eV)) -- differs from Gundlach's by a factor of 2
###############################################################################
# Utilities
###############################################################################

# Need to worry about local argument a having the same name as global parameter a
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


###############################################################################
# transmission: allows different tip/sample work functions
###############################################################################

def transmission(E, V, d, phi_t, phi_s,
                 F1=0.2, z_switch=8.0, fudge=3.69):
    """
    Transmission probability through a trapezoidal vacuum barrier.

    Works for any broadcast-compatible mix of scalars/arrays in E, V, d.
    """

    import numpy as np

    # ---------- small helpers -------------------------------------------------
    def _to_arr(x):          # ndarray wrapper (keeps 0-D scalars cheap)
        return np.asarray(x, dtype=np.float64)

    def _slice_if_arr(x, mask):
        """Return x[mask] if x is array-like, else the scalar x."""
        return x[mask] if isinstance(x, np.ndarray) and x.ndim else x

    def _is_scalar(x):
        return (not isinstance(x, np.ndarray)) or x.ndim == 0
    # -------------------------------------------------------------------------

    # 1) broadcast all inputs to common shape
    E, V, d = np.broadcast_arrays(_to_arr(E), _to_arr(V), _to_arr(d))
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(V)) and np.all(np.isfinite(d))):
        raise ValueError("E, V, and d must contain only finite values")
    if np.any(d <= 0):
        raise ValueError("Barrier thickness d must be positive")

    # 2) relabel electrodes so V_eff ≥ 0
    swap   = V < 0
    V_eff  = np.abs(V)
    phi1   = np.where(swap, phi_s, phi_t)
    phi2   = np.where(swap, phi_t, phi_s)
    # (optional image-charge term would be added here)

    # 3) barrier tops and electric field
    W0         = phi1 + E_F - E
    Ws_minus_V = phi2 + E_F - E - V_eff
    F_s        = (V_eff + phi1 - phi2) / d

    # 4) allocate output
    T = np.zeros_like(E, dtype=np.float64)

    # ---------------------- WKB region  |F| ≤ F1 ------------------------------
    m_wkb = (np.abs(F_s) <= F1) & (W0 > 0) & (Ws_minus_V > 0)
    if np.any(m_wkb):
        F_wkb = F_s[m_wkb]
        W0w   = W0[m_wkb]
        Wsw   = Ws_minus_V[m_wkb]
        delta = W0w**1.5 - Wsw**1.5

        logT = np.empty_like(F_wkb)

        # distance values compatible with mask size
        d_wkb = _slice_if_arr(d, m_wkb)

        # (a) rectangular limit  |F|→0
        mask_rect = np.abs(F_wkb) < 1e-10
        if np.any(mask_rect):
            d_rect = d_wkb if _is_scalar(d_wkb) else d_wkb[mask_rect]
            logT[mask_rect] = -2 * a * d_rect * np.sqrt(W0w[mask_rect])

        # (b) genuine WKB
        mask_wkb = ~mask_rect
        if np.any(mask_wkb):
            logT[mask_wkb] = -(4 * a) / (3 * F_wkb[mask_wkb]) * delta[mask_wkb]

        T[m_wkb] = fudge * np.exp(logT)

    # ------------------------- Airy region  |F| > F1 --------------------------
    m_air = ~m_wkb
    if np.any(m_air):
        F_air = F_s[m_air]
        factor = (a / np.abs(F_air))**(2/3)
        z0  = factor * W0[m_air]
        zs  = factor * Ws_minus_V[m_air]
        zp  = np.sign(F_air) * (a**2 * np.abs(F_air))**(1/3)

        Ai0, Aip0, Bi0, Bip0 = airy_all(z0, z_switch=z_switch)
        Ais, Aips, Bis, Bips = airy_all(zs, z_switch=z_switch)

        E_a = E[m_air]
        V_a = V_eff[m_air]
        k1  = a * np.sqrt(np.maximum(E_a, 1e-3))
        k3  = a * np.sqrt(np.maximum(E_a + V_a, 1e-12))
        num = (k3 / k1) * (4 / np.pi**2)

        t1 = Aip0*Bips - Aips*Bip0
        t2 = Ai0*Bis   - Ais*Bi0
        t3 = Ais*Bip0  - Aip0*Bis
        t4 = Ai0*Bips  - Aips*Bi0

        denom = ((zp/k1)*t1 + (k3/zp)*t2)**2 + ((k3/k1)*t3 + t4)**2
        denom = np.where(np.abs(denom) < 1e-12, np.inf, denom)

        T[m_air] = num / denom

    return np.clip(T, 0.0, 1.0)



def calibrate_fudge(phi_t, phi_s, F1,
                    D_min, D_max, n_D=20,
                    n_E=200,    E_min=0.01,
                    z_switch=8.0):
    """
    Vectorized calibration of the Simmons‐WKB 'fudge' factor so that
      fudge * T_WKB(E)  ≈  T_Airy(E)
    at F=F1, over n_V sample voltages.

    Returns
    -------
    f_avg : float
      The average fudge factor across all valid samples.
    f_i   : ndarray, shape (M,)
      The per-voltage fitted fudge factors.
    """
    print('Calibrating fudge factor')
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
    Es = np.linspace(E_min, E_F - 1e-8, n_E)

    # 3) broadcast into M × n_E arrays
    V_grid = Vs[:, None]          # shape (M,1)
    E_grid = Es[None, :]          # shape (1,N)
    D_grid = Ds[:, None]          # shape (M,1)

    # now shape-broadcast to (M,N)
    V_grid = np.broadcast_to(V_grid, (M, n_E))
    E_grid = np.broadcast_to(E_grid, (M, n_E))
    D_grid = np.broadcast_to(D_grid, (M, n_E))

    # 4) call your transmission() twice
    #    a) WKB‐only  (fudge=1, cutoff=F1)
    T_wkb  = transmission(E_grid, V_grid, D_grid,
                          phi_t, phi_s,
                          F1=F1,
                          z_switch=z_switch,
                          fudge=1.0)

    #    b) Airy‐only (fudge=1, cutoff=0 → forces Airy everywhere)
    T_airy = transmission(E_grid, V_grid, D_grid,
                          phi_t, phi_s,
                          F1=0.0,
                          z_switch=z_switch,
                          fudge=1.0)

    # 5) build mask for energies where WKB would have applied
    W0  = phi_t + E_grid - 0*V_grid  - E_grid + E_F  # compute properly:
    W0  = phi_t + E_F - E_grid
    Ws  = phi_s + E_F - E_grid - np.abs(V_grid)
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

    # 7) average (but ignore elements equal to 1.0)
    x = f_i[good]
    samples = x[x != 1.0]
    if samples.size == 0:
        raise ValueError("Fudge calibration produced no valid samples")
    f_avg = float(np.mean(samples))

    return f_avg, samples

###############################################################################
# LUT with progress
###############################################################################

def _upscaled_axis(values, factor):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("LUT axes must be one-dimensional with at least two points")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("LUT upscale factors must be positive integers")
    return np.linspace(values[0], values[-1], (values.size - 1) * factor + 1)


def build_lut(Eg, Vg, zg, phi_t, phi_s, upscale=1, fudge=1):
    """
    Build a transmission lookup table over (z,V,E), with optional upscaling.

    Parameters
    ----------
    Eg : array-like    Base energy grid
    Vg : array-like    Base DC bias grid
    zg : array-like    Base tip-height grid
    phi_t, phi_s : float
      Tip and sample work functions
    upscale : int or tuple of ints
      Resolution multiplier for (z,V,E) axes. If int, applies to all.

        Returns
        -------
        lut : ndarray, shape (Nz, Nv, Ne)
            Transmission probability on the optionally upscaled axes.
    """
    # parse upscale factors
    if isinstance(upscale, int):
        fz = fV = fE = upscale
    else:
        fz, fV, fE = upscale  # type: ignore

    # build high-res axes
    z_lut = _upscaled_axis(zg, fz)
    V_lut = _upscaled_axis(Vg, fV)
    E_lut = _upscaled_axis(Eg, fE)

    # prepare output
    lut = np.empty((len(z_lut), len(V_lut), len(E_lut)), dtype=np.float64)

    def _calc_z(i, zval):
        print(f"  [LUT] {i+1}/{len(z_lut)} z={zval:.2f} nm")
        Vmesh, Emesh = np.meshgrid(V_lut, E_lut, indexing='ij')
        D = transmission(Emesh, Vmesh, zval, phi_t, phi_s, fudge=fudge).astype(np.float64)
        return i, D

    # parallel compute per z-layer
    results = Parallel(n_jobs=-1)(delayed(_calc_z)(i, z) for i, z in enumerate(z_lut))
    for idx, D_slice in results:  # type: ignore
        lut[idx] = D_slice

    return lut

###############################################################################
# LUT management utilities
###############################################################################

def list_existing_luts(fer_output_dir='C:/Users/willh/OneDrive/Desktop/FER_Simulation/fer_output'):
    """List existing LUT files and their parameters."""
    lut_dir = Path(fer_output_dir)
    if not lut_dir.exists():
        print("No fer_output directory found.")
        return
    
    lut_files = list(lut_dir.glob("transmission_lut_*.h5"))
    if not lut_files:
        print("No LUT files found.")
        return
    
    print(f"Found {len(lut_files)} LUT file(s):")
    for lut_file in sorted(lut_files):
        try:
            with h5py.File(lut_file, 'r') as f:
                phi_tip = f.attrs.get('phi_tip', 'N/A')
                phi_samp = f.attrs.get('phi_samp', 'N/A')
                n_E = len(f['E']) if 'E' in f else 'N/A'
                n_V = len(f['V']) if 'V' in f else 'N/A'
                n_Z = len(f['z']) if 'z' in f else 'N/A'
                file_size = lut_file.stat().st_size / (1024*1024)  # MB
                print(f"  {lut_file.name}")
                print(f"    Size: {file_size:.1f} MB")
                print(f"    Parameters: phi_tip={phi_tip}, phi_samp={phi_samp}")
                print(f"    Grid: n_E={n_E}, n_V={n_V}, n_Z={n_Z}")
        except Exception as e:
            print(f"  {lut_file.name} (error reading: {e})")
        print()

###############################################################################
# RF averaging & current
###############################################################################

def interp_lut(lut, Eg, Vg, zg):
    return RegularGridInterpolator((zg, Vg, Eg), lut, bounds_error=False, fill_value=0.0)


def D_avg(interp, Eg, Vdc, Arf, z, nodes):
    """RF‑averaged D(E) using precomputed LUT interpolator."""
    V_inst = Vdc + Arf * np.cos(nodes)              # (N,)
    nE, nN = Eg.size, V_inst.size
    # repeat grids so shapes match
    E_rep = np.repeat(Eg, nN)                      # (nE*nN,)
    V_rep = np.tile(V_inst, nE)                    # (nE*nN,)
    z_rep = np.full(E_rep.shape, z)

    pts = np.stack([z_rep, V_rep, E_rep], axis=1)  # (nE*nN, 3)
    vals = interp(pts).reshape(nE, nN)             # (nE, N)
    return vals.mean(axis=1)                       # -> (nE,)(pts).reshape(len(V_inst), len(Eg)).mean(axis=0)


def current(D_E, Eg, Vdc):
    D_E = np.asarray(D_E, dtype=np.float64)
    Eg = np.asarray(Eg, dtype=np.float64)
    if D_E.ndim != 1 or Eg.ndim != 1 or D_E.shape != Eg.shape:
        raise ValueError("D_E and Eg must be one-dimensional arrays with matching shapes")
    if not (np.all(np.isfinite(D_E)) and np.all(np.isfinite(Eg)) and np.isfinite(Vdc)):
        raise ValueError("D_E, Eg, and Vdc must contain only finite values")
    if np.any(np.diff(Eg) <= 0):
        raise ValueError("Eg must be strictly increasing")

    mask = Eg < (E_F - Vdc)
    p1 = np.trapezoid(Vdc*D_E[mask], Eg[mask]) if mask.any() else 0.0
    p2 = np.trapezoid((E_F - Eg[~mask])*D_E[~mask], Eg[~mask]) if (~mask).any() else 0.0
    return p1 + p2


###############################################################################
# simulation
###############################################################################

def simulate(a):
    Eg = np.linspace(a.E_min, a.E_max + a.e_extra, a.n_E)
    Vg = np.linspace(a.v_min, a.v_max, a.n_V)
    zg = np.linspace(a.z_min, a.z_max, a.n_Z)
    Ag = np.linspace(a.A_min, a.A_max, a.n_A)
    nodes = np.pi*(2*np.arange(1, a.n_cheb+1)-1)/(2*a.n_cheb)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    
    dV = (a.v_max-a.v_min)/(a.n_V - 1)
    Vg_lut = np.linspace(a.v_min-a.A_max, a.v_max+a.A_max, a.n_V + int(np.ceil(2*a.A_max/dV)))

    fudge=3.69
    fudge, samples = calibrate_fudge(
        phi_t=a.phi_tip, phi_s=a.phi_samp, F1=0.2,
        D_min=a.z_min, D_max=a.z_max, n_D=100, n_E=200
    )
    print(f'(fudge, samples) = ({fudge}, {samples})')

    if getattr(a, 'use_lut', False):
        def grids_match(file, Eg, Vg_lut, zg, phi_tip, phi_samp):
            # Compare all relevant parameters
            try:
                if not np.allclose(file['E'][...], Eg): return False
                if not np.allclose(file['V'][...], Vg_lut): return False
                if not np.allclose(file['z'][...], zg): return False
                if abs(file.attrs['phi_tip'] - phi_tip) > 1e-8: return False
                if abs(file.attrs['phi_samp'] - phi_samp) > 1e-8: return False
                return True
            except Exception:
                return False

        # Determine upscaling factors first
        upscale_z = getattr(a, 'upscale_z', None) or a.upscale
        upscale_v = getattr(a, 'upscale_v', None) or a.upscale  
        upscale_e = getattr(a, 'upscale_e', None) or a.upscale
        upscale_factors = (upscale_z, upscale_v, upscale_e)
        z_lut = _upscaled_axis(zg, upscale_z)
        V_lut = _upscaled_axis(Vg_lut, upscale_v)
        E_lut = _upscaled_axis(Eg, upscale_e)
        
        # Create parameter-based LUT filename
        upscale_str = f"_up{upscale_z}{upscale_v}{upscale_e}" if upscale_factors != (1,1,1) else ""
        lut_filename = f"transmission_lut_phit{a.phi_tip:.1f}_phis{a.phi_samp:.1f}_nE{len(E_lut)}_nV{len(V_lut)}_nZ{len(z_lut)}{upscale_str}.h5"
        lut_file = out / lut_filename
        rebuild_lut = True
        if lut_file.exists() and not getattr(a, 'force_rebuild_lut', False):
            with h5py.File(lut_file, 'r') as f:
                if grids_match(f, E_lut, V_lut, z_lut, a.phi_tip, a.phi_samp):
                    print('[load] LUT')
                    lut = np.array(f['D'][...])
                    rebuild_lut = False
                else:
                    print('[rebuild] LUT: grid or parameters changed.')
        elif getattr(a, 'force_rebuild_lut', False):
            print('[rebuild] LUT: forced rebuild requested.')
        
        if rebuild_lut:
            print(f'[build] LUT with upscaling factors: z={upscale_z}, v={upscale_v}, e={upscale_e}')
            lut = build_lut(Eg, Vg_lut, zg, a.phi_tip, a.phi_samp, upscale=upscale_factors, fudge=fudge)
            with h5py.File(lut_file, 'w') as f:
                f.create_dataset('D', data=lut, compression='gzip')
                f.create_dataset('E', data=E_lut)
                f.create_dataset('V', data=V_lut)
                f.create_dataset('z', data=z_lut)
                f.attrs['phi_tip'] = a.phi_tip
                f.attrs['phi_samp'] = a.phi_samp
        interp = interp_lut(lut, E_lut, V_lut, z_lut)

        I = np.zeros((len(zg), len(Vg), len(Ag)))
        def _row(i,z):
            print(f"  [I] z {i+1}/{len(zg)}")
            r = np.zeros((len(Vg), len(Ag)))
            for j,Vdc in enumerate(Vg):
                for k,Ar in enumerate(Ag):
                    D_bar = D_avg(interp, Eg, Vdc, Ar, z, nodes)
                    r[j,k] = current(D_bar, Eg, Vdc)
            return i,r
        
        res = Parallel(n_jobs=a.threads)(delayed(_row)(i,z) for i,z in enumerate(zg))  # type: ignore
        for i,r in res: I[i]=r  # type: ignore
    else:
        I = np.zeros((len(zg), len(Vg), len(Ag)))
        def _row(i,z):
            print(f"  [I] z {i+1}/{len(zg)} (direct)")
            r = np.zeros((len(Vg), len(Ag)))
            for j,Vdc in enumerate(Vg):
                for k,Ar in enumerate(Ag):
                    V_inst = Vdc + Ar * np.cos(nodes)
                    D_bar = np.zeros_like(Eg)
                    for idx_E, E in enumerate(Eg):
                        E_arr = np.full_like(V_inst, E)
                        z_arr = np.full_like(V_inst, z)
                        T_vals = transmission(E_arr, V_inst, z_arr, a.phi_tip, a.phi_samp)
                        D_bar[idx_E] = T_vals.mean()
                    r[j,k] = current(D_bar, Eg, Vdc)
            return i,r
        res = Parallel(n_jobs=a.threads)(delayed(_row)(i,z) for i,z in enumerate(zg))  # type: ignore
        for i,r in res: I[i]=r  # type: ignore

    # Create parameterized filename for current results
    current_filename = f"current_phit{a.phi_tip:.1f}_phis{a.phi_samp:.1f}_nE{a.n_E}_nV{a.n_V}_nZ{a.n_Z}_nA{a.n_A}.h5"
    current_file = out / current_filename
    
    with h5py.File(current_file, 'w') as f:
        # Main data
        f.create_dataset('I', data=I, compression='gzip')
        f.create_dataset('z', data=zg)
        f.create_dataset('V', data=Vg)
        f.create_dataset('A_rf', data=Ag)
        
        # Simulation parameters as attributes (ensure all are present)
        f.attrs['phi_tip'] = float(a.phi_tip)
        f.attrs['phi_samp'] = float(a.phi_samp)
        f.attrs['n_E'] = int(a.n_E)
        f.attrs['n_V'] = int(a.n_V)
        f.attrs['n_Z'] = int(a.n_Z)
        f.attrs['n_A'] = int(a.n_A)
        f.attrs['n_cheb'] = int(a.n_cheb)
        f.attrs['use_lut'] = getattr(a, 'use_lut', False)
        f.attrs['fudge_factor'] = fudge
        f.attrs['E_F'] = E_F
        
        # Grid information
        f.attrs['dV'] = dV
        f.attrs['n_V_lut'] = len(Vg_lut)
        
    print('[save] current.h5 written with parameters')

# ─── 2. HELPER: apply YAML to the parsed Namespace ───────────────────────────
def _merge_yaml_into_args(yaml_path: str, parser: argparse.ArgumentParser,
                          args: argparse.Namespace) -> argparse.Namespace:
    """
    Read yaml_path and copy keys under 'simulation:' into `args`.
    Command-line overrides take precedence: if the user supplied a value that
    differs from the parser default, keep that value.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    sim_cfg = cfg.get('simulation', cfg)        # tolerate no top-level key
    destinations = {
        action.dest.lower(): action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    for k, v in sim_cfg.items():
        destination = destinations.get(k.lower())
        if destination is None:
            continue                            # ignore unexpected keys
        default_val = parser.get_default(destination)
        current_val = getattr(args, destination)
        # if the current value is still the default, replace it with YAML
        if current_val == default_val:
            setattr(args, destination, v)
    return args

# ─── 3. CLI: add --config flag and call the helper ───────────────────────────
def cli():
    p = argparse.ArgumentParser()
    # 3a) NEW flag – goes before the long list so it’s visible in --help
    p.add_argument('--config', type=str,
                   help='YAML file with a simulation: block of parameters')
    # 3b) EXISTING options (unchanged) …
    p.add_argument('--n_E', type=int, default=100)
    p.add_argument('--n_V', type=int, default=100)
    p.add_argument('--n_Z', type=int, default=100)
    p.add_argument('--n_A', type=int, default=30)
    p.add_argument('--E_min', type=np.float64, default=0.01)
    p.add_argument('--E_max', type=np.float64, default=E_F)
    p.add_argument('--e_extra', type=np.float64, default=0.0)
    p.add_argument('--v_min', type=np.float64, default=0.0)
    p.add_argument('--v_max', type=np.float64, default=10.0)
    p.add_argument('--z_min', type=np.float64, default=0.3)
    p.add_argument('--z_max', type=np.float64, default=5.0)
    p.add_argument('--A_min', type=np.float64, default=0.0)
    p.add_argument('--A_max', type=np.float64, default=5.0)
    p.add_argument('--phi_tip', type=np.float64, default=4.0)
    p.add_argument('--phi_samp', type=np.float64, default=5.0)
    p.add_argument('--n_cheb', type=int, default=32)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--out', type=str,
                   default='C:/Users/willh/OneDrive/Desktop/FER_Simulation/fer_output')
    p.add_argument('--force-rebuild-lut', action='store_true',
                   help='Force rebuild of LUT even if compatible one exists')
    p.add_argument('--list-luts', action='store_true',
                   help='List existing LUT files and exit')
    lut_mode = p.add_mutually_exclusive_group()
    lut_mode.add_argument('--use-lut', dest='use_lut', action='store_true',
                          help='Use the transmission LUT (default)')
    lut_mode.add_argument('--no-lut', '--no-use-lut', dest='use_lut', action='store_false',
                          help='Calculate transmission directly')
    p.set_defaults(use_lut=True)
    p.add_argument('--upscale', type=int, default=1,
                   help='Upscale LUT resolution by this factor (default: 1)')
    p.add_argument('--upscale-z', type=int,
                   help='Upscale z-axis resolution (overrides --upscale)')
    p.add_argument('--upscale-v', type=int,
                   help='Upscale voltage-axis resolution (overrides --upscale)')
    p.add_argument('--upscale-e', type=int,
                   help='Upscale energy-axis resolution (overrides --upscale)')

    # ---- PARSE once, then merge YAML defaults -------------------------------
    args = p.parse_args()
    config_path = Path(args.config) if args.config else Path('fer_config.yaml')
    if config_path.exists():
        args = _merge_yaml_into_args(str(config_path), p, args)
    elif args.config:
        p.error(f"config file not found: {config_path}")

    # ---- Act on --list-luts early -------------------------------------------
    if args.list_luts:
        list_existing_luts(args.out)
        return

    # ---- Launch simulation ---------------------------------------------------
    t0 = time.perf_counter()
    current_file = simulate(args)
    print(f"[done] total runtime {time.perf_counter()-t0:.1f} s")
    return current_file


if __name__ == '__main__':
    print('Begin simulation...')
    cli()
    print('Simulation done')
    # Automatically run quick_check.py after simulation
    import subprocess
    print('Running quick_check.py for immediate analysis...')
    subprocess.run(['python', 'quick_check.py'])
    # os.system('git push')