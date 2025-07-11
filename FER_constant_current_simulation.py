import argparse
import os
from pathlib import Path
import time

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
    Solve a x^3 + b x^2 + c x + d = 0 for arrays a,b,c,d of the same shape.
    Returns two real roots per equation: the two smallest (by value) of all real solutions.
    """
    # Normalize to x^3 + B x^2 + C x + D = 0
    B = b / a
    C = c / a
    D = d / a

    # Depressed cubic substitute x = y - B/3
    p = C - B*B/3
    q = 2*B**3/27 - B*C/3 + D

    # Discriminant
    Δ = (q/2)**2 + (p/3)**3

    # Prepare output
    n = a.size
    roots = np.empty((n, 3), dtype=np.complex128)

    # Case A: Δ >= 0 → one real, two complex
    mA = Δ >= 0
    if np.any(mA):
        sqrtΔ = np.sqrt(Δ[mA])
        u = np.cbrt(-q[mA]/2 + sqrtΔ)
        v = np.cbrt(-q[mA]/2 - sqrtΔ)
        y1 = u + v
        roots[mA, 0] = y1
        # the other two are complex conjugates:
        roots[mA, 1] = -y1/2 + 0.5j*np.sqrt(3)*(u - v)
        roots[mA, 2] = -y1/2 - 0.5j*np.sqrt(3)*(u - v)

    # Case B: Δ < 0 → three real roots
    mB = ~mA
    if np.any(mB):
        r = np.sqrt(-(p[mB]/3)**3)
        θ = np.arccos(-q[mB]/(2*r))
        m = np.cbrt(r)
        roots[mB, 0] = 2*m*np.cos(θ/3)
        roots[mB, 1] = 2*m*np.cos((θ+2*np.pi)/3)
        roots[mB, 2] = 2*m*np.cos((θ+4*np.pi)/3)

    # Shift back x = y - B/3
    roots = roots - B[:,None]/3

    # For each equation, pick the two smallest *real* roots
    real_roots = []
    for sol in roots:
        reals = np.sort(sol.real[np.abs(sol.imag) < tol])
        # if fewer than two real, pad with NaN
        if reals.size < 2:
            reals = np.pad(reals, (0,2-reals.size), constant_values=np.nan)
        real_roots.append(reals[:2])
    return np.array(real_roots)  # shape (n,2)

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

        # clamp t so exp(t) never overflows
        t_clip = np.minimum(t, np.log(np.finfo(np.float64).max))  # ≈ 709
        exp_pos = np.exp(t_clip)
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
        out[2][mask_pos] = preB  * exp_pos * corrB
        out[3][mask_pos] = preBp * exp_pos * corrBp

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
    Two-regime transmission:
      - Simmons trapezoidal WKB for |F| <= F1:
          log T = - (4 a)/(3 F) [W0^1.5 - Ws^1.5]
      - Airy-based trapezoid for |F| > F1
    """
    import numpy as np

    # 1) ensure arrays
    E = np.asarray(E, np.float64)
    V = np.asarray(V, np.float64)

    # 2) relabel electrodes
    swap      = V < 0
    V_eff     = np.abs(V)
    phi1      = np.where(swap, phi_s, phi_t)
    phi2      = np.where(swap, phi_t, phi_s)
    # phi1     += image_potential(d, V_eff, phi1) # This needs work

    # 3) barrier tops & field
    W0         = phi1 + E_F - E
    Ws_minus_V = phi2 + E_F - E - V_eff
    F_s        = (V_eff + phi1 - phi2) / d

    # 4) prepare output
    T = np.zeros_like(E, dtype=np.float64)

    # 5) Simmons-WKB region: |F| <= F1 and both barriers > 0
    m_wkb = (np.abs(F_s) <= F1) & (W0 > 0) & (Ws_minus_V > 0)
    if np.any(m_wkb):
        F_wkb   = F_s[m_wkb]
        W0w     = W0[m_wkb]
        Wsw     = Ws_minus_V[m_wkb]
        delta   = W0w**1.5 - Wsw**1.5
        logT    = - (4*a) / (3 * F_wkb) * delta
        T[m_wkb] = fudge*np.exp(logT)

    # 6) Airy region: |F| > F1
    m_air = ~m_wkb
    if np.any(m_air):
        F_air   = F_s[m_air]
        factor  = (a / np.abs(F_air))**(2/3)
        z0      = factor * W0[m_air]
        zs      = factor * Ws_minus_V[m_air]
        zp      = np.sign(F_air) * (a**2 * np.abs(F_air))**(1/3)

        Ai0, Aip0, Bi0, Bip0 = airy_all(z0, z_switch=z_switch)
        Ais, Aips, Bis, Bips = airy_all(zs, z_switch=z_switch)

        E_a      = E[m_air]
        V_a      = V_eff[m_air]
        k1       = a * np.sqrt(np.maximum(E_a, 1e-3))
        k3       = a * np.sqrt(np.maximum(E_a + V_a, 1e-12))
        num      = (k3 / k1) * (4 / np.pi**2)

        t1 = Aip0*Bips - Aips*Bip0
        t2 = Ai0*Bis   - Ais*Bi0
        t3 = Ais*Bip0  - Aip0*Bis
        t4 = Ai0*Bips  - Aips*Bi0

        denom = ((zp/k1)*t1 + (k3/zp)*t2)**2 \
              + ((k3/k1)*t3 + t4)**2
        denom = np.where(np.abs(denom) < 1e-12, np.inf, denom)

        T[m_air] = num / denom
        #T = np.maximum(T, 1e-30)
    return T


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
    f_avg = float(np.mean(samples))

    return f_avg, samples

###############################################################################
# LUT with progress
###############################################################################

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
      Transmission probability
    z_lut, V_lut, E_lut : ndarray
      The upscaled axis arrays
    """
    # parse upscale factors
    if isinstance(upscale, int):
        fz = fV = fE = upscale
    else:
        fz, fV, fE = upscale  # type: ignore

    # build high-res axes
    z_lut = np.linspace(zg[0], zg[-1], (len(zg)-1)*fz + 1)
    V_lut = np.linspace(Vg[0], Vg[-1], (len(Vg)-1)*fV + 1)
    E_lut = np.linspace(Eg[0], Eg[-1], (len(Eg)-1)*fE + 1)

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

def list_existing_luts(fer_output_dir='fer_output'):
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
    mask = Eg < (E_F - Vdc)
    p1 = np.trapz(Vdc*D_E[mask], Eg[mask]) if mask.any() else 0.0
    p2 = np.trapz((E_F - Eg[~mask])*D_E[~mask], Eg[~mask]) if (~mask).any() else 0.0
    return p1 + p2


###############################################################################
# simulation
###############################################################################

def simulate(a):
    Eg = np.linspace(0.01, E_F + a.e_extra, a.n_E)
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
        D_min=a.v_min, D_max=a.v_max, n_D=100, n_E=200
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

        # Create parameter-based LUT filename
        upscale_str = f"_up{upscale_z}{upscale_v}{upscale_e}" if upscale_factors != (1,1,1) else ""
        lut_filename = f"transmission_lut_phi{a.phi_tip:.1f}_{a.phi_samp:.1f}_nE{len(Eg)}_nV{len(Vg_lut)}_nZ{len(zg)}{upscale_str}.h5"
        lut_file = Path('fer_output') / lut_filename
        rebuild_lut = True
        if lut_file.exists() and not getattr(a, 'force_rebuild_lut', False):
            with h5py.File(lut_file, 'r') as f:
                if grids_match(f, Eg, Vg_lut, zg, a.phi_tip, a.phi_samp):
                    print('[load] LUT')
                    lut = np.array(f['D'][...])
                    Eg = np.array(f['E'][...])
                    Vg_lut = np.array(f['V'][...])
                    zg = np.array(f['z'][...])
                    rebuild_lut = False
                else:
                    print('[rebuild] LUT: grid or parameters changed.')
        elif getattr(a, 'force_rebuild_lut', False):
            print('[rebuild] LUT: forced rebuild requested.')

        # Determine upscaling factors
        upscale_z = getattr(a, 'upscale_z', None) or a.upscale
        upscale_v = getattr(a, 'upscale_v', None) or a.upscale  
        upscale_e = getattr(a, 'upscale_e', None) or a.upscale
        upscale_factors = (upscale_z, upscale_v, upscale_e)
        
        if rebuild_lut:
            print(f'[build] LUT with upscaling factors: z={upscale_z}, v={upscale_v}, e={upscale_e}')
            lut = build_lut(Eg, Vg_lut, zg, a.phi_tip, a.phi_samp, upscale=upscale_factors, fudge=fudge)
            with h5py.File(lut_file, 'w') as f:
                f.create_dataset('D', data=lut, compression='gzip')
                f.create_dataset('E', data=Eg)
                f.create_dataset('V', data=Vg_lut)
                f.create_dataset('z', data=zg)
                f.attrs['phi_tip'] = a.phi_tip
                f.attrs['phi_samp'] = a.phi_samp
        interp = interp_lut(lut, Eg, Vg_lut, zg)

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

    with h5py.File(out/'current.h5','w') as f:
        f.create_dataset('I', data=I, compression='gzip')
        f.create_dataset('z', data=zg); f.create_dataset('V', data=Vg); f.create_dataset('A_rf', data=Ag)
        f.attrs['phi_tip']=a.phi_tip; f.attrs['phi_samp']=a.phi_samp
    print('[save] current.h5 written')

###############################################################################
# CLI
###############################################################################

def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--n_E', type=int, default=100)
    p.add_argument('--n_V', type=int, default=100)
    p.add_argument('--n_Z', type=int, default=100)
    p.add_argument('--n_A', type=int, default=10)
    p.add_argument('--e_extra', type=np.float64, default=0.0)
    p.add_argument('--v_min', type=np.float64, default=1.0)
    p.add_argument('--v_max', type=np.float64, default=10.0)
    p.add_argument('--z_min', type=np.float64, default=0.3)
    p.add_argument('--z_max', type=np.float64, default=5.0)
    p.add_argument('--A_min', type=np.float64, default=0.0)
    p.add_argument('--A_max', type=np.float64, default=5.0)
    p.add_argument('--phi_tip', type=np.float64, default=4.0)
    p.add_argument('--phi_samp', type=np.float64, default=4.0)
    p.add_argument('--n_cheb', type=int, default=32)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--out', type=str, default='fer_output')
    p.add_argument('--force-rebuild-lut', action='store_true', help='Force rebuild of LUT even if compatible one exists')
    p.add_argument('--list-luts', action='store_true', help='List existing LUT files and exit')
    p.add_argument('--use-lut', action='store_true', default=False, help='Use LUT for transmission (default: direct calculation)')
    p.add_argument('--upscale', type=int, default=1, help='Upscale LUT resolution by this factor (default: 1)')
    p.add_argument('--upscale-z', type=int, help='Upscale z-axis resolution (overrides --upscale)')
    p.add_argument('--upscale-v', type=int, help='Upscale voltage-axis resolution (overrides --upscale)')
    p.add_argument('--upscale-e', type=int, help='Upscale energy-axis resolution (overrides --upscale)')
    args = p.parse_args()
    
    if args.list_luts:
        list_existing_luts()
        return
    
    t0 = time.perf_counter()
    simulate(args)
    print(f"[done] total runtime {time.perf_counter()-t0:.1f} s")


if __name__ == '__main__':
    print('Begin simulation...')
    cli()
    print('Simulation done')