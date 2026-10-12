import argparse
import os
from pathlib import Path
import time

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.special import airy as sp_airy

# This code fails to save the data if it can't allocate enough memory

###############################################################################
# constants
###############################################################################
E_F = 5.5  # Fermi energy (eV)
a = 5.12  # sqrt(2m)/hbar with units of 1/(nm*sqrt(eV)) -- differs from Gundlach's by a factor of 2
###############################################################################
# Utilities
###############################################################################

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
        denom = np.where(denom < 1e-12, np.inf, denom)

        T[m_air] = num / denom
        #T = np.maximum(T, 1e-30)
    return T


 
###############################################################################
# LUT with progress
###############################################################################

def build_lut(Eg, Vg, zg, phi_t, phi_s, upscale=1):
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
        D = transmission(Emesh, Vmesh, zval, phi_t, phi_s).astype(np.float64)
        return i, D

    # parallel compute per z-layer
    results = Parallel(n_jobs=-1)(delayed(_calc_z)(i, z) for i, z in enumerate(z_lut))  # type: ignore
    for idx, D_slice in results:  # type: ignore
        lut[idx] = D_slice

    return lut

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

def _D_grid(interp, Eg, Vg, Ag, z_val, nodes):
    """
    Vectorised RF-average for full A×E×V grid at a single tip-height z_val.

    Returns D_ae_v array of shape (nA, nE, nV) where
      D_ae_v[a, e, v] = ⟨D(Eg[e]; Vg[v], Ag[a])⟩_RF
    """
    # lengths
    nA, nE, nV = len(Ag), len(Eg), len(Vg)
    nN = len(nodes)

    # build instantaneous voltage grid: shape (nA, 1, nV, nN)
    Vdc = Vg[None, None, :, None]           # 1×1×nV×1
    Arf = Ag[:, None, None, None]           # nA×1×1×1
    phase = np.cos(nodes)[None, None, None, :]  # 1×1×1×nN
    V_inst = Vdc + Arf*phase                # nA×1×nV×nN

    # broadcast energy grid to same shape
    E_grid = Eg[None, :, None, None]        # 1×nE×1×1
    V_b    = np.broadcast_to(V_inst, (nA, nE, nV, nN))
    E_b    = np.broadcast_to(E_grid, (nA, nE, nV, nN))

    # form the list of points for interpolation
    pts = np.column_stack([
      np.full(V_b.size, z_val),  # z
      V_b.ravel(),               # V
      E_b.ravel()                # E
    ])                            # shape (nA*nE*nV*nN, 3)

    # interpolate and average over the RF cycle (axis=-1)
    D = interp(pts).reshape(nA, nE, nV, nN)
    return D.mean(axis=3)         # → shape (nA, nE, nV)

def current(D_E, Eg, Vdc):
    mask = Eg < (E_F - Vdc)
    p1 = np.trapz(Vdc*D_E[mask], Eg[mask]) if mask.any() else 0.0
    p2 = np.trapz((E_F - Eg[~mask])*D_E[~mask], Eg[~mask]) if (~mask).any() else 0.0
    return p1 + p2

def compute_I_vs_VA(interp, Eg, Vg, Ag, z, nodes,
                    w, mask1, mask2):
    D_ae_v = _D_grid(interp, Eg, Vg, Ag, z, nodes)   # (nA,nE,nV)
    D_v_a_e = D_ae_v.transpose(2,0,1)                # (nV,nA,nE)

    # p1,p2 as you already have
    p1 = Vg[:,None] * np.einsum('vae,ve->va', D_v_a_e, w[None,:]*mask1)
    p2 =          np.einsum('vae,ve->va', D_v_a_e*(E_F-Eg)[None,None,:],
                            w[None,:]*mask2)
    return p1 + p2


###############################################################################
# simulation
###############################################################################


def simulate(a):
    # 1) build all axes
    Eg    = np.linspace(0.01, E_F + a.e_extra, a.n_E)
    Vg    = np.linspace(a.v_min,   a.v_max, a.n_V)
    zg    = np.linspace(a.z_min,   a.z_max, a.n_Z)
    Ag    = np.linspace(a.A_min,   a.A_max, a.n_A)
    nodes = np.pi*(2*np.arange(1,a.n_cheb+1)-1)/(2*a.n_cheb)

    # 2) build LUT / interpolator (exactly as you have)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    
    dV = (a.v_max-a.v_min)/(a.n_V - 1)
    Vg_lut = np.linspace(a.v_min-a.A_max, a.v_max+a.A_max, a.n_V + int(np.ceil(2*a.A_max/dV)))

    lut_file = out / 'transmission_lut.h5'
    if lut_file.exists():
        print('[load] LUT')
        with h5py.File(lut_file, 'r') as f:
            lut = np.array(f['D'][...])  # type: ignore
            Eg = np.array(f['E'][...])  # type: ignore
            Vg_lut = np.array(f['V'][...])  # type: ignore
            zg = np.array(f['z'][...])  # type: ignore
    else:
        print('[build] LUT')
        lut = build_lut(Eg, Vg_lut, zg, a.phi_tip, a.phi_samp, upscale=1)
        with h5py.File(lut_file, 'w') as f:
            f.create_dataset('D', data=lut, compression='gzip')
            f.create_dataset('E', data=Eg); f.create_dataset('V', data=Vg_lut); f.create_dataset('z', data=zg)
    interp = interp_lut(lut, Eg, Vg_lut, zg)

    # 3) precompute masks & weights
    dE     = Eg[1]-Eg[0]
    w      = np.ones_like(Eg)*dE; w[0]*=0.5; w[-1]*=0.5
    thresh = E_F - Vg       # (nV,)
    mask1  = Eg[None,:] < thresh[:,None]
    mask2  = ~mask1

    # 4) parallel over z, calling compute_I_vs_VA
    I = np.empty((len(zg), len(Vg), len(Ag)), dtype=float)
    def _row(i, z):
        print(f" layer {i+1}/{len(zg)}: z={z:.2f} nm")
        return i, compute_I_vs_VA(interp, Eg, Vg, Ag, z, nodes,
                                  w, mask1, mask2)  # pass masks/weights if you refactor signature

    res = Parallel(n_jobs=a.threads)(
        delayed(_row)(i,z) for i,z in enumerate(zg))  # type: ignore
    for i, I_layer in res:  # type: ignore
        I[i] = I_layer

    # 5) dump to HDF5
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
    p.add_argument('--n_V', type=int, default=300)
    p.add_argument('--n_Z', type=int, default=100)
    p.add_argument('--n_A', type=int, default=100)
    p.add_argument('--e_extra', type=np.float64, default=0.0)
    p.add_argument('--v_min', type=np.float64, default=-2.0)
    p.add_argument('--v_max', type=np.float64, default=10.0)
    p.add_argument('--z_min', type=np.float64, default=0.3)
    p.add_argument('--z_max', type=np.float64, default=5.0)
    p.add_argument('--A_min', type=np.float64, default=0.0)
    p.add_argument('--A_max', type=np.float64, default=1.0)
    p.add_argument('--phi_tip', type=np.float64, default=4.0)
    p.add_argument('--phi_samp', type=np.float64, default=4.7)
    p.add_argument('--n_cheb', type=int, default=32)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--out', type=str, default='fer_output')
    args = p.parse_args()
    t0 = time.perf_counter()
    simulate(args)
    print(f"[done] total runtime {time.perf_counter()-t0:.1f} s")


if __name__ == '__main__':
    print('Begin simulation...')
    cli()
    print('Simulation done')