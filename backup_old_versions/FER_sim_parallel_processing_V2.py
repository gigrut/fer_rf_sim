import os
import time
import numpy as np
from joblib import Parallel, delayed
# Rename "airy" upon import to avoid some Windows pickling edge cases
from scipy.special import airy as scipy_airy
from multiprocessing import freeze_support

np.seterr(invalid='warn', divide='warn')

###############################################################################
# Physical Constants & Parameters
###############################################################################
a = 5.12*0.9             # 1 / (nm * sqrt(eV))
k = a
E_F = 5.5                   # Fermi energy (eV)
phi_t = 4                # Tip work function (eV)
phi_s = 4              # Sample work function (eV)

FIELD_ENHANCEMENT_FACTOR = 1
Rectification_Factor = 1
Barrier_Relative_Permitivity = 1
Lateral_Confinement_Potential = 0


# Choose a step-size category
step_size = 'large'  # e.g. 'tiny', 'small', 'medium', 'large', or 'huge'

# Step definitions
if step_size == 'tiny':  # slow
    D_start, D_stop, D_step = 0.5, 5.0, 0.05
    V_start, V_stop, V_step = 2, 10, 0.025
    A_start, A_stop, A_step = 0, 2, 0.02
    x_start, x_stop, x_step = 0.05, E_F, 0.025
elif step_size == 'small':  # slow
    D_start, D_stop, D_step = 0.5, 5.0, 0.05
    V_start, V_stop, V_step = 2.1, 10, 0.025
    A_start, A_stop, A_step = 0, 2, 0.02
    x_start, x_stop, x_step = 0.025, E_F, 0.025
elif step_size == 'medium':
    D_start, D_stop, D_step = 0.5, 5.0, 0.05 * 1.278
    V_start, V_stop, V_step = 2, 10, 0.05 * 1.78
    A_start, A_stop, A_step = 0, 2, 0.02 * 1.78
    x_start, x_stop, x_step = 0.025 * 1.78, E_F, 0.01 * 1.78
elif step_size == 'large':
    D_start, D_stop, D_step = 0.01, 10.0, 0.01* 10
    V_start, V_stop, V_step = 0.0, 10.0, 0.005 * 10
    A_start, A_stop, A_step = 0.0, 4.0, 0.1 * 1
    x_start, x_stop, x_step = 0.01, E_F, 0.01 * 10
elif step_size == 'huge':
    D_start, D_stop, D_step = 0.5, 2.0, 0.5
    V_start, V_stop, V_step = 2, 10, 0.25
    A_start, A_stop, A_step = 0, 1, 0.2
    x_start, x_stop, x_step = 0.25, E_F, 0.25

D_values = np.arange(D_start, D_stop + D_step, D_step, dtype=np.float32)
V_values = np.arange(V_start, V_stop + V_step, V_step, dtype=np.float32)
A_values = np.arange(A_start, A_stop + A_step, A_step, dtype=np.float32)
x_values = np.arange(x_start, x_stop + x_step, x_step, dtype=np.float32)

nD, nV, nA, nX = len(D_values), len(V_values), len(A_values), len(x_values)

# Used for RF averaging
N = 30
n = np.arange(1, N + 1)
angles = np.pi * (2 * n - 1) / (2 * N)
cos_values = np.cos(angles).astype(np.float32)  # shape (N,)

def image_potential(D, V):
    """
    Compute the effective barrier height (phi_eff) using Simmons's
    approximations (Eqs. (37)–(39)).

    Parameters:
      D : barrier thickness (nm) (scalar)
      V : applied voltage (eV) (array or scalar)
      
    Returns:
      phi_eff : effective barrier height (eV)
    """
    
    # Ensure V is an array
    V = np.asarray(V, dtype=float)

    # Define coefficients for the cubic equation
    def solve_roots(V_val):
        """ Solve the cubic equation for a single value of V. """
        coeffs = [V_val/D, phi_t+V_val, phi_t * D, -4.255 * D/Barrier_Relative_Permitivity]
        roots = np.roots(coeffs)
        real_roots = np.sort(roots.real)
        return real_roots[:2]  # Return the smallest two real roots

    # Use np.vectorize to apply `solve_roots` elementwise to V
    s1, s2 = np.vectorize(solve_roots, otypes=[float, float])(V)
    # Compute delta_s (avoid division by zero)
    delta_s = s2 - s1 + 1e-30

    # Compute log argument, adding a small term to prevent division errors
    log_arg = (s2 * (D - s1)) / (s1 * (D - s2))

    # Compute phi_eff elementwise
    L = 3.7 / (D*Barrier_Relative_Permitivity)
    phi_im = - ((V * (s1 + s2) / (2 * D)) - (1.15 * L * D / delta_s) * np.log(np.abs(log_arg) + 1e-30))

    return phi_im

###############################################################################
# Wrappers to evaluate airy and rectangular (helps with pickling issues on Windows) 
###############################################################################
def T_rectangular(E, D, Delta, U):
    """
    Return 1D rectangular barrier transmission (sub-barrier)
    E  = electron energy (eV)
    D  = barrier thickness (nm)
    Delta = offset in region 3 (eV)
    U  = barrier height (eV)
    
    We do sub-barrier formula only:
       T = 1 / [1 + ((k2^2 + k1^2)*(k2^2 + k3^2)) / (4*k1^2*k3^2) * sinh^2(|k2|*D)]
    with k1 = a*sqrt(E), k2^2=-a^2(U-E), k3=a*sqrt(E+Delta).
    
    'a' is the global wavevector constant for your eV->1/nm conversions.
    """

    # ensure arrays
    E     = np.asarray(E, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    U     = np.asarray(U, dtype=float)

    # wavevectors
    k1 = a * np.sqrt(np.maximum(E, 0.0)) 
    # sub barrier => k2^2 = -a^2(U-E), negative => imaginary
    k2sq = -a*a * (U - E)
    k2_abs = np.sqrt(np.maximum(-k2sq, 0.0))
    E3 = E + Delta
    k3 = a * np.sqrt(np.maximum(E3, 0.0))

    denom_factor = 4.0*(k1**2)*(k3**2) + 1e-30
    termA = (k2sq + k1**2) 
    termB = (k2sq + k3**2)
    arg   = k2_abs*D
    S2    = np.sinh(arg)**2

    sub_mask = (E < U)
    T_arr = np.zeros_like(E)
    sub_factor = (termA*termB/denom_factor)*S2

    with np.errstate(over='ignore', invalid='ignore'):
        T_sub = 1.0/(1.0 + sub_factor)

    T_arr[sub_mask] = T_sub[sub_mask]

    # If E>=U => above barrier => not implemented here; you can set T=1 or do a separate formula
    # above_mask = (E>=U)
    # T_arr[above_mask] = 1.0
    return T_arr


def airy_eval(z, z_switch=1000):
    """
    Evaluate Ai(z), Ai'(z), Bi(z), Bi'(z) with:
      - Standard scipy.special.airy for |z| <= z_switch
      - Asymptotic expansions for |z| > z_switch
    Returns (Ai, Ai', Bi, Bi') all same shape as z.
    
    Parameters
    ----------
    z : array-like
        Real argument(s) where we want Ai, Ai', Bi, Bi'.
    z_switch : float
        Threshold beyond which to switch to asymptotic expansions.
        e.g. 8 <= z_switch <= 15 is typical. Adjust as needed.
    """
    z = np.asarray(z, dtype=float)
    
    # Allocate output
    Ai  = np.zeros_like(z)
    Aip = np.zeros_like(z)
    Bi  = np.zeros_like(z)
    Bip = np.zeros_like(z)
    
    # Masks for each regime:
    mask_mid = (np.abs(z) <= z_switch)
    mask_pos = (z >  z_switch)
    mask_neg = (z < -z_switch)
    # The 'remaining' points in large negative or large positive might also
    # include "just above z_switch" or "just below -z_switch".

    # 1) For moderate |z|: call scipy's airy directly
    if np.any(mask_mid):
        Ai_mid, Aip_mid, Bi_mid, Bip_mid = scipy_airy(z[mask_mid])
        Ai[mask_mid]  = Ai_mid
        Aip[mask_mid] = Aip_mid
        Bi[mask_mid]  = Bi_mid
        Bip[mask_mid] = Bip_mid

    # 2) Large Positive z => Asymptotic expansions
    #    We'll do up to first correction in a short function:
    if np.any(mask_pos):
        zp = z[mask_pos]
        # leading dimensionless param
        zeta = (2.0/3.0)*zp**1.5  # 2/3 * z^(3/2)
        # clampin' time
        zeta_eff = np.minimum(zeta, 700.0)
        
        # Ai ~ prefactor * e^(-zeta) * (1 - 5/(72 zeta)), etc.
        # Ai'(z) ~ - z^(1/4)/(2 sqrt(pi)) e^(-zeta) (1 - 7/(72 zeta))
        sqrt_z = np.sqrt(zp)
        z_quart = zp**0.25        # z^(1/4)
        preA  = 1.0/(2.0*np.sqrt(np.pi)*z_quart)
        preAp = -z_quart/(2.0*np.sqrt(np.pi))
        exp_neg = np.exp(-zeta_eff)
        
        # Correction factors
        corr_Ai  = 1.0 - 5.0/(72.0*zeta_eff)
        corr_Aip = 1.0 - 7.0/(72.0*zeta_eff)
        
        Ai_pos  = preA*exp_neg*corr_Ai
        Aip_pos = preAp*exp_neg*corr_Aip
        
        # Bi ~ 1/(sqrt(pi)*z^(1/4)) e^(+zeta)*(1 + 5/(72 zeta)), etc.
        # Bi'(z) ~ z^(1/4)/sqrt(pi) e^(+zeta)*(1 + 7/(72 zeta))
        preB  = 1.0/(np.sqrt(np.pi)*z_quart)
        preBp =  z_quart/(np.sqrt(np.pi))
        exp_pos = np.exp(zeta_eff)
        
        corr_Bi  = 1.0 + 5.0/(72.0*zeta_eff)
        corr_Bip = 1.0 + 7.0/(72.0*zeta_eff)
        
        Bi_pos  = preB *exp_pos*corr_Bi
        Bip_pos = preBp*exp_pos*corr_Bip
        
        # store
        Ai[mask_pos]  = Ai_pos
        Aip[mask_pos] = Aip_pos
        Bi[mask_pos]  = Bi_pos
        Bip[mask_pos] = Bip_pos

    # 3) Large Negative z => Asymptotic expansions
    #    z = -x, x>0. Let x = -z. We'll do minimal leading terms with single correction.
    if np.any(mask_neg):
        zn = z[mask_neg]
        x  = -zn  # positive
        # param
        xi = (2.0/3.0)*(x**1.5)  # = 2/3 * (|z|)^(3/2)
        
        # Leading approach:
        # Ai(-x) ~ 1/(sqrt(pi)*x^(1/4)) * 1/sqrt(2) * [some combination of sin/cos(xi+pi/4)]
        # but let's keep it simpler:
        # We'll do the standard forms:
        #   Ai(-x) ~ (1/sqrt(pi) x^(1/4)) * 0.5 * [ sin(...) - cos(...)/some factor ]
        # We'll do direct known expansions. 
        # For brevity, let's do the typical leading order:
        #   Ai(-x) ~  1/( sqrt(pi) x^(1/4) ) * (1/sqrt(2)) * sin( xi + pi/4 )  [leading amplitude ~ x^(-1/4)]
        # Then add correction factor ~ 1 - c / x^(3/2). We'll do a simple approach:
        
        x_quart = x**0.25
        ampA    = 1.0/(np.sqrt(np.pi)*x_quart)
        # We define something to handle the small correction
        # But let's do the simplest leading for demonstration:
        # The expansions can be quite complicated if we add the next term. We'll do leading only:
        sinp = np.sin(xi + np.pi/4.0)
        cosp = np.cos(xi + np.pi/4.0)
        
        Ai_neg  = ampA * sinp / np.sqrt(2.0)
        # Bi(-x) ~ ampA * cos( xi + pi/4 ) / sqrt(2)
        Bi_neg  = ampA * cosp / np.sqrt(2.0)
        
        # For derivatives:
        # Ai'(-x) ~ derivative =>  ...
        # leading is ~ - x^(1/4)/sqrt(pi)* ...
        # but let's keep it consistent. We'll do partial derivatives:
        # d/dz [Ai(-x)] = d/dx [Ai(-x)] * dx/dz = ...
        # We'll do a simpler approach or just skip the correction. 
        # Leading derivative amplitude is x^(1/4)/ sqrt(pi) * ...
        
        # derivative factor d/dz => - d/dx
        # Ai'(-x) = - d/dx [Ai(-x)] 
        # If Ai(-x) ~ ampA * sin(...), then derivative ~ ampA' * sin(...) + ...
        # We'll do it quickly:
        
        # Leading amplitude for Ai' is  x^(1/4)/ sqrt(pi)* cos( xi + pi/4 +/- something ) / sqrt(2).
        # We'll do the most direct approach:
        
        # d/dz [Ai(-x)] => - [ d/dx (Ai(-x)) ]
        #    = - [ d(ampA)/dx sin(...) + ampA cos(...)* d/dx( xi + pi/4 ) ]
        # This is somewhat involved. We'll do a leading approach from known references:
        # Ai'(-x) ~ -  x^(1/4)/sqrt(pi)* cos(xi + pi/4) / sqrt(2). 
        # Bi'(-x) ~   x^(1/4)/sqrt(pi)* sin(xi + pi/4) / sqrt(2}.
        
        # Then multiply by chain rule factor d/dz => - d/dx, giving an additional minus. 
        # The net is we want the standard known results. We'll do them directly:
        
        Ai_negp = -x_quart/np.sqrt(np.pi)* cosp / np.sqrt(2.0)
        Bi_negp =  x_quart/np.sqrt(np.pi)* sinp / np.sqrt(2.0)
        
        # store
        Ai[mask_neg]  = Ai_neg
        Aip[mask_neg] = Ai_negp
        Bi[mask_neg]  = Bi_neg
        Bip[mask_neg] = Bi_negp

    return Ai, Aip, Bi, Bip

###############################################################################
# Tunneling with Airy Functions (Vectorized)
###############################################################################
def T_airy(x, D, V):
    """
    Vectorized tunneling with Airy, no clamping.
    x, V => shape (nX,nV,N)
    Returns T => shape (nX,nV,N).
    """
    phi_t_corr = phi_t #+ image_potential(D,V)
    dphi = phi_t_corr - phi_s
    swap_mask = (V + dphi)<0
    eff_phi_t = np.where(swap_mask, phi_s, phi_t)
    eff_phi_s = np.where(swap_mask, phi_t, phi_s)
    eff_V     = np.where(swap_mask, -V, V)

    k1 = a*np.sqrt(x)
    k3_argument = x + eff_V
    k3 = a*np.sqrt(np.maximum(k3_argument, 0.0))

    F = np.abs((eff_V + (eff_phi_t - eff_phi_s))/D)
    eps = 1e-12
    F_safe = np.where(F<eps, eps, F)
    factor = (a / F_safe)**(2.0/3.0)

    Wt         = (E_F + eff_phi_t) - x
    Ws_minus_V = (E_F + eff_phi_s) - x - eff_V

    z0 = factor*Wt
    zs = factor*Ws_minus_V
    zp = -(a**2 * F_safe)**(1.0/3.0)

    Ai_z0, Aip_z0, Bi_z0, Bip_z0 = airy_eval(z0)
    Ai_zs, Aip_zs, Bi_zs, Bip_zs = airy_eval(zs)

    numerator = (k3/k1)*(4.0/(np.pi**2))

    # corrected or updated expression:
    term1 = Aip_z0*Bip_zs - Aip_zs*Bip_z0
    term2 = Ai_z0*Bi_zs   - Ai_zs*Bi_z0
    term3 = Ai_zs*Bip_z0 - Aip_z0*Bi_zs
    term4 = Ai_z0*Bip_zs - Aip_zs*Bi_z0

    term5 = (zp/k1)*term1 + (k3/zp)*term2
    term6 = (k3/k1)*term3 + term4
    denom = term5**2 + term6**2

    T_raw = numerator/denom

    # zero out conduction if x+V<0
    conduction_mask = (x + V)<0
    T_raw[conduction_mask] = 0.0
    return T_raw

def T_airy_or_rect(x_3d, D_val, V_3d):
    """
    Hybrid approach:
      if |F| < 1e-12 => rectangular barrier with height=phi_t
      else => Airy approach
    x_3d, V_3d => shape (nX,nV,N)
    D_val => scalar
    Returns T_3d => shape(nX,nV,N)
    """
    x_3d, V_3d = np.broadcast_arrays(x_3d, V_3d)

    # 1) reverse-bias
    dphi = phi_t - phi_s
    swap_mask = (V_3d + dphi)<0
    eff_phi_t = np.where(swap_mask, phi_s, phi_t)
    eff_phi_s = np.where(swap_mask, phi_t, phi_s)
    eff_V     = np.where(swap_mask, -V_3d, V_3d)

    # 2) field
    F_min = 1e-9
    F = (eff_V + (eff_phi_t - eff_phi_s))/D_val

    T_3d = np.zeros_like(x_3d)

    smallF_mask = (np.abs(F)<F_min)
    normal_mask = ~smallF_mask

    if np.any(smallF_mask):
        # define E = x_sub (the electron normal energy)
        E_sub = x_3d[smallF_mask]
        V_sub = V_3d[smallF_mask]

        T_small = T_rectangular(E_sub, D_val, V_sub, eff_phi_t[smallF_mask])
        T_3d[smallF_mask] = T_small

    if np.any(normal_mask):
        # do airy
        T_norm = T_airy(
            x_3d[normal_mask],
            D_val,
            eff_V[normal_mask]
        )
        T_3d[normal_mask] = T_norm

    return T_3d


def T_RF_vectorized(x_arr, D_val, V_arr, A_val, cos_vals):
    """
    Return the RF-averaged transmission T(x, V) for x in x_arr, 
    for all V in V_arr, with an RF amplitude A_val. shape => (nX,nV).
    """
    V_modulated = V_arr[:, None] + A_val * cos_vals[None, :]
    x_bcast = x_arr[:, None, None]
    V_bcast = V_modulated[None, :, :]

    T_3d = T_airy_or_rect(x_bcast, D_val, V_bcast)
    T_avg = np.mean(T_3d, axis=2)  # shape (nX,nV)
    return T_avg

###############################################################################
# Compute T & Current for One D
###############################################################################
def compute_for_one_D(iD):
    """
    For a single D-value index, build:
      T_array_D => shape (nV,nA,nX)
      current_array_D => shape (nV,nA)
    """
    D_val = D_values[iD]
    print(f"[Worker] Computing for D_index={iD+1}/{nD} (D={D_val:.2f} nm)")

    T_array_D = np.zeros((nV, nA, nX), dtype=float)
    current_array_D = np.zeros((nV, nA), dtype=float)

    for iA, A_val in enumerate(A_values):
        # 1) T_rf_all => shape (nX,nV)
        T_rf_all = T_RF_vectorized(x_values, D_val, V_values, A_val, cos_values)
        T_array_D[:, iA, :] = T_rf_all.T

        # 2) Current => integrate piecewise
        T2d = T_rf_all.T
        x2d = x_values[None, :]
        x_cut = np.clip(E_F - V_values, 0.0, None)

        part1_mask = (x2d <= x_cut[:, None])
        part2_mask = (x2d >= x_cut[:, None]) & (x2d <= E_F)

        integrand_1 = np.where(part1_mask, V_values[:, None] * T2d, 0.0)
        integrand_2 = np.where(part2_mask, (E_F - x2d)*T2d, 0.0)
        integrand = integrand_1 + integrand_2

        current_profile = np.trapz(integrand, x_values, axis=1)
        current_array_D[:, iA] = current_profile

    return iD, T_array_D, current_array_D

###############################################################################
# Parallelized Driver
###############################################################################
def compute_T_and_current_parallel(n_jobs=-1):
    """
    Returns (T_array, current_array) with shapes:
      T_array => (nD,nV,nA,nX)
      current_array => (nD,nV,nA)
    """
    from joblib import Parallel, delayed

    T_array = np.zeros((nD, nV, nA, nX), dtype=float)
    current_array = np.zeros((nD, nV, nA), dtype=float)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_one_D)(iD) for iD in range(nD)
    )

    for (iD, T_array_D, current_array_D) in results:
        T_array[iD] = T_array_D
        current_array[iD] = current_array_D
        print(f"[Main] Completed D_index={iD} (D={D_values[iD]:.2f} nm)")

    return T_array, current_array

###############################################################################
# Savers
###############################################################################
def save_T_array_as_csv(T_array, D_arr, V_arr, A_arr, x_arr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    D_mesh, V_mesh, A_mesh, x_mesh = np.meshgrid(
        D_arr, V_arr, A_arr, x_arr, indexing='ij'
    )
    data = np.column_stack((
        D_mesh.ravel(),
        V_mesh.ravel(),
        A_mesh.ravel(),
        x_mesh.ravel(),
        T_array.ravel()
    ))
    header = "D(nm),V(eV),A(eV),x(eV),T"
    np.savetxt(os.path.join(out_dir, "T_array.csv"),
               data, delimiter=",", header=header, comments="", fmt="%g")

def save_current_as_csv(curr_array, D_arr, V_arr, A_arr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    D_mesh, V_mesh, A_mesh = np.meshgrid(D_arr, V_arr, A_arr, indexing='ij')
    data = np.column_stack((
        D_mesh.ravel(),
        V_mesh.ravel(),
        A_mesh.ravel(),
        curr_array.ravel()
    ))
    header = "D(nm),V(eV),A(eV),current(A)"
    np.savetxt(os.path.join(out_dir, "current_array.csv"),
               data, delimiter=",", header=header, comments="", fmt="%g")

###############################################################################
# Main
###############################################################################
def main():
    start_time = time.perf_counter()
    print("Starting simulation...")

    T_array, current_array = compute_T_and_current_parallel(n_jobs=-1)

    save_dir = "countdown"
    os.makedirs(save_dir, exist_ok=True)

    print("Saving T_array...")
    np.save(os.path.join(save_dir, "T_array.npy"), T_array)
    #save_T_array_as_csv(T_array, D_values, V_values, A_values, x_values, save_dir)

    print("Saving current_array...")
    np.save(os.path.join(save_dir, "current_array.npy"), current_array)
    #save_current_as_csv(current_array, D_values, V_values, A_values, save_dir)

    elapsed = time.perf_counter() - start_time
    print(f"Done.  Total execution time: {elapsed:.2f} s")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # necessary on Windows
    main()
