import numpy as np
from scipy.special import airy as scipy_airy

a = 5.12*0.9             # 1 / (nm * sqrt(eV))
k = a
E_F = 5.5                   # Fermi energy (eV)
phi_t = 4                # Tip work function (eV)
phi_s = 4.5              # Sample work function (eV)

def T_rectangular(E, D, V, U):
    """
    Return 1D rectangular barrier transmission (sub-barrier)
    E  = electron energy (eV)
    D  = barrier thickness (nm)
    V = offset in region 3 (eV)
    U  = barrier height (eV)
    
    We do sub-barrier formula only:
       T = 1 / [1 + ((k2^2 + k1^2)*(k2^2 + k3^2)) / (4*k1^2*k3^2) * sinh^2(|k2|*D)]
    with k1 = a*sqrt(E), k2^2=-a^2(U-E), k3=a*sqrt(E+V).
    
    'a' is the global wavevector constant for your eV->1/nm conversions.
    """

    # ensure arrays
    E     = np.asarray(E, dtype=float)
    V = np.asarray(V, dtype=float)
    U     = np.asarray(U, dtype=float)

    # wavevectors
    k1 = a * np.sqrt(np.maximum(E, 0.0)) 
    # sub barrier => k2^2 = -a^2(U-E), negative => imaginary
    k2sq = -a*a * (U - E)
    k2_abs = np.sqrt(np.maximum(-k2sq, 0.0))
    E3 = E + V
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
    conduction_mask = k3_argument<0
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
    F_min = 1e-12
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


def T_RF(x_arr, D_val, V_arr, A_val, cos_vals):
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







N = 30
n = np.arange(1, N + 1)
angles = np.pi * (2 * n - 1) / (2 * N)
cos_values = np.cos(angles).astype(np.float32)  # shape (N,)
x_arr = np.array([4.0])
volts = np.array([0.8])
D=5.0

ans0 = T_airy(x=x_arr, D=D, V=volts)

ans1 = T_rectangular(E=x_arr, D=D, V=volts, U=phi_t)

ans2 = T_airy_or_rect(x_3d=x_arr, D_val=D, V_3d=volts)

ans3 = T_RF(x_arr, D_val=D, V_arr=volts, A_val=0.0, cos_vals=cos_values)

print('T_airy:')
for t in ans0:
    print(t)
print()
print('T_rectangular:')
for t in ans1:
    print(t)
print()
print('T_airy_or_rect:')
for t in ans2:
    print(t)
print()
print('T_RF:')
for t in ans3:
    for p in t:
        print(p)