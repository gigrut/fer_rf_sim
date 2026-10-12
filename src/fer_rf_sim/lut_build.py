import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from .physics import E_F, transmission


def _upscaled_axis(values, factor):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("LUT axes must be one-dimensional with at least two points")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("LUT upscale factors must be positive integers")
    return np.linspace(values[0], values[-1], (values.size - 1) * factor + 1)


def build_lut(Eg, Vg, zg, phi_t, phi_s, upscale=1, n_jobs=-1,
              progress_callback=None, cancel_event=None, fermi_tip=E_F, fermi_samp=None):
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
    Vmesh, Emesh = np.meshgrid(V_lut, E_lut, indexing='ij')

    def _calc_z(i, zval):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Simulation cancelled by user")
        print(f"  [LUT] {i+1}/{len(z_lut)} z={zval:.2f} nm")
        D = np.empty((len(V_lut), len(E_lut)), dtype=np.float64)
        for start in range(0, len(V_lut), 64):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Simulation cancelled by user")
            stop = min(start + 64, len(V_lut))
            D[start:stop] = transmission(
                Emesh[start:stop], Vmesh[start:stop], zval, phi_t, phi_s,
                fermi_tip=fermi_tip, fermi_samp=fermi_samp,
            )
        return i, D

    parallel_options = {"n_jobs": n_jobs}
    if cancel_event is not None:
        parallel_options.update(prefer="threads", return_as="generator_unordered")
    results = Parallel(**parallel_options)(  # type: ignore
        delayed(_calc_z)(i, z) for i, z in enumerate(z_lut)
    )
    for completed, (idx, D_slice) in enumerate(results, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Simulation cancelled by user")
        lut[idx] = D_slice
        if progress_callback is not None:
            progress_callback(completed, len(z_lut))

    return lut


def list_existing_luts(fer_output_dir='fer_output'):
    """List existing LUT files and their parameters."""
    from .lut_cli import list_luts
    return list_luts(fer_output_dir)


def interp_lut(lut, Eg, Vg, zg):
    return RegularGridInterpolator((zg, Vg, Eg), lut, bounds_error=True)
