from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
from joblib import Parallel, delayed
from .physics import E_F, F_SWITCH, F_BLEND, transmission, calibrate_fudge
from .quadrature import current_batch, cosine_nodes
from .lut_build import _upscaled_axis, build_lut, interp_lut


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Simulation cancelled by user")


def direct_rf_current_grid(Eg, Vg, Ag, z, phi_tip, phi_samp,
                           fudge, cos_nodes, amplitude_chunk=8, cancel_event=None):
    """Compute one height's RF current grid with batched direct transmission."""
    def evaluate(energy, voltage):
        return transmission(energy, voltage, z, phi_tip, phi_samp, fudge=fudge)
    return _rf_current_grid(evaluate, Eg, Vg, Ag, cos_nodes, amplitude_chunk, cancel_event)


def lut_rf_current_grid(interpolator, Eg, Vg, Ag, z, cos_nodes, amplitude_chunk=8,
                        cancel_event=None):
    """Use the same batched current/phase integration as the direct path."""
    evaluate = _lut_evaluator(interpolator, z)
    return _rf_current_grid(evaluate, Eg, Vg, Ag, cos_nodes, amplitude_chunk, cancel_event)


def _lut_evaluator(interpolator, z):
    from scipy.interpolate import RegularGridInterpolator

    aligned = None
    if (isinstance(interpolator, RegularGridInterpolator)
            and interpolator.method == "linear" and interpolator.bounds_error
            and len(interpolator.grid) == 3 and interpolator.values.ndim == 3):
        heights, voltages, energies = interpolator.grid
        height_index = np.searchsorted(heights, z)
        if height_index < heights.size and heights[height_index] == z and voltages.size > 1:
            aligned = voltages, energies, interpolator.values[height_index]

    def evaluate(energy, voltage):
        if aligned is not None:
            axis, energies, values = aligned
            requested_energy = energy.ravel()
            energy_indices = np.searchsorted(energies, requested_energy)
            if (np.all(energy_indices < energies.size)
                    and np.array_equal(energies[energy_indices], requested_energy)):
                spectra = values[:, energy_indices]
                queries = voltage[..., 0]
                if (not np.isfinite(queries).all() or np.any(queries < axis[0])
                        or np.any(queries > axis[-1])):
                    raise ValueError("One of the requested xi is out of bounds in dimension 1")
                lower = np.clip(np.searchsorted(axis, queries, side="right") - 1,
                                0, axis.size - 2)
                fraction = ((queries - axis[lower]) / (axis[lower + 1] - axis[lower]))[..., None]
                return spectra[lower] * (1 - fraction) + spectra[lower + 1] * fraction
        energy, voltage = np.broadcast_arrays(energy, voltage)
        points = np.column_stack((np.full(energy.size, z), voltage.ravel(), energy.ravel()))
        return interpolator(points).reshape(energy.shape)
    return evaluate


def _rf_grid_inputs(Eg, Vg, Ag, cos_nodes, amplitude_chunk):
    from .repository import validate_axis
    Eg = validate_axis(Eg, "E", minimum=2)
    Vg = validate_axis(Vg, "V")
    Ag = validate_axis(Ag, "A")
    cos_nodes = np.asarray(cos_nodes, dtype=np.float64)
    if (isinstance(amplitude_chunk, bool) or not isinstance(amplitude_chunk, (int, np.integer))
            or amplitude_chunk < 1):
        raise ValueError("amplitude_chunk must be a positive integer")
    if (cos_nodes.ndim != 1 or not cos_nodes.size or not np.isfinite(cos_nodes).all()
            or np.any(np.abs(cos_nodes) > 1)):
        raise ValueError("cos_nodes must be a finite nonempty array in [-1, 1]")

    Eg = Eg[(Eg >= 0) & (Eg <= E_F)]
    return Eg, Vg, Ag, cos_nodes


def _rf_current_grid(evaluate, Eg, Vg, Ag, cos_nodes, amplitude_chunk, cancel_event=None):
    Eg, Vg, Ag, cos_nodes = _rf_grid_inputs(Eg, Vg, Ag, cos_nodes, amplitude_chunk)
    _raise_if_cancelled(cancel_event)
    result = np.zeros((Vg.size, Ag.size), dtype=np.float64)
    if Eg.size < 2:
        return result
    zero_rf = Ag == 0
    if np.any(zero_rf):
        spectra = evaluate(Eg[None, :], Vg[:, None])
        result[:, zero_rf] = current_batch(spectra, Eg, Vg)[:, None]

    nonzero_rf = np.flatnonzero(~zero_rf)
    energy_grid = Eg[None, None, None, :]
    for start in range(0, nonzero_rf.size, amplitude_chunk):
        _raise_if_cancelled(cancel_event)
        indices = nonzero_rf[start:start + amplitude_chunk]
        voltages = Vg[:, None, None, None] + (
            Ag[None, indices, None, None] * cos_nodes[None, None, :, None]
        )
        transmissions = evaluate(energy_grid, voltages)
        phase_voltages = voltages[..., 0]
        phase_currents = current_batch(
            transmissions, Eg, phase_voltages,
        )
        result[:, indices] = np.mean(phase_currents, axis=-1)
    return result


def dc_table_rf_current_grid(evaluate, Eg, Vg, Ag, cos_nodes, *, initial_points=129,
                             max_points=16385, rtol=1e-3, atol=1e-15, amplitude_chunk=8,
                             cancel_event=None):
    """RF average of an adaptive DC-current table with sampled interpolation checks.

    Check each voltage interval at its quarter, midpoint and three-quarter points.
    These are interpolation checks, not an error bound for arbitrary resonances,
    cancellation near zero current, derivatives, or peak positions.
    """
    from numbers import Real

    Eg, Vg, Ag, cos_nodes = _rf_grid_inputs(Eg, Vg, Ag, cos_nodes, amplitude_chunk)
    if Ag[0] < 0:
        raise ValueError("DC table RF amplitudes must be non-negative")
    for name, value in (("initial_points", initial_points), ("max_points", max_points)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer of at least 2")
    if max_points < initial_points:
        raise ValueError("max_points must be at least initial_points")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not np.isfinite(value) or value < 0 for value in (rtol, atol)) or (rtol == 0 and atol == 0):
        raise ValueError("DC table tolerances must be finite, non-negative and not both zero")

    diagnostics = {"points": 0, "validation_points": 0,
                   "max_absolute_error": 0.0, "max_scaled_error": 0.0}
    result = np.zeros((Vg.size, Ag.size), dtype=np.float64)
    if Eg.size < 2:
        return result, diagnostics
    if np.all(Ag == 0):
        spectra = evaluate(Eg[None, :], Vg[:, None])
        result[:] = current_batch(spectra, Eg, Vg)[:, None]
        diagnostics["points"] = int(Vg.size)
        return result, diagnostics

    def evaluate_currents(voltages):
        currents = np.empty(voltages.size, dtype=np.float64)
        for start in range(0, voltages.size, 256):
            _raise_if_cancelled(cancel_event)
            stop = min(start + 256, voltages.size)
            bias = voltages[start:stop]
            spectra = evaluate(Eg[None, :], bias[:, None])
            currents[start:stop] = current_batch(spectra, Eg, bias)
        return currents

    voltages = np.linspace(Vg[0] - Ag[-1], Vg[-1] + Ag[-1], initial_points)
    currents = evaluate_currents(voltages)
    fractions = np.array([0.25, 0.5, 0.75])
    pending = np.ones(voltages.size - 1, dtype=bool)
    interval_absolute_error = np.zeros(voltages.size - 1)
    interval_scaled_error = np.zeros(voltages.size - 1)
    while True:
        _raise_if_cancelled(cancel_event)
        positions = np.flatnonzero(pending)
        probes = voltages[positions, None] + np.diff(voltages)[positions, None] * fractions
        checked = evaluate_currents(probes.ravel()).reshape(probes.shape)
        estimated = currents[positions, None] * (1 - fractions) + currents[positions + 1, None] * fractions
        errors = np.abs(checked - estimated)
        tolerance = atol + rtol * np.abs(checked)
        scaled = np.divide(errors, tolerance, out=np.full_like(errors, np.inf), where=tolerance > 0)
        scaled[errors == 0] = 0
        interval_absolute_error[positions] = errors.max(axis=1)
        interval_scaled_error[positions] = scaled.max(axis=1)
        diagnostics["validation_points"] += int(probes.size)
        diagnostics["max_absolute_error"] = float(interval_absolute_error.max())
        diagnostics["max_scaled_error"] = float(interval_scaled_error.max())
        rejected = np.any(errors > tolerance, axis=1)
        refine = np.zeros(voltages.size - 1, dtype=bool)
        refine[positions] = rejected
        if not np.any(refine):
            break
        if voltages.size + 3 * np.count_nonzero(refine) > max_points:
            raise RuntimeError(
                f"DC-current table did not converge within {max_points} points; "
                "increase dc_table_max_points, relax tolerances, or disable dc_table"
            )
        refined_voltages = np.concatenate((voltages, probes[rejected].ravel()))
        refined_currents = np.concatenate((currents, checked[rejected].ravel()))
        order = np.argsort(refined_voltages)
        voltages = refined_voltages[order]
        currents = refined_currents[order]
        if np.any(np.diff(voltages) <= 0):
            raise RuntimeError("DC-current table refinement reached floating-point voltage resolution")
        children = np.where(refine, 4, 1)
        pending = np.repeat(refine, children)
        interval_absolute_error = np.repeat(interval_absolute_error, children)
        interval_scaled_error = np.repeat(interval_scaled_error, children)

    diagnostics["points"] = int(voltages.size)
    zero_rf = Ag == 0
    if np.any(zero_rf):
        result[:, zero_rf] = evaluate_currents(Vg)[:, None]
    indices = np.flatnonzero(~zero_rf)
    for start in range(0, indices.size, amplitude_chunk):
        _raise_if_cancelled(cancel_event)
        chunk = indices[start:start + amplitude_chunk]
        phase_voltages = Vg[:, None, None] + Ag[None, chunk, None] * cos_nodes
        result[:, chunk] = np.interp(phase_voltages, voltages, currents).mean(axis=-1)
    return result, diagnostics


def _work_function_filename(value):
    text = str(float(value))
    return text if "." in text or "e" in text.lower() else text + ".0"


def simulate(a, progress_callback=None, cancel_event=None):
    from fer_rf_sim.config import SimulationConfig
    from fer_rf_sim.provenance import (atomic_h5, canonical_json, numerical_identity, lut_spec,
                                     SCHEMA_VERSION, MODEL_VERSION, CALIBRATION_VERSION,
                                     CALIBRATION_SETTINGS)
    from fer_rf_sim.lut_repository import lut_matches
    from uuid import uuid4
    a = SimulationConfig.from_namespace(a).as_namespace()
    _raise_if_cancelled(cancel_event)
    Eg = np.linspace(a.E_min, a.E_max + a.e_extra, a.n_E)
    Vg = np.linspace(a.v_min, a.v_max, a.n_V)
    zg = np.linspace(a.z_min, a.z_max, a.n_Z)
    Ag = np.linspace(a.A_min, a.A_max, a.n_A)
    cos_nodes = cosine_nodes(a.n_cheb)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    
    dV = (a.v_max-a.v_min)/(a.n_V - 1)
    Vg_lut = np.linspace(a.v_min-a.A_max, a.v_max+a.A_max, a.n_V + int(np.ceil(2*a.A_max/dV)))

    fudge, samples = calibrate_fudge(
        phi_t=a.phi_tip, phi_s=a.phi_samp, F1=F_SWITCH,
        D_min=a.z_min, D_max=a.z_max, **CALIBRATION_SETTINGS
    )
    print(f'Calibration: factor={fudge:.8g}, valid_samples={samples.size}')

    lut_file = None
    lut_identity = None
    lut_cache_status = "disabled"
    table_diagnostics = []
    table_options = {"initial_points": a.dc_table_points, "max_points": a.dc_table_max_points,
                     "rtol": a.dc_table_rtol, "atol": a.dc_table_atol,
                     "amplitude_chunk": a.amplitude_chunk}
    if getattr(a, 'use_lut', False):
        # Determine upscaling factors first
        upscale_z = getattr(a, 'upscale_z', None) or a.upscale
        upscale_v = getattr(a, 'upscale_v', None) or a.upscale  
        upscale_e = getattr(a, 'upscale_e', None) or a.upscale
        upscale_factors = (upscale_z, upscale_v, upscale_e)
        z_lut = _upscaled_axis(zg, upscale_z)
        V_lut = _upscaled_axis(Vg_lut, upscale_v)
        E_lut = _upscaled_axis(Eg, upscale_e)
        
        # Create parameter-based LUT filename
        spec = lut_spec(E_lut, V_lut, z_lut, a.phi_tip, a.phi_samp, fudge, E_F, F_SWITCH, F_BLEND)
        lut_identity = numerical_identity(spec)
        lut_filename = f"transmission_lut_{lut_identity}.h5"
        lut_file = out / lut_filename
        rebuild_lut = True
        if lut_file.exists() and not getattr(a, 'force_rebuild_lut', False):
            with h5py.File(lut_file, 'r') as f:
                if lut_matches(f, spec):
                    print('[load] LUT')
                    lut = f['D'][...]
                    rebuild_lut = False
                else:
                    print('[rebuild] LUT: grid or parameters changed.')
        elif getattr(a, 'force_rebuild_lut', False):
            print('[rebuild] LUT: forced rebuild requested.')
        
        if rebuild_lut:
            print(f'[build] LUT with upscaling factors: z={upscale_z}, v={upscale_v}, e={upscale_e}')
            lut = build_lut(
                Eg, Vg_lut, zg, a.phi_tip, a.phi_samp,
                upscale=upscale_factors, fudge=fudge, n_jobs=a.threads,
                progress_callback=(
                    lambda completed, total: progress_callback("LUT", completed, total)
                ) if progress_callback is not None else None,
                cancel_event=cancel_event,
            )
            with atomic_h5(lut_file) as f:
                f.create_dataset('D', data=lut, compression='gzip')
                f.create_dataset('E', data=E_lut)
                f.create_dataset('V', data=V_lut)
                f.create_dataset('z', data=z_lut)
                f.attrs['phi_tip'] = a.phi_tip
                f.attrs['phi_samp'] = a.phi_samp
                f.attrs['F_switch'] = F_SWITCH
                f.attrs['F_blend'] = F_BLEND
                f.attrs['lut_identity'] = lut_identity
                f.attrs['lut_spec_json'] = canonical_json(spec)
                f.attrs['schema_version'] = SCHEMA_VERSION
                f.attrs['model_version'] = MODEL_VERSION
                f.attrs['calibration_version'] = CALIBRATION_VERSION
                f.attrs['fudge_factor'] = fudge
                f['D'].attrs['axis_order'] = 'z,V,E'
                for name, unit in (('z', 'nm'), ('V', 'V'), ('E', 'eV'), ('D', 'probability')):
                    f[name].attrs['units'] = unit
        interp = interp_lut(lut, E_lut, V_lut, z_lut)
        lut_cache_status = "built" if rebuild_lut else "reused"

        I = np.zeros((len(zg), len(Vg), len(Ag)))
        def _row(i,z):
            _raise_if_cancelled(cancel_event)
            print(f"  [I] z {i+1}/{len(zg)}")
            diagnostics = None
            if a.dc_table:
                r, diagnostics = dc_table_rf_current_grid(
                    _lut_evaluator(interp, z), Eg, Vg, Ag, cos_nodes,
                    cancel_event=cancel_event, **table_options,
                )
            else:
                r = lut_rf_current_grid(
                    interp, Eg, Vg, Ag, z, cos_nodes, a.amplitude_chunk, cancel_event,
                )
            return i, r, diagnostics
        
        parallel_options = {"n_jobs": a.threads}
        if cancel_event is not None:
            parallel_options.update(prefer="threads", return_as="generator_unordered")
        res = Parallel(**parallel_options)(delayed(_row)(i,z) for i,z in enumerate(zg))  # type: ignore
        for completed, (i, r, diagnostics) in enumerate(res, 1):
            _raise_if_cancelled(cancel_event)
            I[i] = r
            if diagnostics is not None:
                table_diagnostics.append({"z_nm": float(zg[i]), **diagnostics})
            if progress_callback is not None:
                progress_callback("Current grid", completed, len(zg))
    else:
        I = np.zeros((len(zg), len(Vg), len(Ag)))
        def _row(i,z):
            _raise_if_cancelled(cancel_event)
            print(f"  [I] z {i+1}/{len(zg)} (direct)")
            diagnostics = None
            if a.dc_table:
                def evaluate(energy, voltage):
                    return transmission(energy, voltage, z, a.phi_tip, a.phi_samp, fudge=fudge)
                r, diagnostics = dc_table_rf_current_grid(
                    evaluate, Eg, Vg, Ag, cos_nodes, cancel_event=cancel_event,
                    **table_options,
                )
            else:
                r = direct_rf_current_grid(
                    Eg, Vg, Ag, z, a.phi_tip, a.phi_samp,
                    fudge, cos_nodes, a.amplitude_chunk, cancel_event,
                )
            return i, r, diagnostics
        parallel_options = {"n_jobs": a.threads}
        if cancel_event is not None:
            parallel_options.update(prefer="threads", return_as="generator_unordered")
        res = Parallel(**parallel_options)(delayed(_row)(i,z) for i,z in enumerate(zg))  # type: ignore
        for completed, (i, r, diagnostics) in enumerate(res, 1):
            _raise_if_cancelled(cancel_event)
            I[i] = r
            if diagnostics is not None:
                table_diagnostics.append({"z_nm": float(zg[i]), **diagnostics})
            if progress_callback is not None:
                progress_callback("Current grid", completed, len(zg))

    # Create parameterized filename for current results
    _raise_if_cancelled(cancel_event)
    effective_config = vars(a)
    identity_config = {key: value for key, value in effective_config.items()
                       if key not in ('out', 'threads', 'force_rebuild_lut', 'list_luts')}
    config_identity = numerical_identity({'config': identity_config, 'model': MODEL_VERSION,
                                         'calibration': CALIBRATION_VERSION})
    run_id = uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_filename = (
        f"current_phit{_work_function_filename(a.phi_tip)}"
        f"_phis{_work_function_filename(a.phi_samp)}_nE{a.n_E}_nV{a.n_V}"
        f"_nZ{a.n_Z}_nA{a.n_A}_{timestamp}.h5"
    )
    current_file = out / current_filename
    
    with atomic_h5(current_file) as f:
        # Main data
        f.create_dataset('I', data=I, compression='gzip')
        f.create_dataset('z', data=zg)
        f.create_dataset('V', data=Vg)
        f.create_dataset('A_rf', data=Ag)
        f.create_dataset('E', data=Eg)
        for name, unit in (('z', 'nm'), ('V', 'V'), ('A_rf', 'V_peak'), ('E', 'eV'), ('I', 'normalized')):
            f[name].attrs['units'] = unit
        f['I'].attrs['axis_order'] = 'z,V,A'
        f.attrs['schema_version'] = SCHEMA_VERSION
        f.attrs['model_version'] = MODEL_VERSION
        f.attrs['calibration_version'] = CALIBRATION_VERSION
        f.attrs['calibration_settings_json'] = canonical_json(CALIBRATION_SETTINGS)
        f.attrs['effective_config_json'] = canonical_json(effective_config)
        f.attrs['config_identity'] = config_identity
        f.attrs['run_id'] = run_id
        f.attrs['lut_file'] = lut_file.name if lut_file is not None else ''
        f.attrs['lut_identity'] = lut_identity or ''
        f.attrs['lut_cache_status'] = lut_cache_status
        if a.dc_table:
            f.attrs['dc_table_diagnostics_json'] = canonical_json(table_diagnostics)
        for name, value in effective_config.items():
            if value is not None:
                f.attrs[name] = str(value) if isinstance(value, Path) else value
        
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
        f.attrs['F_switch'] = F_SWITCH
        f.attrs['F_blend'] = F_BLEND
        f.attrs['current_units'] = 'normalized'
        f.attrs['current_prefactor'] = 1.0
        
        # Grid information
        f.attrs['dV'] = dV
        f.attrs['n_V_lut'] = len(Vg_lut)
        
    print(f'[save] {current_file} written with parameters')
    return current_file
