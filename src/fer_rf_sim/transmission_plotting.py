"""RF transmission arrays use (z, V, A, E); surfaces use (y, x)."""

import numpy as np

from fer_rf_sim.repository import validate_axis


AXIS_NAMES = ("Z", "V", "A", "E")


def simulation_transmission_source(data):
    """Use the artifact's linked LUT, or recompute with the active direct model."""
    from .lut_repository import lut_interpolator
    from .physics import E_F, transmission
    from .provenance import MODEL_VERSION

    linkage = data.metadata.get("lut_file")
    if linkage:
        interpolator, _, metadata = lut_interpolator(data.path.parent / linkage)
        expected = data.metadata.get("lut_identity")
        if expected and expected != metadata.get("lut_identity"):
            raise ValueError("Linked LUT identity differs from the simulation artifact")
        return interpolator, "Linked simulation LUT"

    try:
        phi_tip = float(data.metadata["phi_tip"])
        phi_samp = float(data.metadata["phi_samp"])
        fermi_tip = float(data.metadata.get("E_F_tip", data.metadata.get("E_F", E_F)))
        fermi_samp = float(data.metadata.get("E_F_samp", fermi_tip))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Tunneling probability requires phi_tip and phi_samp metadata") from error
    if not np.isfinite([phi_tip, phi_samp]).all() or min(phi_tip, phi_samp) <= 0:
        raise ValueError("Work-function metadata must be finite and positive")
    if not np.isfinite([fermi_tip, fermi_samp]).all() or min(fermi_tip, fermi_samp) <= 0:
        raise ValueError("Fermi-energy metadata must be finite and positive")

    def evaluate(points):
        return transmission(points[:, 2], points[:, 1], points[:, 0], phi_tip, phi_samp,
                            fermi_tip=fermi_tip, fermi_samp=fermi_samp)

    return evaluate, f"Recomputed with active model {MODEL_VERSION} (not stored transmission)"


def rf_transmission_grid(interpolator, heights, voltages, amplitudes, energies, nodes):
    axes = [validate_axis(values, name) for values, name in
            zip((heights, voltages, amplitudes, energies), AXIS_NAMES)]
    heights, voltages, amplitudes, energies = axes
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or not nodes.size or not np.isfinite(nodes).all():
        raise ValueError("RF phase nodes must be a nonempty finite 1D array")
    voltage_mesh, amplitude_mesh, energy_mesh, phase_mesh = np.meshgrid(
        voltages, amplitudes, energies, nodes, indexing="ij",
    )
    instantaneous = voltage_mesh + amplitude_mesh * np.cos(phase_mesh)
    result = np.empty(tuple(axis.size for axis in axes))
    for height_index, height in enumerate(heights):
        points = np.column_stack((np.full(instantaneous.size, height),
                                  instantaneous.ravel(), energy_mesh.ravel()))
        result[height_index] = interpolator(points).reshape(instantaneous.shape).mean(axis=-1)
    return result


def transmission_slice(values, axes, x_axis, y_axis, fixed):
    names = tuple("E" if name == "x" else name.upper() for name in (x_axis, y_axis))
    fixed = {("E" if name == "x" else name.upper()): value for name, value in fixed.items()}
    if len(set(names)) != 2 or not set(names) <= set(AXIS_NAMES):
        raise ValueError("Choose two different transmission axes from Z, V, A, E")
    if set(fixed) != set(AXIS_NAMES) - set(names):
        raise ValueError("A 2D transmission surface requires exactly two fixed axes")
    axis_map = {name: validate_axis(axis, name) for name, axis in zip(AXIS_NAMES, axes)}
    if np.shape(values) != tuple(axis.size for axis in axis_map.values()):
        raise ValueError("Transmission shape must match (z, V, A, E)")
    selectors = []
    for name, axis in axis_map.items():
        if name not in fixed:
            selectors.append(slice(None))
        else:
            value = fixed[name]
            if not np.isfinite(value) or not axis[0] <= value <= axis[-1]:
                raise ValueError(f"Fixed {name} is outside the available grid")
            selectors.append(int(np.argmin(np.abs(axis - value))))
    selected = np.asarray(values)[tuple(selectors)]
    remaining = [name for name in AXIS_NAMES if name not in fixed]
    selected = selected.transpose(remaining.index(names[1]), remaining.index(names[0]))
    return axis_map[names[0]], axis_map[names[1]], selected


def requested_transmission_grid(interpolator, axes, x_axis, y_axis, fixed, nodes):
    """Reduce unused axes before evaluating transmission, not after allocating it."""
    selected_axes = []
    if set(fixed) != set(AXIS_NAMES) - {x_axis, y_axis}:
        raise ValueError("A 2D transmission surface requires exactly two fixed axes")
    for name, values in zip(AXIS_NAMES, axes):
        values = validate_axis(values, name)
        if name in fixed:
            value = fixed[name]
            if not np.isfinite(value) or not values[0] <= value <= values[-1]:
                raise ValueError(f"Fixed {name} is outside the available grid")
            values = values[[np.argmin(np.abs(values - value))]]
        selected_axes.append(values)
    values = rf_transmission_grid(interpolator, *selected_axes, nodes)
    snapped = {name: axis[0] for name, axis in zip(AXIS_NAMES, selected_axes) if name in fixed}
    return transmission_slice(values, selected_axes, x_axis, y_axis, snapped)