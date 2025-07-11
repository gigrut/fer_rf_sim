"""fer_plot.py – interactive Plotly helpers for FER simulation data.

Functions
---------
load        : read *current.h5* into a dict.
heat        : 2‑D heat‑map of current vs two axes.
surface     : 3‑D surface plot of current.
scatter3    : 3‑D scatter of current.
trans_map   : time‑averaged transmission vs any two axes (E, V, A) on‑the‑fly.
trans_heat  : convenience wrapper for V × A transmission map.
"""
from pathlib import Path
import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

__all__ = [
    'load', 'heat', 'surface', 'scatter3', 'trans_map', 'trans_heat'
]

# -----------------------------------------------------------------------------
# Basic I/O helpers
# -----------------------------------------------------------------------------

def load(h5file):
    """Return dict with arrays from *current.h5*."""
    with h5py.File(h5file, 'r') as f:
        d = {k: f[k][...] for k in f.keys() if k != 'I'}
        d['I'] = f['I'][...]
        d.update({k: v for k, v in f.attrs.items()})
    return d

def _nearest(arr, val):
    idx = int(np.abs(arr - val).argmin())
    return idx, arr[idx]

# -----------------------------------------------------------------------------
# LUT helpers for transmission
# -----------------------------------------------------------------------------

def _interp_from_lut(folder):
    with h5py.File(Path(folder) / 'transmission_lut.h5', 'r') as f:
        D = f['D'][...]
        z = f['z'][...]
        V = f['V'][...]
        E = f['E'][...]
    return RegularGridInterpolator((z, V, E), D, bounds_error=False, fill_value=0.0), z, V, E

def _cheb_nodes(N):
    n = np.arange(1, N + 1)
    return np.pi * (2 * n - 1) / (2 * N)

# time‑average helpers

def _D_time_avg(interp, Egrid, Vdc, Arf, z_val, nodes):
    """Average transmission over one RF cycle for a single (Vdc, Arf)."""
    V_inst = Vdc + Arf * np.cos(nodes)
    nE, nN = len(Egrid), len(V_inst)
    pts = np.stack([
        np.full(nE * nN, z_val),
        np.tile(V_inst, nE),
        np.repeat(Egrid, nN)
    ], axis=1)
    return interp(pts).reshape(nE, nN).mean(axis=1)  # (nE,)


def _D_grid(interp, Eg, Vg, Ag, z_val, nodes):
    """Vectorised RF‑average for full A×E×V grid.

    Returns array shape (nA, nE, nV).
    """
    nA, nE, nV, nN = len(Ag), len(Eg), len(Vg), len(nodes)

    Vdc_grid = Vg[None, None, :, None]                 # 1 ×1× nV ×1
    Arf_grid = Ag[:, None, None, None]                 # nA×1× 1 ×1
    phase    = np.cos(nodes)[None, None, None, :]      # 1 ×1× 1 ×nN
    V_inst   = Vdc_grid + Arf_grid * phase             # nA×1×nV×nN

    # broadcast energy
    E_grid   = Eg[None, :, None, None]                 # 1 ×nE×1×1
    V_b      = np.broadcast_to(V_inst, (nA, nE, nV, nN))
    E_b      = np.broadcast_to(E_grid, (nA, nE, nV, nN))

    pts = np.column_stack([
        np.full(V_b.size, z_val),          # z
        V_b.ravel(),                       # V
        E_b.ravel()                        # E
    ])
    D = interp(pts).reshape(nA, nE, nV, nN).mean(axis=3)  # avg over phase -> nA×nE×nV
    return D

# -----------------------------------------------------------------------------
# Current plots
# -----------------------------------------------------------------------------

def heat(h5file, x: str, y: str, *, a_val=None, a_idx=None):
    d = load(h5file)
    if a_idx is None:
        a_idx, a_val = _nearest(d['A_rf'], a_val or d['A_rf'][0])
    Z = d['I'][:, :, a_idx]
    fig = px.imshow(Z, x=d[x], y=d[y], origin='lower', aspect='auto',
                    color_continuous_scale='Cividis',
                    labels={'x': x, 'y': y, 'color': 'I'})
    fig.update_layout(title=f'I heat‑map  A_rf≈{a_val:g}')
    fig.show()


def surface(h5file, *, x='V', y='z', a_val=None, a_idx=None):
    d = load(h5file)
    if a_idx is None:
        a_idx, a_val = _nearest(d['A_rf'], a_val or d['A_rf'][0])
    Zmat = d['I'][:, :, a_idx].T
    X, Y = np.meshgrid(d[x], d[y])
    fig = go.Figure(go.Surface(x=X, y=Y, z=Zmat, colorscale='Cividis'))
    fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title='I'),
                      title=f'I surface  A_rf≈{a_val:g}')
    fig.show()


def scatter3(h5file, *, a_val=None, a_idx=None, scale_function='linear', derivative=False, abs_val=False):
    d = load(h5file)
    if a_idx is None:
        a_idx, a_val = _nearest(d['A_rf'], a_val or d['A_rf'][0])
    I = d['I'][:, :, a_idx]   # shape (nz, nV)
    V = d['V']
    z = d['z']
    if derivative:
        if scale_function=='log':
            with np.errstate(divide='ignore', invalid='ignore'):
                logI = np.log(np.abs(I))
            Z = np.abs(np.gradient(logI, V, axis=1)) if abs_val else np.gradient(logI, V, axis=1)
            label = '|dlog(I)/dV|' if abs_val else 'dlog(I)/dV'
        elif scale_function=='asinh':
            with np.errstate(divide='ignore', invalid='ignore'):
                asinhI = np.arcsinh(I)
            Z = np.abs(np.gradient(asinhI, V, axis=1)) if abs_val else np.gradient(asinhI, V, axis=1)
            label = '|dasinh(I)/dV|' if abs_val else 'dasinh(I)/dV'
        else:
            Z = np.abs(np.gradient(I, V, axis=1)) if abs_val else np.gradient(I, V, axis=1)
            label = '|dI/dV|' if abs_val else 'dI/dV'
    else:
        if scale_function=='log':
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = np.abs(np.log(np.abs(I))) if abs_val else np.log(np.abs(I))
            label = '|log(I)|' if abs_val else 'log(I)'
        elif scale_function=='asinh':
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = np.abs(np.arcsinh(I)) if abs_val else np.arcsinh(I)
                print('Took an asinh')
            label = '|asinh(I)|' if abs_val else 'asinh(I)'
        else:
            Z = np.abs(I) if abs_val else I
            label = '|I|' if abs_val else 'I'
    
    Vg, zg = np.meshgrid(V, z)
    fig = go.Figure(
        go.Scatter3d(
            x=Vg.ravel(),
            y=zg.ravel(),
            z=Z.ravel(),
            mode='markers',
            marker=dict(
                size=3,
                color=Z.ravel(),
                colorscale='Cividis',
                colorbar=dict(title=label)
            )
        )
    )
    fig.update_layout(
        title=f"{label} scatter at A_rf ≈ {a_val:g}",
        scene=dict(
            xaxis_title='V (bias)',
            yaxis_title='z (tip height)',
            zaxis_title=label
        )
    )
    fig.show()

def scatter3_slider(h5file, *, a_val=None, a_idx=None, downsample=1,
                    log=False, derivative=False, abs_value=False, arcsinh=False):
    from scipy import stats  # for neat argmin if you like
    d = load(h5file)
    A_rf = d['A_rf'][downsample-1::downsample]
    # pick initial index
    if a_idx is None:
        a_idx, a_val = _nearest(A_rf, a_val or A_rf[0])

    I_all = d['I']                # shape (nz, nV, nA)
    V      = d['V']
    z      = d['z']

    # precompute Z for every A slice
    nz, nV, nA = I_all.shape
    Z_all = np.empty_like(I_all, dtype=float)
    label = None
    for k in range(nA):
        I = I_all[:, :, k]
        # compute base Z
        if derivative:
            if log:
                with np.errstate(divide='ignore', invalid='ignore'):
                    logI = np.log(np.abs(I))
                Z = np.gradient(logI, V, axis=1)
                if k == 0:
                    label = 'dlog(I)/dV'
            else:
                Z = np.gradient(I, V, axis=1)
                if k == 0:
                    label = 'dI/dV'
        else:
            if log:
                with np.errstate(divide='ignore', invalid='ignore'):
                    Z = np.log(np.abs(I))
                if k == 0:
                    label = 'log(I)'
            else:
                Z = I.copy()
                if k == 0:
                    label = 'I'
        # apply abs() or arcsinh() if requested
        if abs_value:
            Z = np.abs(Z)
            if k == 0:
                label = f'|{label}|' if label else '|Z|'
        if arcsinh:
            Z = np.arcsinh(Z)
            if k == 0:
                label = f'arcsinh({label})' if label else 'arcsinh(Z)'
        Z_all[:, :, k] = Z

    # build mesh once
    Vg, zg = np.meshgrid(V, z)

    # build traces, only one visible at a time
    traces = []
    vmin, vmax = np.nanmin(Z_all), np.nanmax(Z_all)
    for k in range(nA):
        traces.append(go.Scatter3d(
            x=Vg.ravel(),
            y=zg.ravel(),
            z=Z_all[:, :, k].ravel(),
            mode='markers',
            marker=dict(
                size=3,
                color=Z_all[:, :, k].ravel(),
                colorscale='Cividis',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=label) if k==0 else None
            ),
            name=f"A_rf = {A_rf[k]:g}",
            visible=(k == a_idx)
        ))

    # slider steps
    steps = []
    for k in range(nA):
        visible = [False]*nA
        visible[k] = True
        steps.append(dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"{label} scatter at A_rf ≈ {A_rf[k]:g}"}],
            label=f"{A_rf[k]:g}"
        ))

    sliders = [dict(
        active=a_idx,
        currentvalue={"prefix": "A_rf = "},
        pad={"t": 50},
        steps=steps
    )]

    fig = go.Figure(data=traces)
    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis_title='V (bias)',
            yaxis_title='z (tip height)',
            zaxis_title=label
        ),
        title=f"{label} scatter at A_rf ≈ {A_rf[a_idx]:g}"
    )
    fig.show()


# -----------------------------------------------------------------------------
# Transmission maps on‑the‑fly
# -----------------------------------------------------------------------------

def trans_map(folder='fer_output', *, z_val=None, z_idx=None,
              x='V', y='A', V_val=None, A_val=None, E_val=None, N=16):
    """Time‑averaged transmission heat‑map with optional energy axis."""
    x = x.upper()[0]; y = y.upper()[0]
    if x == y or x not in 'EVA' or y not in 'EVA':
        raise ValueError('x and y must be two different choices among E,V,A')

    interp, zg, Vg, Eg = _interp_from_lut(folder)
    Ag = load(Path(folder)/'current.h5')['A_rf']

    # z slice
    if z_idx is None:
        z_idx, z_val = _nearest(zg, z_val or zg[0])
    else:
        z_val = zg[z_idx]

    # fixed scalars
    V_fixed = _nearest(Vg, V_val or Vg[0])[1]
    A_fixed = _nearest(Ag, A_val or Ag[0])[1]
    e_idx, E_fixed = _nearest(Eg, E_val or Eg[0])

    axes = {'V': Vg, 'A': Ag, 'E': Eg}
    X, Y = axes[x], axes[y]

    nodes = _cheb_nodes(N)

    if 'E' in (x, y):
        # full grid computation (vectorised)
        D_AEV = _D_grid(interp, Eg, Vg, Ag, z_val, nodes)  # (nA,nE,nV)
        if x == 'E' and y == 'V':      # colour(E,V) @ fixed A
            Z = D_AEV[_nearest(Ag, A_fixed)[0]].T
        elif x == 'V' and y == 'E':    # colour(V,E)
            Z = D_AEV[_nearest(Ag, A_fixed)[0]]
        elif x == 'E' and y == 'A':    # colour(E,A) @ fixed V
            Z = D_AEV[:, :, _nearest(Vg, V_fixed)[0]].T
        else:                          # x=='A', y=='E'
            Z = D_AEV[:, :, _nearest(Vg, V_fixed)[0]]
    else:
        # no energy axis → small nested loop (cheap)
        Z = np.zeros((len(Y), len(X)))
        for j, xv in enumerate(X):
            for i, yv in enumerate(Y):
                Vdc, Arf = V_fixed, A_fixed
                if x == 'V': Vdc = xv
                if y == 'V': Vdc = yv
                if x == 'A': Arf = xv
                if y == 'A': Arf = yv
                Z[i, j] = _D_time_avg(interp, Eg, Vdc, Arf, z_val, nodes).mean()

    fig = px.imshow(Z, x=X, y=Y, origin='lower', aspect='auto',
                    color_continuous_scale='Cividis', labels={'x': x, 'y': y, 'color': '⟨D⟩'})
    fixed_str = f'V={V_fixed:g}' if 'V' not in (x, y) else f'A={A_fixed:g}' if 'A' not in (x, y) else f'E={E_fixed:g}'
    fig.update_layout(title=f'Transmission map  z≈{z_val:g} nm  {fixed_str}')
    fig.show()

def trans_heat(folder='fer_output', *, z_val=None, z_idx=None, N=16):
    """Convenience: V × A transmission map (energy‑averaged across energy axis)."""
    trans_map(folder, z_val=z_val, z_idx=z_idx, x='V', y='A', N=N)
    
def trans_heat_constE(
    folder='fer_output',
    *,
    E_val=None,        # the single electron energy (eV) you want to plot
    z_val=None,        # tip-height (nm) for the slice
    z_idx=None,        # or index if you prefer
    N: int = 16        # # of Chebyshev nodes for the RF average
):
    """
    2-D heatmap of D(E=E_val; V, A) averaged over one RF cycle at fixed tip-height.
    """

    # 1) load data + interpolator + axes
    d = load(f"{folder}/current.h5")
    interp, zg, Vg, Eg = _interp_from_lut(folder)
    Ag = d['A_rf']
    nodes = _cheb_nodes(N)

    # 2) pick E_val
    if E_val is None:
        # default to first grid‐point
        E_val = Eg[0]

    # 3) pick z_val
    if z_idx is None:
        z_idx, z_val = _nearest(zg, z_val or zg[0])
    else:
        z_val = zg[z_idx]

    # 4) build the 2D array Z[A, V]
    Z = np.zeros((len(Ag), len(Vg)), dtype=float)
    for i, Arf in enumerate(Ag):
        for j, Vdc in enumerate(Vg):
            # this returns a length-1 array since we pass [E_val]
            Dpt = _D_time_avg(interp, [E_val], Vdc, Arf, z_val, nodes)
            Z[i, j] = Dpt[0]

    # 5) plot with A on y, V on x
    fig = px.imshow(
        Z,
        x=Vg, y=Ag,
        origin='lower',
        aspect='auto',
        color_continuous_scale='Cividis',
        labels={'x':'V (eV)', 'y':'A_RF (eV)', 'color':f'D(E={E_val:.2f})'}
    )
    fig.update_layout(
        title=f"⟨D⟩ at E={E_val:.2f} eV, z≈{z_val:.2f} nm",
        xaxis_title='V (eV)',
        yaxis_title='A_RF (eV)'
    )
    fig.show()

def trans_heat_constE_slider(
    folder='fer_output',
    *,
    E_val=None,    # single electron energy (eV)
    z_val=None,    # tip height (nm) or...
    z_idx=None,    # index into the z-grid
    N: int = 16,    # # of RF Chebyshev nodes
    ds_A = 1
):
    # 1) load & build interpolator + axes
    d, = [load(f"{folder}/current.h5")]
    interp, zg, Vg, Eg = _interp_from_lut(folder)
    Ag = d['A_rf'][::ds_A]
    nodes = _cheb_nodes(N)

    # 2) pick E_val
    if E_val is None:
        E_val = Eg[0]

    # 3) pick z slice(s)
    if z_idx is None:
        z_idx, z_val = _nearest(zg, z_val or zg[0])
    else:
        z_val = zg[z_idx]

    # we will build one heatmap per z in zg:
    Z_all = np.zeros((len(zg), len(Ag), len(Vg)))
    for iz, z in enumerate(zg):
        for ia, Arf in enumerate(Ag):
            for jv, Vdc in enumerate(Vg):
                Dpt = _D_time_avg(interp, [E_val], Vdc, Arf, z, nodes)
                Z_all[iz, ia, jv] = Dpt[0]

    # 4) build a figure with one trace per z-slice
    traces = []
    for iz, z in enumerate(zg):
        traces.append(go.Heatmap(
            x=Vg,
            y=Ag,
            z=Z_all[iz],
            colorscale='Cividis',
            colorbar=dict(title=f"⟨D⟩ at E={E_val:.2f}eV"),
            zmin=Z_all.min(),
            zmax=Z_all.max(),
            visible=(iz == z_idx)
        ))

    steps = []
    for iz, z in enumerate(zg):
        vis = [False]*len(zg)
        vis[iz] = True
        steps.append(dict(
            method="update",
            args=[{"visible": vis},
                  {"title": f"⟨D⟩ at E={E_val:.2f}eV, z≈{z:.2f}nm"}],
            label=f"{z:.2f}"
        ))

    sliders = [dict(
        active=z_idx,
        pad={"t": 50},
        currentvalue={"prefix": "z = "},
        steps=steps
    )]

    fig = go.Figure(data=traces)
    fig.update_layout(
        sliders=sliders,
        xaxis_title="V (eV)",
        yaxis_title="A_RF (eV)",
        width=700, height=600
    )
    fig.show()

def trans_heat_constV_slider(
    folder='fer_output',
    *,
    V_val=None,   # fixed DC bias (eV)
    V_idx=None,   # or index into V grid
    z_val=None,   # initial tip height (nm)
    z_idx=None,   # or index into z grid
    N: int = 16   # # of RF Chebyshev nodes
):
    # --- 1) load data + interpolator + axes
    d = load(f"{folder}/current.h5")
    interp, zg, Vg, Eg = _interp_from_lut(folder)
    Ag = d['A_rf']                          # RF amplitude axis
    nodes = _cheb_nodes(N)                 # RF phase nodes

    # --- 2) choose V_dc
    if V_idx is None:
        V_idx, V_val = _nearest(Vg, V_val or Vg[0])
    else:
        V_val = Vg[V_idx]

    # --- 3) choose initial z-slice
    if z_idx is None:
        z_idx, z_val = _nearest(zg, z_val or zg[0])
    else:
        z_val = zg[z_idx]

    # --- 4) precompute cosines
    phase = np.cos(nodes)                  # (N,)

    # --- 5) build a (nZ × nA × nE) array of ⟨D⟩ at fixed V_dc
    nZ, nA, nE = len(zg), len(Ag), len(Eg)
    Z_all = np.empty((nZ, nA, nE), dtype=float)

    for iz, z in enumerate(zg):
        # instantaneous voltage for each A and phase: shape (nA, N)
        V_inst = V_val + np.outer(Ag, phase)   # (nA, nN)

        # broadcast into (nA, nE, nN)
        V_b = V_inst[:, None, :]                # → (nA,1,nN) → broadcast to (nA,nE,nN)
        E_b = Eg[None, :, None]                 # → (1,nE,1)    → broadcast to (nA,nE,nN)
        Z_b = np.full((nA, nE, phase.size), z)  # → (nA,nE,nN)

        # pack points for interpolation
        pts = np.column_stack([
            Z_b.ravel(), V_b.ravel(), E_b.ravel()
        ])                                       # (nA*nE*nN, 3)

        # interpolate & average over RF phase
        D_cycle = interp(pts)\
                  .reshape(nA, nE, phase.size)\
                  .mean(axis=2)                # → (nA, nE)

        Z_all[iz] = D_cycle                  # store per‐z slice

    # --- 6) make one trace per z, only one visible at a time
    traces = []
    vmin, vmax = Z_all.min(), Z_all.max()
    for iz, z in enumerate(zg):
        traces.append(go.Heatmap(
            x=Eg,
            y=Ag,
            z=Z_all[iz].T,        # rows=A, cols=E
            colorscale='Cividis',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title=f"⟨D⟩ at V={V_val:.2f} eV"),
            visible=(iz == z_idx)
        ))

    # --- 7) slider steps for z
    steps = []
    for iz, z in enumerate(zg):
        vis = [False]*nZ
        vis[iz] = True
        steps.append(dict(
            method="update",
            args=[{"visible": vis},
                  {"title": f"⟨D⟩ vs E,A at V={V_val:.2f} eV, z={z:.2f} nm"}],
            label=f"{z:.2f}"
        ))

    sliders = [dict(
        active=z_idx,
        pad={"t": 50},
        currentvalue={"prefix": "z = "},
        steps=steps
    )]

    # --- 8) finalize figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        sliders=sliders,
        xaxis_title="E (eV)",
        yaxis_title="A_rf (eV)",
        width=700, height=600
    )
    fig.show()


def plot_transmission_3d_stack(lut, z_lut, V_lut, E_lut, n_z=8, log_mode=False):
    """
    3D “stacked heatmap” of transmission probability D(z,V,E) with a Z-slider.

    Parameters
    ----------
    lut : ndarray, shape (nz, nV, nE)
        Precomputed transmission lookup table: D(z_lut[i], V_lut[j], E_lut[k]).
    z_lut : 1d array, length nz
        The tip-height grid for the LUT.
    V_lut : 1d array, length nV
        The DC bias grid for the LUT.
    E_lut : 1d array, length nE
        The electron-energy grid for the LUT.
    n_z : int
        Number of tip-height slices to display.
    log_mode : bool
        If True, plot log(D) instead of D.

    This function builds an interpolator over (z,V,E), samples n_z values of z,
    and draws each 2D heatmap as a thin surface at that z.  A slider lets you move
    the surface up/down.
    """
    # 1) Interpolator
    interp = RegularGridInterpolator(
        (z_lut, V_lut, E_lut),
        lut,
        bounds_error=False,
        fill_value=0.0
    )

    # 2) sample Z
    Zs = np.linspace(z_lut[0], z_lut[-1], n_z)

    # 3) use full LUT axes for V and E
    Vs = V_lut
    Es = E_lut
    Vg, Eg = np.meshgrid(Vs, Es, indexing='ij')  # shape (nV, nE)

    # 4) precompute all slices and global color bounds
    slices = []
    for z in Zs:
        pts = np.column_stack([
            np.full(Vg.size, z),
            Vg.ravel(),
            Eg.ravel()
        ])
        D = interp(pts).reshape(len(Vs), len(Es))
        if log_mode:
            # avoid log(0)
            D = np.log(np.clip(D, 1e-30, None))
        slices.append(D)
    allD = np.stack(slices)
    vmin, vmax = allD.min(), allD.max()

    # 5) initial surface at Zs[0]
    fig = go.Figure(
        data=[
            go.Surface(
                x=Vg,
                y=Eg,
                z=np.full_like(Vg, Zs[0]),
                surfacecolor=slices[0].T,  # transpose so rows=E, cols=V
                cmin=vmin,
                cmax=vmax,
                colorscale='Cividis',
                colorbar=dict(title='log(D)' if log_mode else 'D'),
                showscale=True
            )
        ]
    )

    # 6) build frames for each z
    frames = []
    for i, z in enumerate(Zs):
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Surface(
                        x=Vg,
                        y=Eg,
                        z=np.full_like(Vg, z),
                        surfacecolor=slices[i].T,
                        cmin=vmin,
                        cmax=vmax,
                        colorscale='Cividis'
                    )
                ]
            )
        )
    fig.frames = frames

    # 7) slider steps
    steps = []
    for i, z in enumerate(Zs):
        steps.append({
            "method": "animate",
            "label": f"{z:.2f} nm",
            "args": [
                [str(i)],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
            ]
        })
    slider = {
        "active": 0,
        "currentvalue": {"prefix": "Tip height Z = "},
        "pad": {"t": 50},
        "steps": steps
    }

    # 8) finalize layout
    fig.update_layout(
        scene=dict(
            xaxis_title="V (bias)",
            yaxis_title="E (electron energy)",
            zaxis_title="Z (tip height)"
        ),
        sliders=[slider],
        title=f"{'log(D)' if log_mode else 'D'} at Z = {Zs[0]:.2f} nm"
    )

    fig.show()



# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print('Begin plotting...')
    
    
    
    # with h5py.File('fer_output/transmission_lut.h5','r') as f:
    #     lut = f['D'][...]
    #     z_lut = f['z'][...]
    #     V_lut = f['V'][...]
    #     E_lut = f['E'][...]
    # with h5py.File('fer_output/current.h5','r') as f:
    #     A_lut = f['A_rf'][...]
    # plot_transmission_3d_stack(
    #     lut, z_lut, V_lut, E_lut,
    #     n_z=5, log_mode=False)
    
    
    # trans_map('fer_output', z_val=1.0, x='E', y='V', A_val=0.0)
    # trans_heat_constE_slider('fer_output', E_val=1, N=16)
    # trans_heat_constE_slider('fer_output', E_val=5, N=16)
    
    
    
    
    # scatter3('fer_output/current.h5', a_val=0.0, scale_function='linear', derivative=False)
    scatter3('fer_output/current.h5', a_val=0.0, scale_function='log', derivative=False)
    # scatter3('fer_output/current.h5', a_val=0.0, scale_function='log', derivative=True)
    # scatter3('fer_output/current.h5', a_val=0.0, scale_function='log', derivative=True, abs_val=True)
    # scatter3('fer_output/current.h5', a_val=0.0, scale_function='asinh', derivative=False)
    # scatter3('fer_output/current.h5', a_val=0.0, scale_function='asinh', derivative=True)
    # scatter3_slider('fer_output/current.h5', a_val=0.0, scale_function='linear', derivative=False)
    # scatter3_slider('fer_output/current.h5', a_val=0.0, scale_function='linear', derivative=False)
    scatter3('fer_output/current.h5', a_val=0.0, scale_function='log', derivative=True)
    
    
    print('Plotting done')
