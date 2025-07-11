import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import plotly.graph_objects as go    
import plotly.offline as pyo       
import webbrowser    
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

from FER_constant_current_simulation import (
    E_F, a, build_lut, D_avg
)
from FER_plotter import _D_grid

def plot_transmission_3d_stack(
    interp,    # RegularGridInterpolator over (z,V,E)
    E,        # 1D array of electron energies
    V,        # 1D array of DC biases
    z,        # 1D array of tip heights
    A_fixed,   # single RF amplitude to plot
    nodes,     # Chebyshev nodes for RF averaging
    log_mode=True
    ):
    """
    3D “stacked heatmap” of ⟨D(E,V; z, A_fixed)⟩_RF, with a slider over z.
    """
    nZ, nV, nE = len(z), len(V), len(E)
    E_rel = E - E_F
    Vg, Eg = np.meshgrid(V, E_rel, indexing='ij')   # both shape (nV, nE)

    # 1) precompute every (z,V,E) slice at fixed A
    slices = []
    percs = []
    for zi in z:
        # _D_grid can vectorize RF‐avg for a single A over all (E,V)
        # returns shape (nA=1, nE, nV)
        D_ae_v = _D_grid(interp, E, V, np.asarray([A_fixed]), zi, nodes)
        D_slice = D_ae_v[0].T    # (nV, nE)
        if log_mode:
            D_slice = np.log(np.clip(D_slice, 1e-30, None))
        lo_i, hi_i = np.percentile(D_slice, [50,100])
        percs.append((lo_i, hi_i))
        slices.append(D_slice)

    lo0, hi0 = percs[0]
    fig = go.Figure(
        data=[go.Surface(
            x=Eg, y=Vg,
            z=np.full_like(Eg, z[0]),
            surfacecolor=slices[0],
            cmin=lo0, cmax=hi0,
            colorscale='Plasma',
            colorbar=dict(title='log(D)' if log_mode else 'D'),
            showscale=True
        )]
    )

    # 3) frames for each z‐slice
    frames = []
    for i, zi in enumerate(z):
        lo_i, hi_i = percs[i]
        frames.append(
          go.Frame(
            name=str(i),
            data=[go.Surface(
              x=Eg, y=Vg,
              z=np.full_like(Eg, zi),
              surfacecolor=slices[i],
              cmin=lo_i, cmax=hi_i,
              colorscale='Plasma'
            )]
          )
        )  
    fig.frames = frames

    # 4) slider steps
    steps = []
    for i, zi in enumerate(z):
        vis = [False]*nZ
        vis[i] = True
        steps.append({
            'method': 'animate',
            'label': f'{zi:.2f} nm',
            'args': [
                [str(i)],
                {'mode':'immediate','frame':{'duration':0,'redraw':True}}
            ]
        })

    slider = dict(
        active=0,
        currentvalue={'prefix':'Z = '},
        pad={'t':50},
        x=0.05,      # full‐width slider at bottom
        len=0.9,
        xanchor='left',
        y=0.00,
        yanchor='bottom',
        steps=steps
    )

    # 5) finalize
    fig.update_layout(
        title=f"{'log(D)' if log_mode else 'D'} @ A={A_fixed:.2f}, z={z[0]:.2f} nm",
        scene=dict(
            xaxis_title='E-E_F (eV)',
            yaxis_title='V (bias)',
            zaxis=dict(title='Z (nm)', range=[z[0], z[-1]], autorange=False)
        ),
        sliders=[slider]
    )

    fig.show()


def plot_transmission_EV_vs_A(
    interp,    # RegularGridInterpolator over (z,V,E)
    E,        # 1D array of energies
    V,        # 1D array of DC biases
    A,        # 1D array of RF amplitudes
    z0,        # fixed tip-height (scalar)
    nodes,     # Chebyshev nodes for RF averaging
    log_mode=True
):
    """
    At fixed z=z0, build a V×E heatmap of D_avg(interp; E,V,A,z0) for each A,
    and attach a slider over A to swap through them.
    """
    nA, nV, nE = len(A), len(V), len(E)
    E_rel = E - E_F


    # 1) precompute D_avg for every (A, V) at fixed z0 → shape (nA,nV,nE)
    D = np.zeros((nA, nV, nE), dtype=float)
    for i, Ar in enumerate(A):
        for j, Vdc in enumerate(V):
            D[i, j, :] = D_avg(interp, E, Vdc, Ar, z0, nodes)
    if log_mode:
        D = np.log(np.clip(D, 1e-30, None))

    # 2) global color scale
    vmin, vmax = D.min(), D.max()

    # 3) build one Heatmap trace per A
    traces = []
    for i, Ar in enumerate(A):
        hm = go.Heatmap(
            x=E_rel,
            y=V,
            z=D[i],                # shape (nE,nV) → transposed so y-index maps to rows
            colorscale='Inferno',
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title='log(D)' if log_mode else 'D'),
            visible=(i == 0)
        )
        traces.append(hm)

    # 4) slider steps for A
    steps = []
    for i, Ar in enumerate(A):
        vis = [False]*nA
        vis[i] = True
        steps.append({
            'method': 'update',
            'label': f"A={Ar:.2f}",
            'args': [
                {'visible': vis},
                {'title': f"{'log(D)' if log_mode else 'D'} @ z={z0:.2f} nm, A={Ar:.2f}"}
            ]
        })

    slider = dict(
        active=0,
        currentvalue={'prefix': 'A = '},
        pad={'t': 50},
        steps=steps
    )

    # 5) finalize figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{'log(D)' if log_mode else 'D'} @ z={z0:.2f} nm, A={A[0]:.2f}",
        xaxis=dict(title='E-E_F (eV)'),
        yaxis=dict(title='V (bias)'),
        sliders=[slider]
    )
    fig.show()


def plot_transmission_volume(
    interp, Eg, Vg, zg, Ag, nodes,
    log_mode=True, opacity=1.0, opacityscale='uniform',
    percentiles=[5,95], surface_count=3
    ):
    nA, nZ, nV, nE = len(Ag), len(zg), len(Vg), len(Eg)

    # 1) Generate a mesh of the *correct* shape (nV,nE,nZ)
    #    indexing='xy' means x=Eg varies along axis-1, y=Vg axis-0, z=zg axis-2
    V_mesh, E_mesh, Z_mesh = np.meshgrid(Vg, Eg, zg, indexing='ij')
    x0, y0, z0 = E_mesh.ravel(), V_mesh.ravel(), Z_mesh.ravel()

    # 2) Build D_data in (nA, nZ, nV, nE)
    D_data = np.zeros((nA, nZ, nV, nE), float)
    for i, Ai in enumerate(Ag):
        for k, zi in enumerate(zg):
            D_ae_v = _D_grid(interp, Eg, Vg, np.array([Ai]), zi, nodes)
            D_data[i, k] = D_ae_v[0]   # shape (nE,nV)
    if log_mode:
        D_data = np.log(np.clip(D_data, 1e-30, None))

    # 3) Permute to (nA, nV, nE, nZ) to line up with x0/y0/z0 flattening
    D_data = D_data.transpose(0, 2, 3, 1)  # now shape (nA, nV, nE, nZ)

    # 4) Initial volume at A=Ag[0]
    vals0 = D_data[0].ravel()
    fig = go.Figure(go.Volume(
        x=x0, y=y0, z=z0,
        value=vals0,
        isomin=np.percentile(vals0, percentiles[0]),
        isomax=np.percentile(vals0, percentiles[1]),
        opacity=opacity,
        opacityscale=opacityscale,
        colorscale='Cividis',
        colorbar=dict(title='log(D)' if log_mode else 'D'),
        surface_count=surface_count
        ))

    # 5) Frames, each with a brand-new Volume trace carrying the new values
    frames = []
    for i, Ai in enumerate(Ag):
        frames.append(go.Frame(
            name=str(i),
            data=[go.Volume(
                x=x0, y=y0, z=z0,
                value=D_data[i].ravel(),
                isomin=np.percentile(D_data[i], percentiles[0]),
                isomax=np.percentile(D_data[i], percentiles[1]),
                opacity=opacity,
                opacityscale=opacityscale,
                colorscale='Cividis',
                surface_count=surface_count
            )]
        ))
    fig.frames = frames

    # 6) A-slider
    steps = []
    for i, Ai in enumerate(Ag):
        steps.append(dict(
            method="animate",
            label=f"A={Ai:.2f}",
            args=[
                [str(i)],
                {"mode":"immediate",
                 "frame":{"duration":0,"redraw":True},
                 "transition":{"duration":0}}
            ]
        ))
    slider = dict(
        active=0,
        pad={"t":50},
        currentvalue={"prefix":"A = "},
        steps=steps
    )

    # 7) Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="E (eV)",
            yaxis_title="V (bias)",
            zaxis_title="Z (nm)",
        ),
        sliders=[slider],
        title=f"{'log(D)' if log_mode else 'D'} volume @ A={Ag[0]:.2f}"
    )

    fig.show()


def plot_laplacian_of_transmission(
    interp,       # RegularGridInterpolator over (z, V, E)
    Eg, Vg, zg,   # 1D arrays of E, V, Z
    Ag,           # array of RF amplitudes to step through
    nodes,        # chebyshev nodes for RF averaging
    log_mode=True
    ):
    # ------------------------------------------------------------------------
    # 1) Precompute laplacian for each A → lap4.shape == (nA, nE, nV, nZ)
    # ------------------------------------------------------------------------
    nA, nE, nV, nZ = len(Ag), len(Eg), len(Vg), len(zg)
    lap4 = np.zeros((nA, nE, nV, nZ), float)

    for iA, A in enumerate(Ag):
        # build the D3 grid at this amplitude
        D3 = np.zeros((nE, nV, nZ), float)
        for iz, z in enumerate(zg):
            D_e_v = _D_grid(interp, Eg, Vg, np.array([A]), z, nodes)[0]
            if log_mode:
                D_e_v = np.log(np.clip(D_e_v, 1e-30, None))
            D3[:, :, iz] = D_e_v

        # second derivatives
        d2E = np.gradient(np.gradient(D3, Eg, axis=0), Eg, axis=0)
        d2V = np.gradient(np.gradient(D3, Vg, axis=1), Vg, axis=1)
        d2Z = np.gradient(np.gradient(D3, zg, axis=2), zg, axis=2)

        lap4[iA] = d2E + d2V + d2Z

    # ------------------------------------------------------------------------
    # 2) Flatten grid once
    # ------------------------------------------------------------------------
    Em, Vm, Zm = np.meshgrid(Eg, Vg, zg, indexing='ij')  # (nE,nV,nZ)
    x0, y0, z0 = Em.ravel(), Vm.ravel(), Zm.ravel()

    # ------------------------------------------------------------------------
    # 3) Global display range across all A
    # ------------------------------------------------------------------------
    lo, hi = np.percentile(lap4.ravel(), [30, 90])

    # ------------------------------------------------------------------------
    # 4) Initial Volume trace @ A=Ag[0]
    # ------------------------------------------------------------------------
    initial = go.Volume(
        x=x0, y=y0, z=z0,
        value=lap4[0].ravel(),
        isomin=lo, isomax=hi,
        opacity=0.1,
        opacityscale='extremes',
        surface_count=20,
        colorscale="balance",
        colorbar=dict(title="∇²D")
    )
    fig = go.Figure(data=[initial], frames=[])
    frame_list = list(fig.frames)

    # ------------------------------------------------------------------------
    # 5) Frames: one Volume per amplitude
    # ------------------------------------------------------------------------
    for iA, A in enumerate(Ag):
        frame_list.append(go.Frame(
            name=str(iA),
            data=[go.Volume(
                x=x0, y=y0, z=z0,
                value=lap4[iA].ravel(),
                isomin=lo, isomax=hi,
                opacity=0.1,
                opacityscale='extremes',
                surface_count=10,
                colorscale="balance",
                colorbar=dict(title="∇²D")
            )]
        ))
    fig.frames = frame_list

    # ------------------------------------------------------------------------
    # 6) Slider steps to animate frames
    # ------------------------------------------------------------------------
    steps = []
    for iA, A in enumerate(Ag):
        steps.append(dict(
            method="animate",
            label=f"A={A:.2f}",
            args=[[str(iA)],
                  dict(mode="immediate",
                       frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))]
        ))

    slider = dict(
        active=0,
        pad={"t": 50},
        x=0.1, len=0.8, xanchor="left",
        y=0.00, yanchor="bottom",
        currentvalue={"prefix": "A = "},
        steps=steps
    )

    # ------------------------------------------------------------------------
    # 7) Final layout & write out
    # ------------------------------------------------------------------------
    fig.update_layout(
        title=f"Volume render of ∇²D over A ∈ [{Ag[0]:.2f}, {Ag[-1]:.2f}]",
        scene=dict(
            xaxis_title="E (eV)",
            yaxis_title="V (bias)",
            zaxis_title="Z (nm)"
        ),
        sliders=[slider]
    )

    fig.show()


def plot_isosurfaces_over_A(
    interp,       # RegularGridInterpolator over (z, V, E)
    Eg, Vg, zg,   # 1D arrays of E, V, Z
    Ag,           # array of RF amplitudes to step through
    nodes,        # chebyshev nodes for RF averaging
    n_iso=3,
    opacity=0.8,
    colorscale='Cividis',
    log_mode=True,
    derivative_mode=False
    ):
    # ------------------------------------------------------------------------
    # 1) Precompute D4[iA, iE, iV, iZ]
    # ------------------------------------------------------------------------
    nA, nZ, nV, nE = len(Ag), len(zg), len(Vg), len(Eg)
    D4 = np.zeros((nA, nE, nV, nZ), float)

    for iA, A in enumerate(Ag):
        for iz, z in enumerate(zg):
            # get the (nE, nV) slice at amplitude A and height z
            D_e_v = _D_grid(interp, Eg, Vg, np.array([A]), z, nodes)[0]
            if log_mode:
                D_e_v = np.log(np.clip(D_e_v, 1e-30, None))
                if derivative_mode:
                    D_e_v = np.gradient(D_e_v, Eg, axis=0)
            D4[iA, :, :, iz] = D_e_v   # align E→axis0, V→axis1, Z→axis3

    # ------------------------------------------------------------------------
    # 2) Pick a single iso‐level (or a small band)
    # ------------------------------------------------------------------------
    lo, hi = np.percentile(D4, [30, 90])

    # ------------------------------------------------------------------------
    # 3) Build and flatten the (E,V,Z) grid
    # ------------------------------------------------------------------------
    Em, Vm, Zm = np.meshgrid(Eg, Vg, zg, indexing='ij')  # shapes (nE,nV,nZ)
    x0, y0, z0 = Em.ravel(), Vm.ravel(), Zm.ravel()

    # ------------------------------------------------------------------------
    # 4) Initial figure at Ag[0]
    # ------------------------------------------------------------------------
    label = 'd/dE ' if derivative_mode else ''
    fig = go.Figure(
        go.Isosurface(
            x=x0, y=y0, z=z0,
            value=D4[0].ravel(),
            isomin=lo,
            isomax=hi,
            surface_count=n_iso,
            opacity=opacity,
            colorscale=colorscale,
            colorbar=dict(title=label+"log(D)" if log_mode else "D"),
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
    )

    # ------------------------------------------------------------------------
    # 5) One frame per amplitude
    # ------------------------------------------------------------------------
    frames = []
    for iA, A in enumerate(Ag):
        frames.append(go.Frame(
            name=str(iA),
            data=[go.Isosurface(
                x=x0, y=y0, z=z0,
                value=D4[iA].ravel(),
                isomin=lo,
                isomax=hi,
                surface_count=n_iso,
                opacity=opacity,
                colorscale=colorscale,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )]
        ))
    fig.frames = frames

    # ------------------------------------------------------------------------
    # 6) Slider steps to animate frames
    # ------------------------------------------------------------------------
    steps = []
    for iA, A in enumerate(Ag):
        steps.append(dict(
            method="animate",
            label=f"A={A:.2f}",
            args=[[str(iA)],
                  dict(mode="immediate",
                       frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))]
        ))

    slider = dict(
        active=0,
        pad={"t": 50},
        x=0.1, len=0.8, xanchor="left",
        y=0.0, yanchor="bottom",
        currentvalue={"prefix": "A = "},
        steps=steps
    )

    # ------------------------------------------------------------------------
    # 7) Final layout and write HTML
    # ------------------------------------------------------------------------
    fig.update_layout(
        title=f"{label+'log(D)' if log_mode else 'D'} isosurface over A",
        scene=dict(
            xaxis_title="E (eV)",
            yaxis_title="V (bias)",
            zaxis_title="Z (nm)"
        ),
        sliders=[slider]
    )

    fig.show()


def plot_transmission_and_laplacian(
    interp,       # RegularGridInterpolator over (z, V, E)
    Eg, Vg, zg,   # 1D arrays of E, V, Z
    A,            # single RF amplitude to plot
    nodes,        # chebyshev nodes for RF averaging
    isomin,
    isomax,
    iso_N,
    log_mode=True
    ):
    # ------------------------------------------------------------------------
    # 1) Build the 3D transmission grid D3[E,V,Z] at amplitude A
    # ------------------------------------------------------------------------
    nE, nV, nZ = len(Eg), len(Vg), len(zg)
    D3 = np.zeros((nE, nV, nZ), float)
    for iz, z in enumerate(zg):
        D_e_v = _D_grid(interp, Eg, Vg, np.array([A]), z, nodes)[0]  # (nE,nV)
        if log_mode:
            D_e_v = np.log(np.clip(D_e_v, 1e-30, None))
        D3[:, :, iz] = D_e_v

    # ------------------------------------------------------------------------
    # 2) Compute the Laplacian ∇²D on that grid
    # ------------------------------------------------------------------------
    d2E = np.gradient(np.gradient(D3, Eg, axis=0), Eg, axis=0)
    d2V = np.gradient(np.gradient(D3, Vg, axis=1), Vg, axis=1)
    d2Z = np.gradient(np.gradient(D3, zg, axis=2), zg, axis=2)
    lap = d2E + d2V + d2Z

    # ------------------------------------------------------------------------
    # 3) Flatten the coordinate grid and both fields
    # ------------------------------------------------------------------------
    Em, Vm, Zm = np.meshgrid(Eg, Vg, zg, indexing='ij')  # all shape (nE,nV,nZ)
    x0, y0, z0 = Em.ravel(), Vm.ravel(), Zm.ravel()
    D_flat   = D3.ravel()
    lap_flat = lap.ravel()


    isomin, isomax = np.percentile(D_flat, [isomin, isomax])
    loL, hiL = np.percentile(lap_flat, [5, 95])

    # ------------------------------------------------------------------------
    # 6) Build the combined figure
    # ------------------------------------------------------------------------
    fig = go.Figure()

    # 6a) Transmission isosurface (opaque)
    fig.add_trace(
        go.Isosurface(
            x=x0, y=y0, z=z0,
            value=D_flat,
            isomin=isomin,
            isomax=isomax,
            surface_count=iso_N,
            colorscale="Cividis",
            opacity=1.0,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="D")
        )
    )

    # 6b) Laplacian volume (semi‐transparent)
    fig.add_trace(
        go.Volume(
            x=x0, y=y0, z=z0,
            value=lap_flat,
            isomin=loL, isomax=hiL,
            opacity=0.1,
            opacityscale="uniform",
            surface_count=20,
            colorscale="RdBu",
            colorbar=dict(title="∇²D")
        )
    )

    # ------------------------------------------------------------------------
    # 7) Layout
    # ------------------------------------------------------------------------
    fig.update_layout(
        title=(
            f"A={A:.2f}: D‐isosurface   +  "
            f"Volume of ∇²D ∈ [{loL:.2e},{hiL:.2e}]"
        ),
        scene=dict(
            xaxis_title="E (eV)",
            yaxis_title="V (bias)",
            zaxis_title="Z (nm)"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # ------------------------------------------------------------------------
    # 8) Export standalone HTML and open
    # ------------------------------------------------------------------------
    fig.show()






def make_plots(a):
    # 1) build axes
    E = np.linspace(0.1, E_F + a.e_extra, a.n_E)
    V = np.linspace(a.v_min, a.v_max, a.n_V)      # un-padded for current
    z = np.linspace(a.z_min, a.z_max, a.n_Z)
    A = np.linspace(a.A_min, a.A_max, a.n_A)
    nodes = np.pi*(2*np.arange(1, a.n_cheb+1)-1)/(2*a.n_cheb)

    # 2) padded V for LUT
    dV = (a.v_max - a.v_min)/(a.n_V - 1)
    V_padded = np.linspace(a.v_min - a.A_max,
                         a.v_max + a.A_max,
                         a.n_V + int(np.ceil(2*a.A_max/dV)))

    # 3) load or build LUT
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    lut_file = out / 'transmission_lut_for_3d.h5'
    if lut_file.exists():
        print('[load] LUT')
        with h5py.File(lut_file, 'r') as f:
            lut      = f['D'][...]
            E        = f['E'][...]
            V_padded = f['V'][...]
            z        = f['z'][...]
    else:
        print('[build] LUT')
        lut = build_lut(E, V_padded, z, a.phi_tip, a.phi_samp, upscale=1)
        with h5py.File(lut_file, 'w') as f:
            f.create_dataset('D', data=lut,    compression='gzip')
            f.create_dataset('E', data=E)
            f.create_dataset('V', data=V_padded)
            f.create_dataset('z', data=z)

    # 4) build interpolator over (z,V,E)
    interp = RegularGridInterpolator((z, V_padded, E), lut,
                                     bounds_error=False,
                                     fill_value=0.0)
    
    print("Done building interpolator")

    # 5) finally plot it
    A_fixed = A[0]
    z0 = 3
    # plot_transmission_3d_stack(interp, E, V, z, A_fixed, nodes, log_mode=True)
    # plot_transmission_3d_stack(interp, E, V, z, 1, nodes, log_mode=True)
    # plot_transmission_3d_stack(interp, E, V, z, 2, nodes, log_mode=True)
    # plot_transmission_EV_vs_A(interp, E, V, A, z0, nodes, log_mode=True)
    # plot_transmission_volume(interp, E, V, z, A, nodes, log_mode=True, opacity=0.1)
    # plot_isosurfaces_over_A(
    #     interp,     # your RegularGridInterpolator over (z, V, E)
    #     E, V, z,    # 1D arrays of E, V, Z
    #     A,          # array of RF amplitudes to step through
    #     nodes,      # chebyshev nodes for RF averaging
    #     n_iso=3,    # how many candidate isovalues to choose from
    #     opacity=0.8,
    #     log_mode=True,
    #     derivative_mode=False
    #     )
    # plot_laplacian_of_transmission(
    # interp,       # RegularGridInterpolator over (z, V, E)
    # E, V, z,   # 1D arrays of E, V, Z
    # A,
    # nodes,        # chebyshev nodes for RF averaging
    # log_mode=True
    # )
    plot_transmission_and_laplacian(
    interp,       # RegularGridInterpolator over (z, V, E)
    E, V, z,   # 1D arrays of E, V, Z
    2,            # single RF amplitude to plot
    nodes,        # chebyshev nodes for RF averaging
    isomin=30,
    isomax=90,
    iso_N=3,
    log_mode=True
    )







def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--n_E', type=int, default=50)
    p.add_argument('--n_V', type=int, default=50)
    p.add_argument('--n_Z', type=int, default=50)
    p.add_argument('--n_A', type=int, default=6)
    p.add_argument('--e_extra', type=np.float64, default=0.0)
    p.add_argument('--v_min', type=np.float64, default=2.0)
    p.add_argument('--v_max', type=np.float64, default=10.0)
    p.add_argument('--z_min', type=np.float64, default=1.0)
    p.add_argument('--z_max', type=np.float64, default=3.0)
    p.add_argument('--A_min', type=np.float64, default=0.0)
    p.add_argument('--A_max', type=np.float64, default=5.0)
    p.add_argument('--phi_tip', type=np.float64, default=4.0)
    p.add_argument('--phi_samp', type=np.float64, default=4.7)
    p.add_argument('--n_cheb', type=int, default=16)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--out', type=str, default='fer_output')
    args = p.parse_args()

    t0 = time.perf_counter()
    make_plots(args)
    print(f"[done] runtime {time.perf_counter()-t0:.1f}s")





if __name__ == '__main__':
    print("Begin plotting…")
    cli()
    print("Plotting done.")
