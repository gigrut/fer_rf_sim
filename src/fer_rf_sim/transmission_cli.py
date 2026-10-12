"""Import-safe CLI for explicitly selected 2D RF transmission slices."""

import argparse
import sys

import numpy as np

from fer_rf_sim.config import load_config
from fer_rf_sim.transmission_plotting import AXIS_NAMES, requested_transmission_grid
from fer_rf_sim.lut_repository import lut_interpolator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-path", required=True)
    parser.add_argument("--config")
    parser.add_argument("--x", choices=AXIS_NAMES, default="V")
    parser.add_argument("--y", choices=AXIS_NAMES, default="A")
    for name in AXIS_NAMES:
        parser.add_argument("--" + name.lower(), type=float)
    parser.add_argument("--html", help="Write HTML without opening a browser")
    args = parser.parse_args(argv)
    try:
        import plotly.graph_objects as go
        config = load_config(args.config)
        axes = (np.linspace(config.z_min, config.z_max, config.n_Z),
                np.linspace(config.v_min, config.v_max, config.n_V),
                np.linspace(config.A_min, config.A_max, config.n_A),
                np.linspace(config.E_min, config.E_max + config.e_extra, config.n_E))
        fixed = {name: (getattr(args, name.lower()) if getattr(args, name.lower()) is not None else axis[0])
                 for name, axis in zip(AXIS_NAMES, axes) if name not in (args.x, args.y)}
        interpolator, _, _ = lut_interpolator(args.lut_path)
        nodes = np.pi * (2 * np.arange(1, config.n_cheb + 1) - 1) / (2 * config.n_cheb)
        x_values, y_values, values = requested_transmission_grid(
            interpolator, axes, args.x, args.y, fixed, nodes,
        )
        figure = go.Figure(go.Surface(x=x_values, y=y_values, z=values))
        figure.update_layout(scene=dict(xaxis_title=args.x, yaxis_title=args.y, zaxis_title="RF-averaged T"))
        if args.html:
            figure.write_html(args.html)
        else:
            figure.show()
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Transmission plotting failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())