# Starter STM energy diagram (matplotlib, fully editable)
# - Two rectangles: Tip and Sample (electrodes)
# - Separated by a vacuum gap
# - All sizes and positions are parameterized so you can tweak easily
#
# Notes:
# • Output is saved as both SVG (vector for papers) and PNG.
# • No specific colors are set; uses matplotlib defaults so you can restyle globally later.
# • You can later add Fermi-level lines, bias tilt, band edges, work functions, etc.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

def draw_stm_energy_diagram(
    tip_height=4.0,        # "height" of the rectangle in arbitrary energy units
    sample_height=4.0,
    fermi_y=2.5,           # vertical reference for Fermi level (you can shift as you like)
    tip_width=1.0,         # rectangle widths (arbitrary lateral units)
    sample_width=1.0,
    gap_width=2.0,         # vacuum gap between tip and sample
    margin=0.8,            # margin around the diagram
    annotate=True,         # add simple labels
    fermi_line=True,       # draw a horizontal Fermi reference line
    fig_size=(6,4),        # figure size in inches
    lw=2.0,                # line width for rectangle edges
):
    # Derived positions (left edges)
    tip_x0 = 0.0
    gap_x0 = tip_x0 + tip_width
    sample_x0 = gap_x0 + gap_width

    # Compute overall bounds
    x_min = tip_x0 - margin
    x_max = sample_x0 + sample_width + margin
    y_min = 0.0 - margin
    y_max = max(tip_height, sample_height) + margin

    fig, ax = plt.subplots(figsize=fig_size, dpi=200)

    # Tip rectangle
    tip_rect = Rectangle(
        (tip_x0, 0.0), tip_width, tip_height,
        fill=True, linewidth=lw, color='#C5C5F0'  # soft periwinkle
    )
    ax.add_patch(tip_rect)

    # Sample rectangle
    sample_rect = Rectangle(
        (sample_x0, 0.0), sample_width, sample_height,
        fill=True, linewidth=lw, color='#C5C5F0'  # soft periwinkle
    )
    ax.add_patch(sample_rect)

    # Optional Fermi "reference" line across the whole diagram
    if fermi_line:
        ax.hlines(fermi_y, x_min, x_max, linestyles="--", linewidth=lw*0.8, color='#C5C5F0')

    # Clean up axes - remove all axes and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('auto')
    ax.axis('off')  # Remove all axes, ticks, and labels

    # Minor layout polish
    fig.tight_layout()

    # Save outputs (vector + raster)
    svg_path = "stm_energy_diagram.svg"
    png_path = "stm_energy_diagram.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    
    # Show the plot on screen
    plt.show()
    plt.close(fig)

    return svg_path, png_path

svg_path, png_path = draw_stm_energy_diagram()

svg_path, png_path
