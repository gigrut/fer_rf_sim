"""Optional standalone example of a filled sine-wave figure."""


def make_figure():
    import numpy as np
    import plotly.graph_objects as go

    positions = np.linspace(0, 2 * np.pi, 1000)
    values = -np.sin(positions)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=positions, y=np.full_like(positions, values.min() - 0.5), mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, name="lower_boundary",
    ))
    figure.add_trace(go.Scatter(
        x=positions, y=values, mode="lines", line=dict(color="black", width=8),
        fill="tonexty", fillcolor="#959FBF", name="Y = -sin(x)",
    ))
    figure.update_layout(
        showlegend=False, width=800, height=600,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return figure


def main():
    make_figure().show()


if __name__ == "__main__":
    main()
