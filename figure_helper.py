import numpy as np
import plotly.graph_objects as go

# Generate one full cycle of sine wave (0 to 2π)
x = np.linspace(0, 2*np.pi, 1000)
y = -np.sin(x)

# Create the figure
fig = go.Figure()

# Create a lower boundary for the fill (below the minimum of the curve)
y_min = np.min(y) - 0.5  # Extend below the curve
y_lower = np.full_like(x, y_min)

# Add the lower boundary (invisible)
fig.add_trace(go.Scatter(
    x=x,
    y=y_lower,
    mode='lines',
    line=dict(color='rgba(0,0,0,0)', width=0),  # Invisible line
    showlegend=False,
    name='lower_boundary'
))

# Add the sine wave line with fill below the curve
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='lines',
    line=dict(color='black', width=8),  # Much thicker line
    fill='tonexty',  # Fill to next y (the lower boundary)
    fillcolor='#959FBF',  # Fill color below the curve
    name='Y = -sin(x)'
))

# Update layout with no background and no labels/titles/legends
fig.update_layout(
    showlegend=False,  # Remove legend
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
    xaxis=dict(
        showticklabels=False,  # Remove x-axis tick labels
        showgrid=False,        # Remove grid lines
        zeroline=False,        # Remove zero line
        title=""               # Remove axis title
    ),
    yaxis=dict(
        showticklabels=False,  # Remove y-axis tick labels
        showgrid=False,        # Remove grid lines
        zeroline=False,        # Remove zero line
        title=""               # Remove axis title
    ),
    margin=dict(l=0, r=0, t=0, b=0)  # Remove all margins
)

# Show the figure
fig.show()
