# bpcomp/plots.py
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def growth_plot(t, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="OD600"))
    fig.add_trace(go.Scatter(x=t, y=np.log10(np.maximum(y,1e-6)), mode="lines", name="log10(OD600)", line=dict(dash="dot"), yaxis="y2"))
    fig.update_layout(title="Simulated growth phases", template="plotly_white", height=420,
                      xaxis_title="Time (h)",
                      yaxis=dict(title="OD600"),
                      yaxis2=dict(title="log10(OD600)", overlaying="y", side="right"))
    return fig

def cfu_plot(t, cfu):
    fig = px.line(x=t, y=cfu, labels={"x":"Time (h)","y":"CFU/mL"}, title="Estimated CFU/mL (heuristic)")
    fig.update_layout(template="plotly_white", height=380)
    return fig
