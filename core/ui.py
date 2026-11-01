"""Small shared UI helpers: colours, plotly styling, formatting."""
import plotly.graph_objects as go

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b"]
NEW_COLOR, OLD_COLOR, ALERT_COLOR = "#2563eb", "#94a3b8", "#ef4444"


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=70, b=10), colorway=PALETTE,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1) if legend else None,
        showlegend=legend, hovermode="closest", title=dict(x=0, xanchor="left", y=0.97, yanchor="top"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def feature_group(name, new_features=()):
    if name in new_features:
        return "Client-ID features (new)"
    if name.startswith("card"):
        return "Card"
    if name.startswith("addr") or name.startswith("dist"):
        return "Address / distance"
    if name.endswith("emaildomain"):
        return "Email domain"
    if name.startswith("Transaction"):
        return "Amount / time"
    if name == "ProductCD":
        return "Product"
    if name[:1] == "C" and name[1:].isdigit():
        return "C (counts)"
    if name[:1] == "D" and name[1:].isdigit():
        return "D (time deltas)"
    if name[:1] == "M" and name[1:].isdigit():
        return "M (match flags)"
    if name[:1] == "V" and name[1:].isdigit():
        return "V (Vesta features)"
    if name.startswith("id_") or name.startswith("Device"):
        return "Identity / device"
    return "Other"


__all__ = ["style", "pct", "feature_group", "go", "PALETTE", "NEW_COLOR", "OLD_COLOR", "ALERT_COLOR"]
