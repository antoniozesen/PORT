from __future__ import annotations

import pandas as pd


def fp(v, d=2):
    return "—" if pd.isna(v) else f"{v*100:.{d}f}%"


def fn(v, d=2):
    return "—" if pd.isna(v) else f"{v:.{d}f}"


def mc(label, value, sub="", css="neu"):
    return (
        f'<div class="m-card"><div class="m-label">{label}</div>'
        f'<div class="m-value {css}">{value}</div>'
        + (f'<div class="m-sub">{sub}</div>' if sub else "")
        + "</div>"
    )
