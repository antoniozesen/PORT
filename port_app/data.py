from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class Position:
    ticker: str
    shares: float
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None


@st.cache_data(ttl=3600)
def fetch_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    symbols = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    if not symbols:
        return pd.DataFrame()
    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            px = raw["Close"].copy()
        else:
            px = raw[["Close"]].copy()
            px.columns = [symbols[0]]
        px = px.dropna(how="all")
    except Exception:
        px = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    missing = [s for s in symbols if s not in px.columns or px[s].dropna().empty]
    for sym in missing:
        try:
            d = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if not d.empty:
                px[sym] = d["Close"]
        except Exception:
            continue
    return px.dropna(how="all").sort_index()
