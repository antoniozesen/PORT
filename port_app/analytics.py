from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize

from .constants import TRADING_DAYS
from .data import Position


def safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    v = ann_vol(r)
    return (ann_return(r) - rf) / v if v > 0 else np.nan


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    down = r[r < 0]
    dv = down.std() * np.sqrt(TRADING_DAYS)
    return float((ann_return(r) - rf) / dv) if dv > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    c = (1 + r.fillna(0)).cumprod()
    return float(((c - c.cummax()) / c.cummax()).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    c = (1 + r.fillna(0)).cumprod()
    return (c - c.cummax()) / c.cummax()


def var_hist(r: pd.Series, c: float = 0.95) -> float:
    x = r.dropna()
    return float(np.percentile(x, (1 - c) * 100)) if not x.empty else np.nan


def cvar_hist(r: pd.Series, c: float = 0.95) -> float:
    x = r.dropna()
    if x.empty:
        return np.nan
    v = var_hist(x, c)
    return float(x[x <= v].mean())


def beta_alpha(pr: pd.Series, br: pd.Series, rf: float = 0.0) -> tuple[float, float]:
    a = pd.concat([pr, br], axis=1).dropna()
    if len(a) < 5:
        return np.nan, np.nan
    x = a.iloc[:, 1].values - rf / TRADING_DAYS
    y = a.iloc[:, 0].values - rf / TRADING_DAYS
    sl, ic, *_ = stats.linregress(x, y)
    return float(sl), float(ic * TRADING_DAYS)


def tracking_error(pr: pd.Series, br: pd.Series) -> float:
    return float((pr - br).dropna().std() * np.sqrt(TRADING_DAYS))


def info_ratio(pr: pd.Series, br: pd.Series) -> float:
    te = tracking_error(pr, br)
    return float((ann_return(pr) - ann_return(br)) / te) if te > 0 else np.nan


def rolling_sharpe(r: pd.Series, w: int = 63) -> pd.Series:
    return (r.rolling(w).mean() * TRADING_DAYS) / (r.rolling(w).std() * np.sqrt(TRADING_DAYS))


def build_portfolio_value(prices: pd.DataFrame, positions: list[Position], cash_flows: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    idx = prices.index
    holdings = pd.DataFrame(0.0, index=idx, columns=[p.ticker for p in positions])
    for p in positions:
        if p.ticker not in prices.columns:
            continue
        active = (idx.date >= p.entry_date) & ((p.exit_date is None) | (idx.date <= p.exit_date))
        holdings.loc[active, p.ticker] += p.shares

    mv_assets = (holdings * prices.reindex(columns=holdings.columns).fillna(method="ffill").fillna(method="bfill")).sum(axis=1)
    cash_ledger = pd.Series(0.0, index=idx)

    for p in positions:
        ed = pd.Timestamp(p.entry_date)
        if ed in cash_ledger.index:
            cash_ledger.loc[ed] -= p.shares * p.entry_price
        if p.exit_date and p.exit_price is not None:
            xd = pd.Timestamp(p.exit_date)
            if xd in cash_ledger.index:
                cash_ledger.loc[xd] += p.shares * p.exit_price

    if not cash_flows.empty:
        cf = cash_flows.copy()
        cf["date"] = pd.to_datetime(cf["date"]).dt.normalize()
        for _, row in cf.iterrows():
            d, amt = row["date"], float(row["amount"])
            if d in cash_ledger.index:
                cash_ledger.loc[d] += amt

    cash_balance = cash_ledger.cumsum()
    total_value = mv_assets + cash_balance
    out = pd.DataFrame({"asset_value": mv_assets, "cash_balance": cash_balance, "total_value": total_value})
    out["returns"] = safe_pct_change(out["total_value"])
    return out


def time_weighted_return(total_value: pd.Series, external_flows: pd.Series) -> float:
    df = pd.DataFrame({"v": total_value, "f": external_flows}).dropna()
    if len(df) < 2:
        return np.nan
    chain = 1.0
    prev = df["v"].iloc[0]
    for i in range(1, len(df)):
        cur = df["v"].iloc[i]
        flow = df["f"].iloc[i]
        base = prev + flow
        if base <= 0:
            prev = cur
            continue
        chain *= (cur / base)
        prev = cur
    return float(chain - 1)


def xirr(dates: list[date], amounts: list[float]) -> float:
    if len(dates) < 2:
        return np.nan
    t0 = min(dates)
    years = np.array([(d - t0).days / 365.25 for d in dates], dtype=float)
    amts = np.array(amounts, dtype=float)
    if not (np.any(amts > 0) and np.any(amts < 0)):
        return np.nan

    def npv(r: float) -> float:
        return float(np.sum(amts / (1 + r) ** years))

    try:
        return float(brentq(npv, -0.9999, 10.0, maxiter=1000))
    except Exception:
        return np.nan


def money_weighted_return(cash_flow_dates: list[date], cash_flow_amounts: list[float], final_date: date, final_value: float) -> float:
    dates = list(cash_flow_dates) + [final_date]
    amounts = list(cash_flow_amounts) + [final_value]
    return xirr(dates, amounts)


def efficient_frontier(returns_df: pd.DataFrame, n: int = 3000, rf: float = 0.02) -> dict:
    mu = returns_df.mean() * TRADING_DAYS
    cov = returns_df.cov() * TRADING_DAYS
    na = len(mu)
    mc_ret, mc_vol, mc_sr = [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(na))
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        mc_ret.append(r)
        mc_vol.append(v)
        mc_sr.append((r - rf) / v if v > 0 else np.nan)

    bounds = tuple((0, 1) for _ in range(na))
    eq = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sr(w):
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        return -((r - rf) / v)

    max_sr = minimize(neg_sr, x0=np.ones(na) / na, method="SLSQP", bounds=bounds, constraints=eq)
    min_vol = minimize(lambda w: np.sqrt(w @ cov.values @ w), x0=np.ones(na) / na, method="SLSQP", bounds=bounds, constraints=eq)
    return {
        "mc_ret": mc_ret,
        "mc_vol": mc_vol,
        "mc_sr": mc_sr,
        "tickers": list(returns_df.columns),
        "mu": mu,
        "cov": cov,
        "max_sr_w": max_sr.x if max_sr.success else np.ones(na) / na,
        "min_vol_w": min_vol.x if min_vol.success else np.ones(na) / na,
    }
