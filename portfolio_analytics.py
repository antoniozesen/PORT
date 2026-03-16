"""PORTFOLIO ANALYTICS DASHBOARD — streamlit run portfolio_analytics.py"""

from __future__ import annotations

from datetime import date, datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from port_app.analytics import (
    ann_return,
    ann_vol,
    beta_alpha,
    cvar_hist,
    drawdown_series,
    info_ratio,
    max_drawdown,
    money_weighted_return,
    rolling_sharpe,
    sharpe,
    sortino,
    time_weighted_return,
    tracking_error,
    var_hist,
)
from port_app.constants import APP_CSS, BBG_LAYOUT, BENCHMARK_CATALOGUE, BLUE, GREEN, ORANGE, RED
from port_app.data import fetch_prices
from port_app.ui import fn, fp, mc

st.set_page_config(page_title="PORT | Portfolio Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(APP_CSS, unsafe_allow_html=True)

EXAMPLE_CSV = """date,event_type,identifier_type,identifier,side,shares,price,currency,fx_to_base,cash_amount,fees,note
2024-01-02,CASH,,,,,,EUR,1.0,25000,0,Capital inicial
2024-01-03,ORDER,TICKER,AAPL,BUY,20,185,USD,0.92,,2,Compra AAPL
2024-01-04,ORDER,ISIN,US5949181045,BUY,15,410,USD,0.92,,2,Compra MSFT por ISIN
2024-03-20,CASH,,,,,,EUR,1.0,-1500,0,Retirada parcial
2024-05-10,ORDER,TICKER,AAPL,SELL,5,205,USD,0.93,,2,Venta parcial AAPL
"""

ISIN_TO_TICKER = {
    "US5949181045": "MSFT",
    "US0378331005": "AAPL",
    "US02079K3059": "GOOGL",
    "US67066G1040": "NVDA",
    "US88160R1014": "TSLA",
}


def parse_transactions(df: pd.DataFrame, base_currency: str) -> tuple[pd.DataFrame, list[str]]:
    required = {"date", "event_type", "currency", "fx_to_base"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(), [f"Faltan columnas obligatorias: {', '.join(sorted(missing))}"]

    tx = df.copy()
    tx.columns = [c.strip() for c in tx.columns]
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    tx["event_type"] = tx["event_type"].astype(str).str.upper().str.strip()
    tx["identifier_type"] = tx.get("identifier_type", "").astype(str).str.upper().str.strip()
    tx["identifier"] = tx.get("identifier", "").astype(str).str.upper().str.strip()
    tx["side"] = tx.get("side", "").astype(str).str.upper().str.strip()
    tx["currency"] = tx["currency"].astype(str).str.upper().str.strip()
    tx["shares"] = pd.to_numeric(tx.get("shares", np.nan), errors="coerce")
    tx["price"] = pd.to_numeric(tx.get("price", np.nan), errors="coerce")
    tx["fx_to_base"] = pd.to_numeric(tx["fx_to_base"], errors="coerce").fillna(1.0)
    tx["cash_amount"] = pd.to_numeric(tx.get("cash_amount", np.nan), errors="coerce")
    tx["fees"] = pd.to_numeric(tx.get("fees", 0.0), errors="coerce").fillna(0.0)

    tx["ticker"] = ""
    is_ticker = tx["identifier_type"].eq("TICKER")
    is_isin = tx["identifier_type"].eq("ISIN")
    tx.loc[is_ticker, "ticker"] = tx.loc[is_ticker, "identifier"]
    tx.loc[is_isin, "ticker"] = tx.loc[is_isin, "identifier"].map(ISIN_TO_TICKER).fillna("")

    errors: list[str] = []
    unresolved = tx[(tx["event_type"] == "ORDER") & (tx["ticker"] == "")]
    if not unresolved.empty:
        missing_ids = ", ".join(sorted(unresolved["identifier"].unique()))
        errors.append(f"ISIN/Ticker no resoluble para ORDER: {missing_ids}. Añade ticker o usa ISIN soportado.")

    bad_dates = tx[tx["date"].isna()]
    if not bad_dates.empty:
        errors.append("Hay filas con date inválida.")

    invalid_orders = tx[(tx["event_type"] == "ORDER") & ((tx["shares"] <= 0) | (tx["price"] <= 0) | ~tx["side"].isin(["BUY", "SELL"]))]
    if not invalid_orders.empty:
        errors.append("Hay ORDER con shares/price/side inválidos.")

    invalid_cash = tx[(tx["event_type"] == "CASH") & (tx["cash_amount"].isna())]
    if not invalid_cash.empty:
        errors.append("Hay CASH con cash_amount vacío.")

    tx["fx_to_base"] = np.where(tx["currency"].eq(base_currency), 1.0, tx["fx_to_base"])
    tx["gross_base"] = 0.0
    order_mask = tx["event_type"] == "ORDER"
    tx.loc[order_mask, "gross_base"] = tx.loc[order_mask, "shares"] * tx.loc[order_mask, "price"] * tx.loc[order_mask, "fx_to_base"]

    tx["cash_flow_base"] = 0.0
    tx.loc[tx["event_type"].eq("CASH"), "cash_flow_base"] = tx.loc[tx["event_type"].eq("CASH"), "cash_amount"] * tx.loc[tx["event_type"].eq("CASH"), "fx_to_base"]
    tx.loc[tx["event_type"].eq("ORDER") & tx["side"].eq("BUY"), "cash_flow_base"] = -tx["gross_base"] - tx["fees"]
    tx.loc[tx["event_type"].eq("ORDER") & tx["side"].eq("SELL"), "cash_flow_base"] = tx["gross_base"] - tx["fees"]

    return tx.sort_values("date"), errors


def build_holdings_and_value(prices: pd.DataFrame, tx: pd.DataFrame, base_currency: str) -> tuple[pd.DataFrame, pd.Series]:
    idx = prices.index
    order_tx = tx[tx["event_type"] == "ORDER"].copy()
    tickers = sorted(order_tx["ticker"].unique())

    holdings = pd.DataFrame(0.0, index=idx, columns=tickers)
    for _, row in order_tx.iterrows():
        sign = 1.0 if row["side"] == "BUY" else -1.0
        d = row["date"]
        if d in holdings.index and row["ticker"] in holdings.columns:
            holdings.loc[d:, row["ticker"]] += sign * float(row["shares"])

    cash_flows = tx.groupby("date", as_index=True)["cash_flow_base"].sum().reindex(idx, fill_value=0.0)
    cash_balance = cash_flows.cumsum()

    px = prices.reindex(columns=tickers).ffill().bfill()
    asset_value = (holdings * px).sum(axis=1)
    total_value = asset_value + cash_balance

    df = pd.DataFrame(
        {
            "asset_value": asset_value,
            f"cash_balance_{base_currency}": cash_balance,
            f"total_value_{base_currency}": total_value,
            "portfolio_return": total_value.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "external_flow": tx[tx["event_type"] == "CASH"].groupby("date")["cash_flow_base"].sum().reindex(idx, fill_value=0.0),
        },
        index=idx,
    )
    return df, cash_flows


with st.sidebar:
    st.markdown("### ▶ PORT ANALYTICS")
    base_currency = st.selectbox("Divisa base cartera", ["EUR", "USD", "GBP", "CHF", "JPY"], index=0)

    st.download_button(
        "⬇ Descargar CSV de ejemplo",
        EXAMPLE_CSV.encode("utf-8"),
        file_name="portfolio_template.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader("Sube CSV de cartera", type=["csv"])

    st.caption("Soporta CASH (+/-) y ORDER (BUY/SELL) por TICKER o ISIN. Para divisas, usa fx_to_base.")

    bm_group = st.selectbox("Categoría benchmark", list(BENCHMARK_CATALOGUE.keys()))
    bm_name = st.selectbox("Benchmark", list(BENCHMARK_CATALOGUE[bm_group].keys()))
    bm_ticker = st.text_input("Custom benchmark", "").strip().upper() or BENCHMARK_CATALOGUE[bm_group][bm_name]

    start_date = st.date_input("Start", date(2020, 1, 1))
    end_date = st.date_input("End", date.today())
    rf_rate = st.number_input("Risk-free anual (%)", 0.0, 20.0, 4.0, 0.1) / 100
    var_conf = st.selectbox("VaR", ["95%", "99%", "90%"], index=0)
    var_conf_f = float(var_conf.replace("%", "")) / 100
    roll_days = int(st.selectbox("Rolling window", [21, 63, 126, 252], index=1))
    run = st.button("⚡ RUN ANALYSIS", width="stretch")

st.markdown(
    f'<div class="bbg-topbar"><span>PORT | PORTFOLIO ANALYTICS TERMINAL</span>'
    f'<span>{datetime.now().strftime("%d %b %Y %H:%M")} | BASE: {base_currency} | BM: {bm_ticker}</span></div>',
    unsafe_allow_html=True,
)

if not run:
    st.info("1) Descarga ejemplo CSV 2) súbelo con tus operaciones/caja 3) pulsa RUN ANALYSIS.")
    st.stop()

if uploaded is None:
    st.error("Debes subir un CSV de cartera.")
    st.stop()

raw = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
tx, errs = parse_transactions(raw, base_currency)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

order_tickers = sorted(tx.loc[tx["event_type"] == "ORDER", "ticker"].unique())
if not order_tickers:
    st.error("No hay ORDER válidas en el CSV.")
    st.stop()

all_tickers = tuple(sorted(set(order_tickers + [bm_ticker])))
prices = fetch_prices(all_tickers, str(start_date), str(end_date))
if prices.empty:
    st.error("No se pudieron descargar precios.")
    st.stop()

missing = [t for t in order_tickers if t not in prices.columns]
if missing:
    st.warning(f"Sin precios para: {', '.join(missing)}")
    tx = tx[~tx["ticker"].isin(missing)]

if tx[tx["event_type"] == "ORDER"].empty:
    st.error("No quedan órdenes con ticker descargable.")
    st.stop()

val_df, all_flows = build_holdings_and_value(prices, tx, base_currency)
pr = val_df["portfolio_return"].dropna()
bm_rets = prices[bm_ticker].pct_change().dropna() if bm_ticker in prices.columns else None
if bm_rets is not None:
    ix = pr.index.intersection(bm_rets.index)
    pr, br = pr.loc[ix], bm_rets.loc[ix]
else:
    br = None

external_flow = val_df["external_flow"]
twr_total = time_weighted_return(val_df[f"total_value_{base_currency}"], external_flow)
cf_dates = [d.date() for d in tx[tx["event_type"] == "CASH"]["date"].tolist()]
cf_amts = [-float(a) for a in tx[tx["event_type"] == "CASH"]["cash_flow_base"].tolist()]
mwr_total = money_weighted_return(cf_dates, cf_amts, val_df.index[-1].date(), float(val_df[f"total_value_{base_currency}"].iloc[-1]))

P = {
    "ann_ret": ann_return(pr),
    "ann_vol": ann_vol(pr),
    "sharpe": sharpe(pr, rf_rate),
    "sortino": sortino(pr, rf_rate),
    "mdd": max_drawdown(pr),
    "var": var_hist(pr, var_conf_f),
    "cvar": cvar_hist(pr, var_conf_f),
    "twr": twr_total,
    "mwr": mwr_total,
    "cum": (1 + pr).cumprod(),
}

if br is not None:
    B = {
        "ann_ret": ann_return(br),
        "ann_vol": ann_vol(br),
        "sharpe": sharpe(br, rf_rate),
        "mdd": max_drawdown(br),
        "cum": (1 + br).cumprod(),
    }
    P["beta"], P["alpha"] = beta_alpha(pr, br, rf_rate)
    P["te"], P["ir"] = tracking_error(pr, br), info_ratio(pr, br)
else:
    B = None
    P["beta"] = P["alpha"] = P["te"] = P["ir"] = np.nan

tabs = st.tabs(["📈 OVERVIEW", "⚠️ RISK", "🔄 ROLLING", "📋 OPERATIONS"])

with tabs[0]:
    cols = st.columns(8)
    cards = [
        ("Ann Return", fp(P["ann_ret"]), "", "pos" if (P["ann_ret"] or 0) > 0 else "neg"),
        ("Ann Vol", fp(P["ann_vol"]), "", "neu"),
        ("Sharpe", fn(P["sharpe"]), f"rf={rf_rate*100:.1f}%", "orange"),
        ("Sortino", fn(P["sortino"]), "", "neu"),
        ("Max DD", fp(P["mdd"]), "", "neg"),
        (f"VaR {var_conf}", fp(P["var"]), "", "neg"),
        ("TWR", fp(P["twr"]), "Time weighted", "pos" if (P["twr"] or 0) > 0 else "neg"),
        ("MWR/XIRR", fp(P["mwr"]), "Money weighted", "pos" if (P["mwr"] or 0) > 0 else "neg"),
    ]
    for c, item in zip(cols, cards):
        c.markdown(mc(*item), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P["cum"].index, y=P["cum"].values, name="Portfolio", line=dict(color=ORANGE, width=2.4)))
    if B:
        fig.add_trace(go.Scatter(x=B["cum"].index, y=B["cum"].values, name=bm_ticker, line=dict(color=BLUE, width=1.3, dash="dot")))
    fig.update_layout(**BBG_LAYOUT, title="Cumulative Return (Base=1)")
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df["asset_value"], name="Assets", line=dict(color=ORANGE)))
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df[f"cash_balance_{base_currency}"], name="Cash", line=dict(color=BLUE)))
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df[f"total_value_{base_currency}"], name="Total", line=dict(color=GREEN, width=2.3)))
    fig2.update_layout(**BBG_LAYOUT, title=f"Portfolio Value in {base_currency}")
    st.plotly_chart(fig2, width="stretch")

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        dd = drawdown_series(pr)
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=RED)))
        fig.update_layout(**BBG_LAYOUT, title="Underwater", yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(go.Histogram(x=pr * 100, nbinsx=70, marker_color=ORANGE))
        fig.update_layout(**BBG_LAYOUT, title="Return Distribution", xaxis_title="Daily %")
        st.plotly_chart(fig, width="stretch")

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        rs = rolling_sharpe(pr, roll_days)
        fig = go.Figure(go.Scatter(x=rs.index, y=rs.values, line=dict(color=ORANGE)))
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling Sharpe ({roll_days}d)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        rv = pr.rolling(roll_days).std() * np.sqrt(252)
        fig = go.Figure(go.Scatter(x=rv.index, y=rv.values, line=dict(color=BLUE)))
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling Vol ({roll_days}d)", yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

    if br is not None:
        rc = pr.rolling(roll_days).corr(br)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values, line=dict(color=ORANGE, width=2), name="Rolling corr"))
        fig.add_hline(y=0, line_color="#333", line_dash="dot")
        # Evita conflicto de kwargs con BBG_LAYOUT (yaxis ya existe)
        layout = {**BBG_LAYOUT}
        layout["yaxis"] = {**layout.get("yaxis", {}), "range": [-1.1, 1.1]}
        fig.update_layout(**layout, title=f"Rolling {roll_days}d Correlation vs {bm_ticker}")
        st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.markdown("#### Operaciones parseadas")
    st.dataframe(tx, width="stretch")
    export = prices.copy()
    export = export.join(val_df, how="left")
    st.download_button(
        "⬇ Descargar resultados CSV",
        export.to_csv().encode("utf-8"),
        file_name=f"portfolio_results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
    )
