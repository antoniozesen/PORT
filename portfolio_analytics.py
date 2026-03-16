"""PORTFOLIO ANALYTICS DASHBOARD — streamlit run portfolio_analytics.py"""

from __future__ import annotations

from datetime import date, datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from port_app.analytics import (
    ann_return,
    ann_vol,
    beta_alpha,
    cvar_hist,
    drawdown_series,
    efficient_frontier,
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
from port_app.constants import (
    APP_CSS,
    BBG_LAYOUT,
    BENCHMARK_CATALOGUE,
    HISTORICAL_SCENARIOS,
    BLUE,
    GREEN,
    ORANGE,
    PALETTE,
    PURPLE,
    RED,
)
from port_app.data import fetch_prices
from port_app.ui import fn, fp, mc

st.set_page_config(page_title="PORT | Portfolio Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(APP_CSS, unsafe_allow_html=True)

EXAMPLE_CSV = """date,event_type,identifier_type,identifier,side,shares,price,currency,fx_to_base,cash_amount,fees,note
2024-01-02,CASH,,,,,,EUR,1.0,30000,0,Capital inicial
2024-01-03,ORDER,TICKER,SPY,BUY,15,475,USD,0.92,,1,ETF USA
2024-01-04,ORDER,ISIN,US5949181045,BUY,10,410,USD,0.92,,1,Fondo/acción por ISIN
2024-03-20,CASH,,,,,,EUR,1.0,-1500,0,Retirada parcial
2024-05-10,ORDER,TICKER,SPY,SELL,3,520,USD,0.93,,1,Venta parcial
"""

ISIN_TO_TICKER = {
    "US5949181045": "MSFT",
    "US0378331005": "AAPL",
    "US02079K3059": "GOOGL",
    "US67066G1040": "NVDA",
    "US88160R1014": "TSLA",
    "IE00B5BMR087": "IWDA.AS",
}


def calmar_ratio(r: pd.Series) -> float:
    dd = max_drawdown(r)
    ar = ann_return(r)
    return ar / abs(dd) if dd and not np.isnan(dd) else np.nan


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
    tx["note"] = tx.get("note", "").astype(str)

    tx["ticker"] = ""
    tk = tx["identifier_type"].eq("TICKER")
    isin = tx["identifier_type"].eq("ISIN")
    tx.loc[tk, "ticker"] = tx.loc[tk, "identifier"]
    tx.loc[isin, "ticker"] = tx.loc[isin, "identifier"].map(ISIN_TO_TICKER).fillna("")

    errors: list[str] = []
    unresolved = tx[(tx["event_type"] == "ORDER") & (tx["ticker"] == "")]
    if not unresolved.empty:
        errors.append(f"ISIN/Ticker no resoluble: {', '.join(sorted(unresolved['identifier'].unique()))}")
    if tx["date"].isna().any():
        errors.append("Hay filas con fecha inválida.")
    bad_orders = tx[(tx["event_type"] == "ORDER") & ((tx["shares"] <= 0) | (tx["price"] <= 0) | ~tx["side"].isin(["BUY", "SELL"]))]
    if not bad_orders.empty:
        errors.append("Hay ORDER inválidas (shares/price/side).")
    bad_cash = tx[(tx["event_type"] == "CASH") & tx["cash_amount"].isna()]
    if not bad_cash.empty:
        errors.append("Hay CASH sin cash_amount.")

    tx["fx_to_base"] = np.where(tx["currency"].eq(base_currency), 1.0, tx["fx_to_base"])
    tx["gross_base"] = np.where(tx["event_type"].eq("ORDER"), tx["shares"] * tx["price"] * tx["fx_to_base"], 0.0)
    tx["cash_flow_base"] = 0.0
    tx.loc[tx["event_type"].eq("CASH"), "cash_flow_base"] = tx.loc[tx["event_type"].eq("CASH"), "cash_amount"] * tx.loc[tx["event_type"].eq("CASH"), "fx_to_base"]
    tx.loc[tx["event_type"].eq("ORDER") & tx["side"].eq("BUY"), "cash_flow_base"] = -tx["gross_base"] - tx["fees"]
    tx.loc[tx["event_type"].eq("ORDER") & tx["side"].eq("SELL"), "cash_flow_base"] = tx["gross_base"] - tx["fees"]

    return tx.sort_values("date"), errors


def build_holdings_and_values(prices: pd.DataFrame, tx: pd.DataFrame, base_currency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = prices.index
    order_tx = tx[tx["event_type"] == "ORDER"].copy()
    tickers = sorted(order_tx["ticker"].unique())
    holdings = pd.DataFrame(0.0, index=idx, columns=tickers)

    for _, row in order_tx.iterrows():
        d = row["date"]
        sign = 1.0 if row["side"] == "BUY" else -1.0
        if d in holdings.index and row["ticker"] in holdings.columns:
            holdings.loc[d:, row["ticker"]] += sign * float(row["shares"])

    px = prices.reindex(columns=tickers).ffill().bfill()
    value_by_asset = holdings * px

    cash_moves = tx.groupby("date", as_index=True)["cash_flow_base"].sum().reindex(idx, fill_value=0.0)
    cash_bal = cash_moves.cumsum()
    asset_val = value_by_asset.sum(axis=1)
    total_val = asset_val + cash_bal

    df = pd.DataFrame(
        {
            "asset_value": asset_val,
            f"cash_balance_{base_currency}": cash_bal,
            f"total_value_{base_currency}": total_val,
            "portfolio_return": total_val.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "external_flow": tx[tx["event_type"] == "CASH"].groupby("date")["cash_flow_base"].sum().reindex(idx, fill_value=0.0),
        },
        index=idx,
    )
    return df, value_by_asset


with st.sidebar:
    st.markdown("### ▶ PORT ANALYTICS")
    base_currency = st.selectbox("Divisa base", ["EUR", "USD", "GBP", "CHF", "JPY"], index=0)
    st.download_button("⬇ CSV de ejemplo", EXAMPLE_CSV.encode("utf-8"), "portfolio_template.csv", "text/csv", width="stretch")
    uploaded = st.file_uploader("Sube CSV de cartera", type=["csv"])
    st.caption("Permite CASH (+/-), ORDER BUY/SELL, títulos, fecha, precio, TICKER o ISIN.")

    bm_group = st.selectbox("Categoría benchmark", list(BENCHMARK_CATALOGUE.keys()))
    bm_name = st.selectbox("Benchmark", list(BENCHMARK_CATALOGUE[bm_group].keys()))
    bm_ticker = st.text_input("Benchmark custom", "").strip().upper() or BENCHMARK_CATALOGUE[bm_group][bm_name]

    start_date = st.date_input("Start", date(2020, 1, 1))
    end_date = st.date_input("End", date.today())
    rf_rate = st.number_input("Risk-free anual (%)", 0.0, 20.0, 4.0, 0.1) / 100
    var_conf = st.selectbox("VaR", ["95%", "99%", "90%"], index=0)
    var_conf_f = float(var_conf.replace("%", "")) / 100
    roll_days = int(st.selectbox("Rolling window", [21, 63, 126, 252], index=1))
    run = st.button("⚡ RUN ANALYSIS", width="stretch")

st.markdown(
    f'<div class="bbg-topbar"><span>PORT | PORTFOLIO ANALYTICS TERMINAL</span>'
    f'<span>{datetime.now().strftime("%d %b %Y %H:%M")} | BASE {base_currency} | BM {bm_ticker}</span></div>',
    unsafe_allow_html=True,
)

if not run:
    st.info("Sube tu CSV y pulsa RUN ANALYSIS.")
    st.stop()
if uploaded is None:
    st.error("Debes subir un CSV.")
    st.stop()

raw = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
tx, errs = parse_transactions(raw, base_currency)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

order_tickers = sorted(tx.loc[tx["event_type"] == "ORDER", "ticker"].unique())
if not order_tickers:
    st.error("No hay ORDER válidas.")
    st.stop()

all_tickers = tuple(sorted(set(order_tickers + [bm_ticker])))
prices = fetch_prices(all_tickers, str(start_date), str(end_date))
if prices.empty:
    st.error("No se pudieron descargar precios.")
    st.stop()

missing = [t for t in order_tickers if t not in prices.columns]
if missing:
    st.warning(f"Sin datos para: {', '.join(missing)}")
    tx = tx[~tx["ticker"].isin(missing)]
    order_tickers = sorted(tx.loc[tx["event_type"] == "ORDER", "ticker"].unique())
if not order_tickers:
    st.error("No quedan activos con precios descargables.")
    st.stop()

val_df, value_by_asset = build_holdings_and_values(prices, tx, base_currency)
pr = val_df["portfolio_return"].dropna()
bm_rets = prices[bm_ticker].pct_change().dropna() if bm_ticker in prices.columns else None
if bm_rets is not None:
    idx = pr.index.intersection(bm_rets.index)
    pr, br = pr.loc[idx], bm_rets.loc[idx]
else:
    br = None

asset_rets = {tk: prices[tk].pct_change().dropna() for tk in order_tickers if tk in prices.columns}
returns_df = pd.DataFrame(asset_rets).dropna()

twr_total = time_weighted_return(val_df[f"total_value_{base_currency}"], val_df["external_flow"])
cf_dates = [d.date() for d in tx[tx["event_type"] == "CASH"]["date"].tolist()]
cf_amts = [-float(a) for a in tx[tx["event_type"] == "CASH"]["cash_flow_base"].tolist()]
mwr_total = money_weighted_return(cf_dates, cf_amts, val_df.index[-1].date(), float(val_df[f"total_value_{base_currency}"].iloc[-1]))

P = {
    "ann_ret": ann_return(pr),
    "ann_vol": ann_vol(pr),
    "sharpe": sharpe(pr, rf_rate),
    "sortino": sortino(pr, rf_rate),
    "calmar": calmar_ratio(pr),
    "mdd": max_drawdown(pr),
    "var": var_hist(pr, var_conf_f),
    "cvar": cvar_hist(pr, var_conf_f),
    "skew": float(stats.skew(pr.dropna())) if len(pr.dropna()) else np.nan,
    "kurt": float(stats.kurtosis(pr.dropna())) if len(pr.dropna()) else np.nan,
    "twr": twr_total,
    "mwr": mwr_total,
    "cum": (1 + pr).cumprod(),
}

if br is not None:
    B = {"ann_ret": ann_return(br), "ann_vol": ann_vol(br), "sharpe": sharpe(br, rf_rate), "mdd": max_drawdown(br), "cum": (1 + br).cumprod()}
    P["beta"], P["alpha"] = beta_alpha(pr, br, rf_rate)
    P["te"], P["ir"] = tracking_error(pr, br), info_ratio(pr, br)
else:
    B = None
    P["beta"] = P["alpha"] = P["te"] = P["ir"] = np.nan

tabs = st.tabs(["📈 OVERVIEW", "⚠️ RISK", "📊 ATTRIBUTION", "🔗 CORRELATION", "🎯 EFFICIENT FRONTIER", "📉 SCENARIOS", "🔄 ROLLING", "📋 OPERATIONS"])

with tabs[0]:
    cols = st.columns(8)
    cards = [
        ("Ann Return", fp(P["ann_ret"]), "", "pos" if (P["ann_ret"] or 0) > 0 else "neg"),
        ("Ann Vol", fp(P["ann_vol"]), "", "neu"),
        ("Sharpe", fn(P["sharpe"]), f"rf={rf_rate*100:.1f}%", "orange"),
        ("Sortino", fn(P["sortino"]), "", "neu"),
        ("Calmar", fn(P["calmar"]), "", "neu"),
        ("Max DD", fp(P["mdd"]), "", "neg"),
        ("TWR", fp(P["twr"]), "time-weighted", "pos" if (P["twr"] or 0) > 0 else "neg"),
        ("MWR", fp(P["mwr"]), "money-weighted", "pos" if (P["mwr"] or 0) > 0 else "neg"),
    ]
    for col, data in zip(cols, cards):
        col.markdown(mc(*data), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P["cum"].index, y=P["cum"].values, name="Portfolio", line=dict(color=ORANGE, width=2.4)))
    if B:
        fig.add_trace(go.Scatter(x=B["cum"].index, y=B["cum"].values, name=bm_ticker, line=dict(color=BLUE, width=1.2, dash="dot")))
    fig.update_layout(**BBG_LAYOUT, title="Cumulative Return (Base=1)")
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df["asset_value"], name="Assets", line=dict(color=ORANGE)))
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df[f"cash_balance_{base_currency}"], name="Cash", line=dict(color=BLUE)))
    fig2.add_trace(go.Scatter(x=val_df.index, y=val_df[f"total_value_{base_currency}"], name="Total", line=dict(color=GREEN, width=2.3)))
    fig2.update_layout(**BBG_LAYOUT, title=f"Portfolio Value ({base_currency})")
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
        fig.add_vline(x=P["var"] * 100, line_color=RED, line_dash="dash")
        fig.add_vline(x=P["cvar"] * 100, line_color=PURPLE, line_dash="dot")
        fig.update_layout(**BBG_LAYOUT, title="Return Distribution", xaxis_title="Daily %")
        st.plotly_chart(fig, width="stretch")

    risk_df = pd.DataFrame({
        "Metric": ["Ann Return","Ann Vol","Sharpe","Sortino","Calmar","Max DD",f"VaR {var_conf}",f"CVaR {var_conf}","Skew","Kurtosis","Beta","Alpha","TE","IR"],
        "Portfolio": [fp(P["ann_ret"]),fp(P["ann_vol"]),fn(P["sharpe"]),fn(P["sortino"]),fn(P["calmar"]),fp(P["mdd"]),fp(P["var"]),fp(P["cvar"]),fn(P["skew"]),fn(P["kurt"]),fn(P["beta"]),fp(P["alpha"]),fp(P["te"]),fn(P["ir"])],
        "Benchmark": [fp(B["ann_ret"]) if B else "—",fp(B["ann_vol"]) if B else "—",fn(B["sharpe"]) if B else "—","—","—",fp(B["mdd"]) if B else "—","—","—","—","—","1.00","0.00%","—","—"],
    }).set_index("Metric")
    st.dataframe(risk_df, width="stretch")

with tabs[2]:
    weights = value_by_asset.iloc[-1]
    weights = weights[weights > 0]
    c1, c2 = st.columns(2)
    with c1:
        if weights.empty:
            st.info("Sin posiciones abiertas.")
        else:
            fig = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.5, marker=dict(colors=PALETTE[: len(weights)])))
            fig.update_layout(**BBG_LAYOUT, title="Current Allocation")
            st.plotly_chart(fig, width="stretch")
    with c2:
        contrib = {}
        total_assets = max(val_df["asset_value"].iloc[-1], 1e-9)
        for tk, r in asset_rets.items():
            if tk in weights.index:
                contrib[tk] = (weights[tk] / total_assets) * ann_return(r)
        if contrib:
            cs = pd.Series(contrib).sort_values()
            fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h", marker_color=[GREEN if v > 0 else RED for v in cs.values]))
            fig.update_layout(**BBG_LAYOUT, title="Return Contribution", xaxis_tickformat=".1%")
            st.plotly_chart(fig, width="stretch")

with tabs[3]:
    if len(returns_df.columns) < 2:
        st.info("No hay suficientes activos para correlación.")
    else:
        corr = returns_df.corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=[[0, "#1a0000"], [0.5, "#111"], [1, "#003d15"]]))
        fig.update_layout(**BBG_LAYOUT, title="Correlation Matrix")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.scatter_matrix(returns_df.iloc[:, : min(6, len(returns_df.columns))] * 100)
        fig2.update_layout(**BBG_LAYOUT, title="Scatter Matrix")
        st.plotly_chart(fig2, width="stretch")

with tabs[4]:
    if len(returns_df.columns) < 2:
        st.info("Se necesitan >=2 activos para frontera eficiente.")
    else:
        n_sim = st.slider("Simulaciones MC", 1000, 12000, 4000, 500)
        ef = efficient_frontier(returns_df, n=n_sim, rf=rf_rate)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.array(ef["mc_vol"]) * 100, y=np.array(ef["mc_ret"]) * 100, mode="markers", marker=dict(size=3, color=ef["mc_sr"], colorscale="Turbo"), name="Sim"))
        ms = ef["max_sr_w"]
        mv = ef["min_vol_w"]
        ms_r, ms_v = ms @ ef["mu"].values, np.sqrt(ms @ ef["cov"].values @ ms)
        mv_r, mv_v = mv @ ef["mu"].values, np.sqrt(mv @ ef["cov"].values @ mv)
        fig.add_trace(go.Scatter(x=[ms_v * 100], y=[ms_r * 100], mode="markers", marker=dict(size=14, color=ORANGE, symbol="star"), name="Max Sharpe"))
        fig.add_trace(go.Scatter(x=[mv_v * 100], y=[mv_r * 100], mode="markers", marker=dict(size=12, color=BLUE), name="Min Vol"))
        fig.update_layout(**BBG_LAYOUT, title="Efficient Frontier", xaxis_title="Vol %", yaxis_title="Ret %")
        st.plotly_chart(fig, width="stretch")

with tabs[5]:
    scen_sel = st.multiselect("Escenarios", list(HISTORICAL_SCENARIOS.keys()), default=["COVID Crash (Feb–Mar 2020)", "2022 Global Bear Market"])
    rows = []
    for scen in scen_sel:
        if scen == "Custom Period":
            continue
        s, e = HISTORICAL_SCENARIOS[scen]
        sp = fetch_prices(tuple(sorted(set(order_tickers + [bm_ticker]))), s, e)
        if sp.empty:
            continue
        tx_s = tx[tx["date"].between(pd.Timestamp(s), pd.Timestamp(e))]
        if tx_s.empty:
            continue
        vdf, _ = build_holdings_and_values(sp, tx_s, base_currency)
        sr = vdf["portfolio_return"].dropna()
        pt = (1 + sr).prod() - 1 if not sr.empty else np.nan
        bt = (1 + sp[bm_ticker].pct_change().dropna()).prod() - 1 if bm_ticker in sp.columns else np.nan
        rows.append({"Scenario": scen, "Portfolio": pt, "Benchmark": bt, "Excess": pt - bt if pd.notna(bt) else np.nan})
    if rows:
        sdf = pd.DataFrame(rows)
        st.dataframe(sdf.assign(Portfolio=sdf["Portfolio"].map(fp), Benchmark=sdf["Benchmark"].map(fp), Excess=sdf["Excess"].map(fp)).set_index("Scenario"), width="stretch")

with tabs[6]:
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
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values, line=dict(color=ORANGE, width=2)))
        fig.add_hline(y=0, line_color="#333", line_dash="dot")
        layout = {**BBG_LAYOUT}
        layout["yaxis"] = {**layout.get("yaxis", {}), "range": [-1.1, 1.1]}
        fig.update_layout(**layout, title=f"Rolling Corr {roll_days}d vs {bm_ticker}")
        st.plotly_chart(fig, width="stretch")

with tabs[7]:
    st.markdown("#### Operaciones parseadas")
    st.dataframe(tx, width="stretch")
    st.markdown("#### Posiciones actuales")
    open_pos = value_by_asset.iloc[-1].rename("market_value").reset_index().rename(columns={"index": "ticker"})
    st.dataframe(open_pos, width="stretch")
    export = prices.copy().join(val_df, how="left")
    st.download_button("⬇ Descargar resultados CSV", export.to_csv().encode("utf-8"), f"portfolio_results_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", width="stretch")
