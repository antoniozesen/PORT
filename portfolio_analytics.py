"""PORTFOLIO ANALYTICS DASHBOARD — streamlit run portfolio_analytics.py"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from port_app.analytics import (
    ann_return,
    ann_vol,
    beta_alpha,
    build_portfolio_value,
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
    BLUE,
    GREEN,
    HISTORICAL_SCENARIOS,
    ORANGE,
    PALETTE,
    PURPLE,
    RED,
)
from port_app.data import Position, fetch_prices
from port_app.ui import fn, fp, mc

st.set_page_config(page_title="PORT | Portfolio Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(APP_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ▶ PORT ANALYTICS")
    st.caption("Aporte de caja = + (entra dinero al portfolio) · Retiro = -")

    positions_df = st.data_editor(
        pd.DataFrame(
            [
                {"ticker": "AAPL", "shares": 50.0, "entry_date": date(2022, 1, 3), "entry_price": 171.0, "exit_date": pd.NaT, "exit_price": np.nan},
                {"ticker": "MSFT", "shares": 25.0, "entry_date": date(2022, 1, 3), "entry_price": 334.0, "exit_date": pd.NaT, "exit_price": np.nan},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="positions_editor",
    )

    cash_df = st.data_editor(
        pd.DataFrame(
            [
                {"date": date(2022, 1, 3), "amount": 25000.0, "note": "Capital inicial"},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="cash_editor",
    )

    bm_group = st.selectbox("Categoría benchmark", list(BENCHMARK_CATALOGUE.keys()))
    bm_name = st.selectbox("Benchmark", list(BENCHMARK_CATALOGUE[bm_group].keys()))
    bm_ticker = st.text_input("Custom benchmark (opcional)", "").strip().upper() or BENCHMARK_CATALOGUE[bm_group][bm_name]

    start_date = st.date_input("Start", date(2020, 1, 1))
    end_date = st.date_input("End", date.today())
    rf_rate = st.number_input("Risk-free anual (%)", 0.0, 20.0, 4.0, 0.1) / 100
    var_conf = st.selectbox("VaR", ["95%", "99%", "90%"], index=0)
    var_conf_f = float(var_conf.replace("%", "")) / 100
    roll_days = int(st.selectbox("Rolling window", [21, 63, 126, 252], index=1))
    run = st.button("⚡ RUN ANALYSIS", use_container_width=True)

st.markdown(
    f'<div class="bbg-topbar"><span>PORT | PORTFOLIO ANALYTICS TERMINAL</span>'
    f'<span>{datetime.now().strftime("%d %b %Y %H:%M")} | BM: {bm_ticker}</span></div>',
    unsafe_allow_html=True,
)

if not run:
    st.info("Carga/edita posiciones y caja en sidebar, luego pulsa RUN ANALYSIS.")
    st.stop()

positions: list[Position] = []
for _, r in positions_df.iterrows():
    tk = str(r.get("ticker", "")).strip().upper()
    shares = float(r.get("shares", 0) or 0)
    if not tk or shares <= 0:
        continue
    ed = pd.to_datetime(r.get("entry_date"), errors="coerce")
    if pd.isna(ed):
        continue
    xd = pd.to_datetime(r.get("exit_date"), errors="coerce")
    xp = r.get("exit_price", np.nan)
    positions.append(
        Position(
            ticker=tk,
            shares=shares,
            entry_date=ed.date(),
            entry_price=float(r.get("entry_price", np.nan) or np.nan),
            exit_date=None if pd.isna(xd) else xd.date(),
            exit_price=None if pd.isna(xp) else float(xp),
        )
    )

if not positions:
    st.error("Necesitas al menos una posición válida (ticker + shares + entry_date + entry_price).")
    st.stop()

cash_df = cash_df.copy()
if not cash_df.empty:
    cash_df["date"] = pd.to_datetime(cash_df["date"], errors="coerce").dt.date
    cash_df["amount"] = pd.to_numeric(cash_df["amount"], errors="coerce").fillna(0.0)
    cash_df = cash_df.dropna(subset=["date"])

all_tickers = tuple(sorted({p.ticker for p in positions} | {bm_ticker}))
prices = fetch_prices(all_tickers, str(start_date), str(end_date))
if prices.empty:
    st.error("No se pudo descargar precios. Revisa tickers y periodo.")
    st.stop()

missing_tickers = [p.ticker for p in positions if p.ticker not in prices.columns]
if missing_tickers:
    st.warning(f"Sin datos para: {', '.join(sorted(set(missing_tickers)))}")
positions = [p for p in positions if p.ticker in prices.columns]
if not positions:
    st.error("Tras filtrar tickers sin datos, no quedan posiciones válidas.")
    st.stop()

port_df = build_portfolio_value(prices[[c for c in prices.columns if c != bm_ticker] + ([bm_ticker] if bm_ticker in prices.columns else [])], positions, cash_df)
if port_df.empty:
    st.error("No se pudo construir el valor de cartera.")
    st.stop()

pr = port_df["returns"].dropna()
bm_rets = prices[bm_ticker].pct_change().dropna() if bm_ticker in prices.columns else None
if bm_rets is not None:
    idx = pr.index.intersection(bm_rets.index)
    pr, br = pr.loc[idx], bm_rets.loc[idx]
else:
    br = None

asset_rets = {p.ticker: prices[p.ticker].pct_change().dropna() for p in positions if p.ticker in prices.columns}
returns_df = pd.DataFrame(asset_rets).dropna()

external_flow_series = pd.Series(0.0, index=port_df.index)
for _, row in cash_df.iterrows():
    d = pd.Timestamp(row["date"])
    if d in external_flow_series.index:
        external_flow_series.loc[d] += float(row["amount"])

twr_total = time_weighted_return(port_df["total_value"], external_flow_series)
cash_flow_dates = [pd.Timestamp(d).date() for d in cash_df["date"].tolist()] if not cash_df.empty else []
cash_flow_amounts = [-float(a) for a in cash_df["amount"].tolist()] if not cash_df.empty else []
mwr = money_weighted_return(cash_flow_dates, cash_flow_amounts, port_df.index[-1].date(), float(port_df["total_value"].iloc[-1]))

P = {
    "ann_ret": ann_return(pr),
    "ann_vol": ann_vol(pr),
    "sharpe": sharpe(pr, rf_rate),
    "sortino": sortino(pr, rf_rate),
    "mdd": max_drawdown(pr),
    "var": var_hist(pr, var_conf_f),
    "cvar": cvar_hist(pr, var_conf_f),
    "twr": twr_total,
    "mwr": mwr,
    "cum": (1 + pr).cumprod(),
}

if br is not None:
    B = {"ann_ret": ann_return(br), "ann_vol": ann_vol(br), "sharpe": sharpe(br, rf_rate), "mdd": max_drawdown(br), "cum": (1 + br).cumprod()}
    P["beta"], P["alpha"] = beta_alpha(pr, br, rf_rate)
    P["te"], P["ir"] = tracking_error(pr, br), info_ratio(pr, br)
else:
    B = None
    P["beta"] = P["alpha"] = P["te"] = P["ir"] = np.nan

tabs = st.tabs(["📈 OVERVIEW", "⚠️ RISK", "📊 ATTRIBUTION", "🔗 CORRELATION", "🎯 EFFICIENT FRONTIER", "📉 SCENARIOS", "🔄 ROLLING", "📋 POSITIONS & CASH"])

with tabs[0]:
    cols = st.columns(8)
    cards = [
        ("Ann Return", fp(P["ann_ret"]), "", "pos" if (P["ann_ret"] or 0) > 0 else "neg"),
        ("Ann Vol", fp(P["ann_vol"]), "", "neu"),
        ("Sharpe", fn(P["sharpe"]), f"rf={rf_rate*100:.1f}%", "orange"),
        ("Sortino", fn(P["sortino"]), "", "neu"),
        ("Max DD", fp(P["mdd"]), "", "neg"),
        (f"VaR {var_conf}", fp(P["var"]), "", "neg"),
        ("TWR", fp(P["twr"]), "Time-weighted", "pos" if (P["twr"] or 0) > 0 else "neg"),
        ("MWR/XIRR", fp(P["mwr"]), "Money-weighted", "pos" if (P["mwr"] or 0) > 0 else "neg"),
    ]
    for c, item in zip(cols, cards):
        c.markdown(mc(*item), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P["cum"].index, y=P["cum"].values, name="Portfolio", line=dict(color=ORANGE, width=2.5)))
    if B:
        fig.add_trace(go.Scatter(x=B["cum"].index, y=B["cum"].values, name=bm_ticker, line=dict(color=BLUE, width=1.4, dash="dot")))
    fig.update_layout(**BBG_LAYOUT, title="Cumulative Return (Base=1.0)")
    st.plotly_chart(fig, use_container_width=True)

    tv = go.Figure()
    tv.add_trace(go.Scatter(x=port_df.index, y=port_df["asset_value"], name="Assets", line=dict(color=ORANGE)))
    tv.add_trace(go.Scatter(x=port_df.index, y=port_df["cash_balance"], name="Cash", line=dict(color=PURPLE)))
    tv.add_trace(go.Scatter(x=port_df.index, y=port_df["total_value"], name="Total", line=dict(color=GREEN, width=2.2)))
    tv.update_layout(**BBG_LAYOUT, title="Portfolio Value: Assets + Cash")
    st.plotly_chart(tv, use_container_width=True)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        dd = drawdown_series(pr)
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=RED), name="Drawdown"))
        fig.update_layout(**BBG_LAYOUT, title="Underwater", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Histogram(x=pr * 100, nbinsx=60, marker_color=ORANGE))
        fig.update_layout(**BBG_LAYOUT, title="Return Distribution", xaxis_title="Daily %")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Metric": ["Ann Return", "Ann Vol", "Sharpe", "Sortino", "Max DD", f"VaR {var_conf}", f"CVaR {var_conf}", "Beta", "Alpha", "TE", "IR", "TWR", "MWR"],
        "Portfolio": [fp(P["ann_ret"]), fp(P["ann_vol"]), fn(P["sharpe"]), fn(P["sortino"]), fp(P["mdd"]), fp(P["var"]), fp(P["cvar"]), fn(P["beta"]), fp(P["alpha"]), fp(P["te"]), fn(P["ir"]), fp(P["twr"]), fp(P["mwr"])],
        "Benchmark": [fp(B["ann_ret"]) if B else "—", fp(B["ann_vol"]) if B else "—", fn(B["sharpe"]) if B else "—", "—", fp(B["mdd"]) if B else "—", "—", "—", "1.00", "0.00%", "—", "—", "—", "—"],
    }).set_index("Metric"), use_container_width=True)

with tabs[2]:
    alloc = pd.DataFrame([{"ticker": p.ticker, "shares": p.shares} for p in positions]).groupby("ticker", as_index=False)["shares"].sum()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=alloc["ticker"], values=alloc["shares"], hole=0.5, marker=dict(colors=PALETTE[: len(alloc)])))
        fig.update_layout(**BBG_LAYOUT, title="Allocation by Shares")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        contrib = {}
        for t, r in asset_rets.items():
            px_last = prices[t].iloc[-1]
            sh = float(alloc.loc[alloc["ticker"] == t, "shares"].sum())
            weight_mv = (px_last * sh) / max(port_df["asset_value"].iloc[-1], 1e-9)
            contrib[t] = weight_mv * ann_return(r)
        cs = pd.Series(contrib).sort_values()
        fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h", marker_color=[GREEN if v > 0 else RED for v in cs.values]))
        fig.update_layout(**BBG_LAYOUT, title="Return Contribution", xaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    corr = returns_df.corr() if not returns_df.empty else pd.DataFrame()
    if corr.empty:
        st.info("No hay datos suficientes para correlación.")
    else:
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=[[0, "#1a0000"], [0.5, "#111"], [1, "#003d15"]]))
        fig.update_layout(**BBG_LAYOUT, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        if len(returns_df.columns) >= 2:
            fig2 = px.scatter_matrix(returns_df.iloc[:, : min(6, len(returns_df.columns))] * 100)
            fig2.update_layout(**BBG_LAYOUT, title="Scatter Matrix")
            st.plotly_chart(fig2, use_container_width=True)

with tabs[4]:
    if len(returns_df.columns) < 2:
        st.info("Necesitas mínimo 2 activos con datos para frontier.")
    else:
        n_sim = st.slider("Monte Carlo sims", 1000, 10000, 3000, 500)
        ef = efficient_frontier(returns_df, n=n_sim, rf=rf_rate)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.array(ef["mc_vol"]) * 100, y=np.array(ef["mc_ret"]) * 100, mode="markers", marker=dict(size=3, color=ef["mc_sr"], colorscale="Turbo"), name="Simulated"))
        ms = ef["max_sr_w"]
        mv = ef["min_vol_w"]
        ms_r, ms_v = ms @ ef["mu"].values, np.sqrt(ms @ ef["cov"].values @ ms)
        mv_r, mv_v = mv @ ef["mu"].values, np.sqrt(mv @ ef["cov"].values @ mv)
        fig.add_trace(go.Scatter(x=[ms_v * 100], y=[ms_r * 100], mode="markers", marker=dict(size=14, color=ORANGE, symbol="star"), name="Max Sharpe"))
        fig.add_trace(go.Scatter(x=[mv_v * 100], y=[mv_r * 100], mode="markers", marker=dict(size=12, color=BLUE), name="Min Vol"))
        fig.update_layout(**BBG_LAYOUT, title="Efficient Frontier", xaxis_title="Vol %", yaxis_title="Return %")
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    selected = st.multiselect("Escenarios", list(HISTORICAL_SCENARIOS.keys()), default=["COVID Crash (Feb–Mar 2020)", "2022 Global Bear Market"])
    out = []
    for sname in selected:
        if sname == "Custom Period":
            continue
        s, e = HISTORICAL_SCENARIOS[sname]
        sp = fetch_prices(all_tickers, s, e)
        if sp.empty:
            continue
        sdf = build_portfolio_value(sp[[c for c in sp.columns if c != bm_ticker] + ([bm_ticker] if bm_ticker in sp.columns else [])], positions, cash_df)
        if sdf.empty:
            continue
        sr = sdf["returns"].dropna()
        port_total = (1 + sr).prod() - 1 if not sr.empty else np.nan
        bm_total = (1 + sp[bm_ticker].pct_change().dropna()).prod() - 1 if bm_ticker in sp.columns else np.nan
        out.append({"Scenario": sname, "Portfolio": port_total, "Benchmark": bm_total, "Excess": port_total - bm_total if pd.notna(bm_total) else np.nan})
    if out:
        odf = pd.DataFrame(out)
        st.dataframe(odf.assign(Portfolio=odf["Portfolio"].map(fp), Benchmark=odf["Benchmark"].map(fp), Excess=odf["Excess"].map(fp)).set_index("Scenario"), use_container_width=True)

with tabs[6]:
    c1, c2 = st.columns(2)
    with c1:
        rs = rolling_sharpe(pr, roll_days)
        fig = go.Figure(go.Scatter(x=rs.index, y=rs.values, line=dict(color=ORANGE)))
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling Sharpe ({roll_days}d)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rv = pr.rolling(roll_days).std() * np.sqrt(252)
        fig = go.Figure(go.Scatter(x=rv.index, y=rv.values, line=dict(color=BLUE)))
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling Vol ({roll_days}d)", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

with tabs[7]:
    rows = []
    for p in positions:
        last = prices[p.ticker].iloc[-1] if p.ticker in prices.columns else np.nan
        rows.append({
            "Ticker": p.ticker,
            "Shares": p.shares,
            "Entry Date": p.entry_date,
            "Entry Px": p.entry_price,
            "Exit Date": p.exit_date,
            "Exit Px": p.exit_price,
            "Last Px": last,
            "Market Value": p.shares * last if pd.notna(last) else np.nan,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("#### Cash Movements")
    st.dataframe(cash_df if not cash_df.empty else pd.DataFrame(columns=["date", "amount", "note"]), use_container_width=True)

    export = prices.copy()
    export["asset_value"] = port_df["asset_value"]
    export["cash_balance"] = port_df["cash_balance"]
    export["total_value"] = port_df["total_value"]
    export["portfolio_return"] = port_df["returns"]
    st.download_button("⬇ Download CSV", export.to_csv().encode("utf-8"), f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
