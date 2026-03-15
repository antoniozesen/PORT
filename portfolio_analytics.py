"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PORTFOLIO ANALYTICS DASHBOARD  —  Bloomberg PORT-style             ║
║          Deploy: streamlit run portfolio_analytics.py                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  pip install streamlit yfinance pandas numpy plotly scipy                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize
from datetime import date, datetime
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PORT | Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BLOOMBERG TERMINAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #050505;
    color: #d4d4d4;
}
.stApp { background-color: #050505; }
.block-container { padding-top: 1rem; max-width: 100%; }

.bbg-topbar {
    background: #f5a623;
    padding: 8px 24px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #000;
    letter-spacing: 2px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
}
.m-card {
    background: #0d0d0d;
    border: 1px solid #1f1f1f;
    border-top: 2px solid #f5a623;
    padding: 14px 18px;
    margin-bottom: 4px;
}
.m-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; color: #555;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px;
}
.m-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px; font-weight: 600; line-height: 1.1;
}
.m-sub { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444; margin-top: 3px; }
.pos { color: #00e676; }
.neg { color: #ff3d3d; }
.neu { color: #d4d4d4; }
.orange { color: #f5a623; }
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #f5a623;
    letter-spacing: 2.5px; text-transform: uppercase;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 8px; margin: 24px 0 14px 0;
}
[data-testid="stSidebar"] { background: #080808 !important; border-right: 1px solid #141414; }
[data-testid="stSidebar"] label { color: #777 !important; font-size: 12px; }
[data-testid="stSidebar"] p { color: #666; font-size: 12px; }
[data-testid="stSidebar"] .stButton > button {
    background: #f5a623 !important; color: #000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important; font-size: 11px !important;
    letter-spacing: 1.5px !important; border: none !important;
    width: 100% !important; padding: 10px !important;
    border-radius: 2px !important; text-transform: uppercase;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1a1a1a; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 1.5px; color: #444; padding: 8px 16px;
    background: transparent; border-bottom: 2px solid transparent; text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: #f5a623 !important; border-bottom: 2px solid #f5a623 !important; background: transparent !important;
}
.stTextInput input, .stNumberInput input {
    background: #0d0d0d !important; color: #d4d4d4 !important;
    border: 1px solid #222 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important;
}
.stSelectbox > div > div { background: #0d0d0d !important; border: 1px solid #222 !important; color: #d4d4d4 !important; }
.stDateInput input { background: #0d0d0d !important; color: #d4d4d4 !important; font-family: 'IBM Plex Mono', monospace !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CATALOGUE — GLOBAL COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_CATALOGUE = {
    "🇺🇸 Equity — USA": {
        "S&P 500 (SPY)": "SPY", "S&P 500 iShares (IVV)": "IVV",
        "Nasdaq 100 (QQQ)": "QQQ", "Dow Jones (DIA)": "DIA",
        "Russell 2000 (IWM)": "IWM", "Russell 1000 Growth (IWF)": "IWF",
        "Russell 1000 Value (IWD)": "IWD", "S&P 500 Equal Weight (RSP)": "RSP",
        "US Total Market (VTI)": "VTI", "S&P 400 Mid Cap (MDY)": "MDY",
    },
    "🇪🇺 Equity — Europe": {
        "MSCI Europe (VGK)": "VGK", "MSCI Europe iShares (IEUR)": "IEUR",
        "EURO STOXX 50 (FEZ)": "FEZ", "Germany DAX (EWG)": "EWG",
        "France CAC 40 (EWQ)": "EWQ", "UK FTSE 100 (EWU)": "EWU",
        "Switzerland (EWL)": "EWL", "Netherlands (EWN)": "EWN",
        "Spain IBEX (EWP)": "EWP", "Italy (EWI)": "EWI",
        "Sweden (EWD)": "EWD", "Europe ex-UK (IEV)": "IEV",
        "Europe Small Cap (DFE)": "DFE", "Europe Value (EFV)": "EFV",
    },
    "🇯🇵 Equity — Japan": {
        "MSCI Japan (EWJ)": "EWJ", "Japan Hedged USD (DXJ)": "DXJ",
        "Japan Small Cap (SCJ)": "SCJ", "Japan TOPIX (BBJP)": "BBJP",
    },
    "🌏 Equity — Asia Pacific": {
        "MSCI Pacific ex-Japan (EPP)": "EPP", "Australia (EWA)": "EWA",
        "South Korea (EWY)": "EWY", "Hong Kong (EWH)": "EWH",
        "Singapore (EWS)": "EWS", "Taiwan (EWT)": "EWT",
        "India MSCI (INDA)": "INDA", "India Nifty 50 (INDY)": "INDY",
    },
    "🇨🇳 Equity — China": {
        "MSCI China (MCHI)": "MCHI", "China Large Cap (FXI)": "FXI",
        "China A-Shares (CNYA)": "CNYA", "China Tech (KWEB)": "KWEB",
    },
    "🌍 Equity — Emerging Markets": {
        "MSCI EM (EEM)": "EEM", "MSCI EM Vanguard (VWO)": "VWO",
        "EM ex-China (EMXC)": "EMXC", "EM Small Cap (EWX)": "EWX",
        "Latin America (ILF)": "ILF", "Brazil (EWZ)": "EWZ",
        "Mexico (EWW)": "EWW", "South Africa (EZA)": "EZA",
        "Turkey (TUR)": "TUR", "Saudi Arabia (KSA)": "KSA",
        "Indonesia (EIDO)": "EIDO", "Vietnam (VNM)": "VNM",
        "Poland (EPOL)": "EPOL", "MSCI Frontier (FM)": "FM",
    },
    "🌐 Equity — Global": {
        "MSCI ACWI (ACWI)": "ACWI", "MSCI World Developed (URTH)": "URTH",
        "All World Vanguard (VT)": "VT", "Intl Developed ex-US (EFA)": "EFA",
        "Intl Developed iShares (IDEV)": "IDEV",
        "Global ex-US Small Cap (VSS)": "VSS",
        "Global Min Volatility (ACWV)": "ACWV",
        "Global Quality (QUAL)": "QUAL", "Global Dividend (VYMI)": "VYMI",
    },
    "🏭 Equity — Sectors USA": {
        "Technology (XLK)": "XLK", "Financials (XLF)": "XLF",
        "Healthcare (XLV)": "XLV", "Energy (XLE)": "XLE",
        "Utilities (XLU)": "XLU", "Real Estate (XLRE)": "XLRE",
        "Consumer Staples (XLP)": "XLP", "Consumer Discret. (XLY)": "XLY",
        "Industrials (XLI)": "XLI", "Materials (XLB)": "XLB",
        "Communication (XLC)": "XLC", "Semiconductors (SOXX)": "SOXX",
        "Biotech (IBB)": "IBB", "Cybersecurity (CIBR)": "CIBR",
        "Cleantech (ICLN)": "ICLN", "Defence & Aerospace (ITA)": "ITA",
    },
    "💵 Fixed Income — USA": {
        "US Agg Bond (AGG)": "AGG", "Total Bond (BND)": "BND",
        "Short Treasury 1-3Y (SHY)": "SHY", "Mid Treasury 7-10Y (IEF)": "IEF",
        "Long Treasury 20Y+ (TLT)": "TLT", "Ultra-Long (EDV)": "EDV",
        "TIPS Inflation (TIP)": "TIP", "Corp Bond IG (LQD)": "LQD",
        "Corp Bond HY (HYG)": "HYG", "Corp Bond HY (JNK)": "JNK",
        "Muni Bond (MUB)": "MUB", "Floating Rate (FLOT)": "FLOT",
        "Senior Loans (BKLN)": "BKLN", "Convertible (CWB)": "CWB",
        "Cash / T-Bill (SGOV)": "SGOV",
    },
    "💶 Fixed Income — Europe": {
        "Euro Govt Bond (IBGE.L)": "IBGE.L", "Euro Corp Bond (IEAC.L)": "IEAC.L",
        "Euro HY (IHYG.L)": "IHYG.L", "Germany Bund (EXX6.DE)": "EXX6.DE",
        "Euro Inflation (IBCI.L)": "IBCI.L", "EUR Corp IG (LQDE.L)": "LQDE.L",
    },
    "🌐 Fixed Income — Global/EM": {
        "Global Bond Agg (BNDW)": "BNDW", "Intl Treasury ex-US (BWX)": "BWX",
        "EM Bond USD (EMB)": "EMB", "EM Bond Local (EMLC)": "EMLC",
        "EM Corp Bond (CEMB)": "CEMB",
    },
    "🪙 Commodities": {
        "Gold (GLD)": "GLD", "Gold iShares (IAU)": "IAU",
        "Silver (SLV)": "SLV", "Platinum (PPLT)": "PPLT",
        "Oil WTI (USO)": "USO", "Oil Brent (BNO)": "BNO",
        "Natural Gas (UNG)": "UNG", "Copper (CPER)": "CPER",
        "Agriculture (DBA)": "DBA", "Wheat (WEAT)": "WEAT",
        "Broad Commodities (PDBC)": "PDBC", "Uranium (URA)": "URA",
        "Lithium (LIT)": "LIT", "Rare Earth (REMX)": "REMX",
    },
    "🏢 Real Estate (REITs)": {
        "US REITs (VNQ)": "VNQ", "Global REITs (VNQI)": "VNQI",
        "Industrial REITs (INDS)": "INDS", "Data Centre REITs (SRVR)": "SRVR",
        "Residential REITs (REZ)": "REZ", "Mortgage REITs (REM)": "REM",
    },
    "⚖️ Multi-Asset & Alternatives": {
        "Balanced 60/40 (AOR)": "AOR", "Aggressive 80/20 (AOA)": "AOA",
        "Conservative 40/60 (AOK)": "AOK", "Risk Parity (RPAR)": "RPAR",
        "Cash T-Bill (BIL)": "BIL", "Volatility (VXX)": "VXX",
        "Managed Futures (DBMF)": "DBMF", "Infrastructure (IGF)": "IGF",
        "Dividend Growth (VIG)": "VIG", "High Dividend (HDV)": "HDV",
    },
    "🔬 Factor / Smart Beta": {
        "US Momentum (MTUM)": "MTUM", "US Quality (QUAL)": "QUAL",
        "US Min Volatility (USMV)": "USMV", "US Value (VTV)": "VTV",
        "Intl Momentum (IMTM)": "IMTM", "Intl Min Vol (EFAV)": "EFAV",
        "Intl Value (EFV)": "EFV", "Global Multi-Factor (ACWF)": "ACWF",
        "US ESG (ESGU)": "ESGU", "Global ESG (ESGV)": "ESGV",
    },
    "₿ Crypto": {
        "Bitcoin (BTC-USD)": "BTC-USD", "Ethereum (ETH-USD)": "ETH-USD",
        "Solana (SOL-USD)": "SOL-USD", "BNB (BNB-USD)": "BNB-USD",
        "XRP (XRP-USD)": "XRP-USD", "Bitcoin ETF (IBIT)": "IBIT",
        "Ethereum ETF (ETHA)": "ETHA",
    },
}

HISTORICAL_SCENARIOS = {
    "COVID Crash (Feb–Mar 2020)":         ("2020-01-15", "2020-03-23"),
    "COVID Recovery (Mar–Dec 2020)":      ("2020-03-23", "2020-12-31"),
    "2022 Global Bear Market":            ("2022-01-01", "2022-12-31"),
    "2022 Rate Hike Shock (Q1–Q3)":       ("2022-01-01", "2022-09-30"),
    "GFC Crisis (2008–2009)":             ("2008-01-01", "2009-03-09"),
    "GFC Recovery (2009–2010)":           ("2009-03-09", "2010-12-31"),
    "Dot-com Crash (2000–2002)":          ("2000-03-10", "2002-10-09"),
    "EM Taper Tantrum (2013)":            ("2013-05-01", "2013-09-30"),
    "China Devaluation Shock (2015)":     ("2015-08-01", "2015-09-30"),
    "Oil Crash H2 2014":                  ("2014-06-01", "2014-12-31"),
    "2018 Q4 US Selloff":                 ("2018-10-01", "2018-12-24"),
    "Brexit Vote (Jun 2016)":             ("2016-06-01", "2016-07-15"),
    "European Debt Crisis (2011)":        ("2011-01-01", "2011-12-31"),
    "Ukraine Invasion (Feb–Mar 2022)":    ("2022-02-24", "2022-03-15"),
    "SVB / Bank Crisis (Mar 2023)":       ("2023-03-08", "2023-03-31"),
    "Post-COVID Inflation Surge (2021)":  ("2021-01-01", "2021-12-31"),
    "Japan BOJ YCC Shock (Dec 2022)":     ("2022-12-15", "2022-12-31"),
    "Custom Period":                       None,
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
BBG_LAYOUT = dict(
    paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
    font=dict(family="IBM Plex Mono, monospace", color="#666", size=11),
    xaxis=dict(gridcolor="#141414", linecolor="#1f1f1f", zerolinecolor="#222",
               tickfont=dict(size=10, color="#555")),
    yaxis=dict(gridcolor="#141414", linecolor="#1f1f1f", zerolinecolor="#222",
               tickfont=dict(size=10, color="#555")),
    legend=dict(bgcolor="#0d0d0d", bordercolor="#1a1a1a", borderwidth=1,
                font=dict(size=10)),
    margin=dict(l=50, r=20, t=50, b=40),
    hovermode="x unified",
)
ORANGE = "#f5a623"; GREEN = "#00e676"; RED = "#ff3d3d"
BLUE = "#4fc3f7"; PURPLE = "#b39ddb"; YELLOW = "#fff176"; TEAL = "#80cbc4"
PALETTE = [ORANGE, BLUE, GREEN, PURPLE, RED, YELLOW, TEAL,
           "#ef9a9a", "#a5d6a7", "#ffe082", "#ce93d8", "#80deea", "#ffab91"]

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    tickers = list(tickers)
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]
        prices.dropna(how="all", inplace=True)
        missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
        for t in missing:
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if not d.empty:
                    prices[t] = d["Close"]
            except Exception:
                pass
        return prices
    except Exception:
        frames = {}
        for t in tickers:
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if not d.empty:
                    frames[t] = d["Close"]
            except Exception:
                pass
        return pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()


def compute_portfolio_returns(prices: pd.DataFrame, positions: list) -> pd.Series:
    if prices.empty or not positions:
        return pd.Series(dtype=float)
    daily_rets = prices.pct_change()
    port_rets = pd.Series(0.0, index=daily_rets.index)
    for dt in daily_rets.index:
        d = dt.date() if hasattr(dt, "date") else dt
        active = [p for p in positions
                  if p["ticker"] in daily_rets.columns
                  and not pd.isna(daily_rets.loc[dt, p["ticker"]])
                  and d >= p["buy_date"]
                  and (p["sell_date"] is None or d <= p["sell_date"])]
        if not active:
            continue
        total_w = sum(p["weight"] for p in active)
        if total_w == 0:
            continue
        port_rets[dt] = sum(
            (p["weight"] / total_w) * daily_rets.loc[dt, p["ticker"]]
            for p in active
        )
    return port_rets.replace([np.inf, -np.inf], np.nan).fillna(0)


def ann_return(r):
    n = len(r)
    if n < 2: return np.nan
    return (1 + r).prod() ** (TRADING_DAYS / n) - 1

def ann_vol(r):        return r.std() * np.sqrt(TRADING_DAYS)
def sharpe(r, rf=0.0): return (ann_return(r) - rf) / ann_vol(r) if ann_vol(r) > 0 else np.nan
def sortino(r, rf=0.0):
    down = r[r < 0]
    dv = down.std() * np.sqrt(TRADING_DAYS)
    return (ann_return(r) - rf) / dv if dv > 0 else np.nan

def max_drawdown(r):
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def drawdown_series(r):
    cum = (1 + r).cumprod()
    return (cum - cum.cummax()) / cum.cummax()

def calmar(r):
    dd = max_drawdown(r)
    return ann_return(r) / abs(dd) if dd != 0 else np.nan

def var_hist(r, c=0.95):  return float(np.percentile(r.dropna(), (1-c)*100))
def cvar_hist(r, c=0.95): v = var_hist(r, c); return float(r[r <= v].mean())
def var_para(r, c=0.95):  return float(r.mean() + r.std() * stats.norm.ppf(1-c))

def beta_alpha(pr, br, rf=0.0):
    a = pd.concat([pr, br], axis=1).dropna()
    if len(a) < 5: return np.nan, np.nan
    x = a.iloc[:,1].values - rf/TRADING_DAYS
    y = a.iloc[:,0].values - rf/TRADING_DAYS
    sl, ic, *_ = stats.linregress(x, y)
    return sl, ic * TRADING_DAYS

def tracking_error(pr, br):
    return (pr - br).dropna().std() * np.sqrt(TRADING_DAYS)

def info_ratio(pr, br):
    te = tracking_error(pr, br)
    return (ann_return(pr) - ann_return(br)) / te if te > 0 else np.nan

def rolling_sharpe(r, w=63):
    return (r.rolling(w).mean() * TRADING_DAYS) / (r.rolling(w).std() * np.sqrt(TRADING_DAYS))

def up_down_capture(pr, br):
    a = pd.concat([pr, br], axis=1).dropna(); a.columns = ["p","b"]
    up = a[a["b"] > 0]; dn = a[a["b"] < 0]
    uc = ann_return(up["p"]) / ann_return(up["b"]) if len(up) > 5 else np.nan
    dc = ann_return(dn["p"]) / ann_return(dn["b"]) if len(dn) > 5 else np.nan
    return uc, dc


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENT FRONTIER
# ─────────────────────────────────────────────────────────────────────────────
def efficient_frontier(returns_df: pd.DataFrame, n=5000, rf=0.02):
    mu  = returns_df.mean() * TRADING_DAYS
    cov = returns_df.cov()  * TRADING_DAYS
    na  = len(mu)
    mc_ret, mc_vol, mc_sr, mc_w = [], [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(na))
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        mc_ret.append(r); mc_vol.append(v)
        mc_sr.append((r - rf) / v); mc_w.append(w)
    bounds = tuple((0, 1) for _ in range(na))
    eq_con = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opt_ret, opt_vol = [], []
    for tr in np.linspace(min(mc_ret), max(mc_ret), 60):
        cons = [eq_con, {"type": "eq", "fun": lambda w, t=tr: w @ mu.values - t}]
        res = minimize(lambda w: np.sqrt(w @ cov.values @ w),
                       x0=np.ones(na)/na, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol":1e-10,"maxiter":1000})
        if res.success:
            opt_vol.append(res.fun); opt_ret.append(tr)
    def neg_sr(w):
        r = w @ mu.values; v = np.sqrt(w @ cov.values @ w)
        return -(r - rf) / v
    r_ms = minimize(neg_sr, x0=np.ones(na)/na, method="SLSQP",
                    bounds=bounds, constraints=eq_con, options={"ftol":1e-10,"maxiter":1000})
    r_mv = minimize(lambda w: np.sqrt(w @ cov.values @ w),
                    x0=np.ones(na)/na, method="SLSQP",
                    bounds=bounds, constraints=eq_con, options={"ftol":1e-10,"maxiter":1000})
    return dict(mc_ret=mc_ret, mc_vol=mc_vol, mc_sr=mc_sr,
                opt_ret=opt_ret, opt_vol=opt_vol,
                max_sr_w=r_ms.x if r_ms.success else np.ones(na)/na,
                min_vol_w=r_mv.x if r_mv.success else np.ones(na)/na,
                tickers=list(returns_df.columns), mu=mu, cov=cov)


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fp(v, d=2):
    return "—" if pd.isna(v) else f"{v*100:.{d}f}%"
def fn(v, d=2):
    return "—" if pd.isna(v) else f"{v:.{d}f}"
def mc(label, value, sub="", css="neu"):
    return (f'<div class="m-card"><div class="m-label">{label}</div>'
            f'<div class="m-value {css}">{value}</div>'
            + (f'<div class="m-sub">{sub}</div>' if sub else "")
            + '</div>')


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
                'font-weight:700;color:#f5a623;letter-spacing:2px;margin-bottom:16px;">'
                '▶ PORT ANALYTICS</div>', unsafe_allow_html=True)

    st.markdown("##### Portfolio Positions")
    n_pos = st.number_input("Number of positions", 1, 30, 3, 1)

    positions = []
    defaults = [("AAPL", 33.3), ("MSFT", 33.3), ("GOOGL", 33.4)]

    with st.expander("📋 Enter Positions", expanded=True):
        for i in range(int(n_pos)):
            st.markdown(f"**Position {i+1}**")
            c = st.columns([2, 1])
            ticker = c[0].text_input(f"Ticker {i+1}",
                                     value=defaults[i][0] if i < 3 else "",
                                     key=f"tk_{i}").upper().strip()
            weight = c[1].number_input(f"Weight % {i+1}",
                                       min_value=0.0, max_value=100.0,
                                       value=defaults[i][1] if i < 3 else round(100/n_pos, 1),
                                       step=0.5, key=f"wt_{i}")
            c2 = st.columns(2)
            buy_date  = c2[0].date_input(f"Buy {i+1}",  date(2022, 1, 1), key=f"bd_{i}")
            sell_date = c2[1].date_input(f"Sell {i+1}", date.today(),      key=f"sd_{i}")
            holding   = st.checkbox(f"Still holding #{i+1}", True, key=f"sh_{i}")
            if ticker:
                positions.append({
                    "ticker":    ticker,
                    "weight":    weight,
                    "buy_date":  buy_date,
                    "sell_date": None if holding else sell_date,
                })
            st.markdown("---")

    st.markdown("##### Benchmark")
    bm_group  = st.selectbox("Category", list(BENCHMARK_CATALOGUE.keys()))
    bm_name   = st.selectbox("Benchmark", list(BENCHMARK_CATALOGUE[bm_group].keys()))
    bm_ticker = BENCHMARK_CATALOGUE[bm_group][bm_name]
    custom_bm = st.text_input("Custom ticker (overrides above)", "")
    if custom_bm.strip():
        bm_ticker = custom_bm.strip().upper()

    st.markdown("##### Analysis Period")
    min_buy    = min((p["buy_date"] for p in positions), default=date(2020, 1, 1))
    start_date = st.date_input("Start", min_buy)
    end_date   = st.date_input("End",   date.today())

    st.markdown("##### Parameters")
    rf_rate   = st.number_input("Risk-free rate (annual %)", 0.0, 20.0, 4.5, 0.1) / 100
    var_conf  = st.selectbox("VaR confidence", ["95%", "99%", "90%"])
    var_conf_f = float(var_conf.replace("%", "")) / 100
    roll_days = int(st.selectbox("Rolling window", ["21d","63d","126d","252d"], 1).replace("d",""))

    st.markdown("---")
    run = st.button("⚡  RUN ANALYSIS")

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="bbg-topbar">'
    f'<span>PORT &nbsp;|&nbsp; PORTFOLIO ANALYTICS TERMINAL</span>'
    f'<span>{datetime.now().strftime("%d %b %Y  %H:%M")} &nbsp;|&nbsp; '
    f'{len([p for p in positions if p["ticker"]])} POS &nbsp;|&nbsp; BM: {bm_ticker}</span>'
    f'</div>', unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <div style="text-align:center;padding:100px 0;color:#333;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:48px;font-weight:700;
                  color:#1a1a1a;letter-spacing:6px;">PORT</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#2a2a2a;
                  letter-spacing:3px;margin-top:10px;">
        ENTER POSITIONS IN THE SIDEBAR AND CLICK ⚡ RUN ANALYSIS
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────────────────────────
positions = [p for p in positions if p["ticker"]]
if not positions:
    st.error("Add at least one position with a valid ticker.")
    st.stop()

total_w = sum(p["weight"] for p in positions)
if total_w == 0:
    st.error("Total weight is 0.")
    st.stop()
if abs(total_w - 100) > 0.5:
    st.warning(f"Weights sum to {total_w:.1f}% — they will be normalised to 100%.")

all_tickers = tuple(set([p["ticker"] for p in positions] + [bm_ticker]))
with st.spinner("📡  Fetching market data..."):
    prices = fetch_prices(all_tickers, str(start_date), str(end_date))

if prices.empty:
    st.error("Could not fetch price data. Check tickers and connection.")
    st.stop()

# Drop positions with no data
missing = [p["ticker"] for p in positions if p["ticker"] not in prices.columns]
if missing:
    st.warning(f"No data found for: {', '.join(missing)}. Excluded.")
    positions = [p for p in positions if p["ticker"] not in missing]

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RETURNS
# ─────────────────────────────────────────────────────────────────────────────
port_rets = compute_portfolio_returns(prices, positions)
bm_rets   = prices[bm_ticker].pct_change().dropna() if bm_ticker in prices.columns else None

if bm_rets is not None:
    idx = port_rets.index.intersection(bm_rets.index)
    pr  = port_rets.loc[idx]
    br  = bm_rets.loc[idx]
else:
    pr = port_rets; br = None

asset_rets = {p["ticker"]: prices[p["ticker"]].pct_change().dropna()
              for p in positions if p["ticker"] in prices.columns}
returns_df = pd.DataFrame(asset_rets).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
P = {}
P["ann_ret"] = ann_return(pr)
P["ann_vol"] = ann_vol(pr)
P["sharpe"]  = sharpe(pr, rf_rate)
P["sortino"] = sortino(pr, rf_rate)
P["calmar"]  = calmar(pr)
P["mdd"]     = max_drawdown(pr)
P["var"]     = var_hist(pr, var_conf_f)
P["cvar"]    = cvar_hist(pr, var_conf_f)
P["var_p"]   = var_para(pr, var_conf_f)
P["skew"]    = float(stats.skew(pr.dropna()))
P["kurt"]    = float(stats.kurtosis(pr.dropna()))
P["cum"]     = (1 + pr).cumprod()

if br is not None:
    B = {}
    B["ann_ret"] = ann_return(br)
    B["ann_vol"] = ann_vol(br)
    B["sharpe"]  = sharpe(br, rf_rate)
    B["mdd"]     = max_drawdown(br)
    B["cum"]     = (1 + br).cumprod()
    P["beta"], P["alpha"] = beta_alpha(pr, br, rf_rate)
    P["te"]  = tracking_error(pr, br)
    P["ir"]  = info_ratio(pr, br)
    P["uc"], P["dc"] = up_down_capture(pr, br)
else:
    B = None
    P["beta"] = P["alpha"] = P["te"] = P["ir"] = P["uc"] = P["dc"] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈  OVERVIEW",
    "⚠️  RISK",
    "📊  ATTRIBUTION",
    "🔗  CORRELATION",
    "🎯  EFFICIENT FRONTIER",
    "📉  SCENARIOS",
    "🔄  ROLLING",
    "📋  POSITIONS",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-title">Key Performance Metrics — Portfolio</div>', unsafe_allow_html=True)
    cols = st.columns(8)
    mdef = [
        ("Ann. Return",      fp(P["ann_ret"]),  "portfolio",     "pos" if P["ann_ret"] > 0 else "neg"),
        ("Ann. Volatility",  fp(P["ann_vol"]),  "annualised",    "neu"),
        ("Sharpe Ratio",     fn(P["sharpe"]),   f"rf={rf_rate*100:.1f}%",
         "pos" if (P["sharpe"] or 0) > 1 else ("orange" if (P["sharpe"] or 0) > 0 else "neg")),
        ("Sortino Ratio",    fn(P["sortino"]),  "downside-adj",
         "pos" if (P["sortino"] or 0) > 1 else "neu"),
        ("Calmar Ratio",     fn(P["calmar"]),   "ret / MDD",
         "pos" if (P["calmar"] or 0) > 0 else "neg"),
        ("Max Drawdown",     fp(P["mdd"]),      "peak-to-trough","neg"),
        (f"VaR {var_conf}",  fp(P["var"]),      "1-day hist.",   "neg"),
        (f"CVaR {var_conf}", fp(P["cvar"]),     "expected tail", "neg"),
    ]
    for col, (l, v, s, c) in zip(cols, mdef):
        col.markdown(mc(l, v, s, c), unsafe_allow_html=True)

    if B:
        st.markdown(f'<div class="sec-title">vs Benchmark — {bm_ticker}</div>', unsafe_allow_html=True)
        cols2 = st.columns(8)
        excess = P["ann_ret"] - B["ann_ret"]
        bmdef = [
            ("BM Ann. Return",  fp(B["ann_ret"]),  bm_ticker, "pos" if B["ann_ret"] > 0 else "neg"),
            ("BM Volatility",   fp(B["ann_vol"]),  bm_ticker, "neu"),
            ("BM Sharpe",       fn(B["sharpe"]),   bm_ticker, "pos" if (B["sharpe"] or 0) > 1 else "neu"),
            ("BM Max DD",       fp(B["mdd"]),      bm_ticker, "neg"),
            ("Excess Return",   fp(excess),        "port − bm","pos" if excess > 0 else "neg"),
            ("Beta",            fn(P["beta"]),     "vs bm",   "neu"),
            ("Alpha (ann.)",    fp(P["alpha"]),    "Jensen",  "pos" if (P["alpha"] or 0) > 0 else "neg"),
            ("Info Ratio",      fn(P["ir"]),       "active/TE","pos" if (P["ir"] or 0) > 0 else "neg"),
        ]
        for col, (l, v, s, c) in zip(cols2, bmdef):
            col.markdown(mc(l, v, s, c), unsafe_allow_html=True)

    # Cumulative chart
    st.markdown('<div class="sec-title">Cumulative Performance</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P["cum"].index, y=P["cum"].values,
                             name="Portfolio", line=dict(color=ORANGE, width=2.5)))
    if B:
        fig.add_trace(go.Scatter(x=B["cum"].index, y=B["cum"].values,
                                 name=bm_ticker, line=dict(color=BLUE, width=1.5, dash="dot")))
    fig.update_layout(**BBG_LAYOUT, title="Cumulative Return (Base = 1.0)",
                      yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

    # Individual holdings
    st.markdown('<div class="sec-title">Individual Holdings</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for i, (tk, r) in enumerate(asset_rets.items()):
        cum = (1 + r).cumprod()
        fig2.add_trace(go.Scatter(x=cum.index, y=cum.values, name=tk,
                                  line=dict(color=PALETTE[i % len(PALETTE)], width=1.2)))
    fig2.add_trace(go.Scatter(x=P["cum"].index, y=P["cum"].values,
                              name="Portfolio", line=dict(color=ORANGE, width=2.5)))
    fig2.update_layout(**BBG_LAYOUT, title="Individual Holdings — Cumulative Return")
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-title">Underwater / Drawdown</div>', unsafe_allow_html=True)
        dd = drawdown_series(pr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
                                 fill="tozeroy", fillcolor="rgba(255,61,61,0.15)",
                                 line=dict(color=RED, width=1.5), name="Portfolio DD"))
        if br is not None:
            bm_dd = drawdown_series(br)
            fig.add_trace(go.Scatter(x=bm_dd.index, y=bm_dd.values,
                                     line=dict(color=BLUE, width=1, dash="dot"), name=bm_ticker))
        fig.update_layout(**BBG_LAYOUT, title="Underwater Chart", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-title">Return Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pr*100, nbinsx=80,
                                   marker_color=ORANGE, opacity=0.75, name="Returns"))
        fig.add_vline(x=P["var"]*100,  line_color=RED,    line_dash="dash",
                      annotation_text=f"VaR {var_conf}")
        fig.add_vline(x=P["cvar"]*100, line_color=PURPLE, line_dash="dot",
                      annotation_text=f"CVaR {var_conf}")
        fig.update_layout(**BBG_LAYOUT, title="Daily Return Distribution",
                          xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Full risk table
    st.markdown('<div class="sec-title">Full Risk Metrics</div>', unsafe_allow_html=True)
    rows = [
        ("Annualised Return",    fp(P["ann_ret"]),   fp(B["ann_ret"]) if B else "—"),
        ("Annualised Volatility",fp(P["ann_vol"]),   fp(B["ann_vol"]) if B else "—"),
        ("Sharpe Ratio",         fn(P["sharpe"]),    fn(B["sharpe"])  if B else "—"),
        ("Sortino Ratio",        fn(P["sortino"]),   "—"),
        ("Calmar Ratio",         fn(P["calmar"]),    "—"),
        ("Max Drawdown",         fp(P["mdd"]),       fp(B["mdd"])     if B else "—"),
        (f"VaR {var_conf} hist.",fp(P["var"]),       fp(var_hist(br, var_conf_f)) if br is not None else "—"),
        (f"CVaR {var_conf}",     fp(P["cvar"]),      "—"),
        ("VaR Parametric",       fp(P["var_p"]),     "—"),
        ("Skewness",             fn(P["skew"]),      "—"),
        ("Excess Kurtosis",      fn(P["kurt"]),      "—"),
        ("Beta",                 fn(P["beta"]),      "1.00"),
        ("Alpha (ann.)",         fp(P["alpha"]),     "0.00%"),
        ("Tracking Error",       fp(P["te"]),        "—"),
        ("Information Ratio",    fn(P["ir"]),        "—"),
        ("Up Capture",           fp(P["uc"]),        "—"),
        ("Down Capture",         fp(P["dc"]),        "—"),
    ]
    rdf = pd.DataFrame(rows, columns=["Metric","Portfolio","Benchmark"]).set_index("Metric")
    st.dataframe(rdf, use_container_width=True, height=560)

    # Annual bar chart
    st.markdown('<div class="sec-title">Annual Returns</div>', unsafe_allow_html=True)
    annual = pr.resample("YE").apply(lambda r: (1+r).prod()-1)
    annual.index = annual.index.year
    fig = go.Figure(go.Bar(x=annual.index.astype(str), y=annual.values,
                           marker_color=[GREEN if v > 0 else RED for v in annual.values],
                           text=[f"{v*100:.1f}%" for v in annual.values], textposition="outside"))
    if br is not None:
        bm_ann = br.resample("YE").apply(lambda r: (1+r).prod()-1)
        bm_ann.index = bm_ann.index.year
        fig.add_trace(go.Scatter(x=bm_ann.index.astype(str), y=bm_ann.values,
                                 name=bm_ticker, mode="lines+markers",
                                 line=dict(color=BLUE, width=2), marker=dict(size=6)))
    fig.update_layout(**BBG_LAYOUT, title="Annual Return by Year", yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-title">Allocation & Return Attribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        labels  = [p["ticker"] for p in positions]
        weights = [p["weight"] / total_w * 100 for p in positions]
        fig = go.Figure(go.Pie(
            labels=labels, values=weights, hole=0.55,
            marker=dict(colors=PALETTE[:len(labels)], line=dict(color="#050505", width=2)),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig.update_layout(**BBG_LAYOUT, title="Weight Allocation", showlegend=False,
                          annotations=[dict(text="ALLOC", x=0.5, y=0.5, showarrow=False,
                                            font=dict(size=12, color=ORANGE, family="IBM Plex Mono"))])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        contrib = {p["ticker"]: (p["weight"] / total_w) * ann_return(asset_rets[p["ticker"]])
                   for p in positions if p["ticker"] in asset_rets}
        cs = pd.Series(contrib).sort_values()
        fig = go.Figure(go.Bar(
            x=cs.values, y=cs.index, orientation="h",
            marker_color=[GREEN if v > 0 else RED for v in cs.values],
            text=[fp(v) for v in cs.values], textposition="outside",
        ))
        fig.update_layout(**BBG_LAYOUT, title="Contribution to Portfolio Return (weighted)",
                          xaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    # Per-asset stats
    st.markdown('<div class="sec-title">Individual Asset Statistics</div>', unsafe_allow_html=True)
    rows = []
    for p in positions:
        tk = p["ticker"]
        if tk not in asset_rets: continue
        r = asset_rets[tk]
        b_, a_ = beta_alpha(r, br, rf_rate) if br is not None else (np.nan, np.nan)
        rows.append({
            "Ticker": tk, "Weight %": f"{p['weight']/total_w*100:.1f}%",
            "Ann. Return": fp(ann_return(r)), "Ann. Vol": fp(ann_vol(r)),
            "Sharpe": fn(sharpe(r, rf_rate)), "Max DD": fp(max_drawdown(r)),
            f"VaR {var_conf}": fp(var_hist(r, var_conf_f)),
            "Beta": fn(b_), "Alpha (ann)": fp(a_),
            "Buy Date": str(p["buy_date"]),
            "Sell Date": str(p["sell_date"]) if p["sell_date"] else "Holding",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-title">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#1a0000"],[0.25,"#ff3d3d"],[0.5,"#0d0d0d"],[0.75,"#004d20"],[1,"#00e676"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=11, family="IBM Plex Mono"),
        colorbar=dict(title="ρ", tickfont=dict(size=10)),
    ))
    fig.update_layout(**BBG_LAYOUT, title="Pairwise Return Correlation",
                      height=max(300, len(corr)*60+100))
    st.plotly_chart(fig, use_container_width=True)

    if br is not None:
        st.markdown(f'<div class="sec-title">Rolling Correlation vs {bm_ticker}</div>', unsafe_allow_html=True)
        rc = pr.rolling(roll_days).corr(br)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc.index, y=rc.values,
                                 line=dict(color=ORANGE, width=2), name=f"{roll_days}d corr"))
        fig.add_hline(y=0, line_color="#333", line_dash="dot")
        fig.update_layout(**BBG_LAYOUT,
                          title=f"Rolling {roll_days}d Correlation vs {bm_ticker}",
                          yaxis=dict(range=[-1.1, 1.1]))
        st.plotly_chart(fig, use_container_width=True)

    if len(returns_df.columns) >= 2:
        st.markdown('<div class="sec-title">Pairwise Return Scatter</div>', unsafe_allow_html=True)
        tks = returns_df.columns[:6].tolist()
        fig = px.scatter_matrix(returns_df[tks]*100, dimensions=tks,
                                color_discrete_sequence=[ORANGE])
        fig.update_traces(diagonal_visible=False, marker=dict(size=2, opacity=0.4))
        fig.update_layout(**BBG_LAYOUT, title="Pairwise Scatter Matrix", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-title">Mean-Variance Efficient Frontier</div>', unsafe_allow_html=True)
    if len(returns_df.columns) < 2:
        st.info("Need ≥ 2 assets to compute the Efficient Frontier.")
    else:
        n_sim = st.slider("Monte Carlo simulations", 1000, 15000, 5000, 500)
        with st.spinner("Optimising..."):
            ef = efficient_frontier(returns_df, n=n_sim, rf=rf_rate)

        fig = go.Figure()
        # Random portfolios
        fig.add_trace(go.Scatter(
            x=[v*100 for v in ef["mc_vol"]], y=[r*100 for r in ef["mc_ret"]],
            mode="markers",
            marker=dict(size=3, opacity=0.5, color=ef["mc_sr"],
                        colorscale=[[0,"#1a0a00"],[0.5,"#7a3c00"],[1,"#f5a623"]],
                        colorbar=dict(title="Sharpe", x=1.02, thickness=10,
                                      tickfont=dict(size=9)), showscale=True),
            name="Simulated", hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
        ))
        # Efficient frontier line
        if ef["opt_ret"]:
            fig.add_trace(go.Scatter(
                x=[v*100 for v in ef["opt_vol"]], y=[r*100 for r in ef["opt_ret"]],
                mode="lines", line=dict(color=GREEN, width=2.5), name="Efficient Frontier"))
        # Max Sharpe
        ms_w = ef["max_sr_w"]
        ms_r = ms_w @ ef["mu"].values; ms_v = np.sqrt(ms_w @ ef["cov"].values @ ms_w)
        fig.add_trace(go.Scatter(x=[ms_v*100], y=[ms_r*100], mode="markers+text",
                                 marker=dict(color=ORANGE, size=14, symbol="star"),
                                 text=["Max Sharpe"], textposition="top center",
                                 textfont=dict(size=10, color=ORANGE), name="Max Sharpe"))
        # Min Vol
        mv_w = ef["min_vol_w"]
        mv_r = mv_w @ ef["mu"].values; mv_v = np.sqrt(mv_w @ ef["cov"].values @ mv_w)
        fig.add_trace(go.Scatter(x=[mv_v*100], y=[mv_r*100], mode="markers+text",
                                 marker=dict(color=BLUE, size=12, symbol="diamond"),
                                 text=["Min Vol"], textposition="top center",
                                 textfont=dict(size=10, color=BLUE), name="Min Vol"))
        # Individual assets
        for i, tk in enumerate(ef["tickers"]):
            fig.add_trace(go.Scatter(
                x=[np.sqrt(ef["cov"].loc[tk,tk])*100], y=[ef["mu"][tk]*100],
                mode="markers+text", marker=dict(color=PALETTE[i%len(PALETTE)], size=10),
                text=[tk], textposition="top right",
                textfont=dict(size=9, color=PALETTE[i%len(PALETTE)]), name=tk))
        # Current portfolio
        port_tks = [p["ticker"] for p in positions if p["ticker"] in ef["tickers"]]
        if len(port_tks) == len(ef["tickers"]):
            cw = np.array([p["weight"]/total_w for p in positions if p["ticker"] in ef["tickers"]])
            cr = cw @ ef["mu"].values; cv = np.sqrt(cw @ ef["cov"].values @ cw)
            fig.add_trace(go.Scatter(x=[cv*100], y=[cr*100], mode="markers+text",
                                     marker=dict(color=RED, size=14, symbol="x"),
                                     text=["Current Portfolio"], textposition="bottom right",
                                     textfont=dict(size=10, color=RED), name="Current"))

        fig.update_layout(**BBG_LAYOUT, title="Efficient Frontier — Mean-Variance",
                          xaxis_title="Annualised Volatility (%)",
                          yaxis_title="Annualised Return (%)", height=550)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-title">Max Sharpe Weights</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Ticker": ef["tickers"],
                "Weight": [f"{w*100:.2f}%" for w in ef["max_sr_w"]],
                "Exp. Return": [fp(ef["mu"][t]) for t in ef["tickers"]],
            }).set_index("Ticker"), use_container_width=True)
        with c2:
            st.markdown('<div class="sec-title">Min Volatility Weights</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Ticker": ef["tickers"],
                "Weight": [f"{w*100:.2f}%" for w in ef["min_vol_w"]],
                "Ann. Vol": [fp(np.sqrt(ef["cov"].loc[t,t])) for t in ef["tickers"]],
            }).set_index("Ticker"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-title">Historical Stress Scenarios</div>', unsafe_allow_html=True)
    selected = st.multiselect("Select scenarios", list(HISTORICAL_SCENARIOS.keys()),
                              default=["COVID Crash (Feb–Mar 2020)",
                                       "2022 Global Bear Market",
                                       "GFC Crisis (2008–2009)"])
    custom_s = custom_e = None
    if "Custom Period" in selected:
        cc = st.columns(2)
        custom_s = cc[0].date_input("Custom start", date(2020,1,1), key="cs")
        custom_e = cc[1].date_input("Custom end",   date(2020,6,1), key="ce")

    results = []
    for scen in selected:
        if scen == "Custom Period":
            if not custom_s or not custom_e: continue
            s, e, label = str(custom_s), str(custom_e), f"Custom ({custom_s}→{custom_e})"
        else:
            s, e = HISTORICAL_SCENARIOS[scen]; label = scen

        sp = fetch_prices(all_tickers, s, e)
        if sp.empty:
            results.append({"Scenario": label[:40], "Portfolio": "No data",
                             "Max DD": "—", "Volatility": "—", "Benchmark": "—", "Excess": "—"})
            continue
        sr = compute_portfolio_returns(sp, positions)
        if sr.empty:
            results.append({"Scenario": label[:40], "Portfolio": "No data",
                             "Max DD": "—", "Volatility": "—", "Benchmark": "—", "Excess": "—"})
            continue
        pt = (1 + sr).prod() - 1
        bmt = np.nan
        if bm_ticker in sp.columns:
            bm_r = sp[bm_ticker].pct_change().dropna()
            if not bm_r.empty:
                bmt = (1 + bm_r).prod() - 1
        excess_s = pt - bmt if not np.isnan(bmt) else np.nan
        results.append({
            "Scenario":  label[:40],
            "Portfolio": fp(pt),
            "Max DD":    fp(max_drawdown(sr)),
            "Volatility":fp(ann_vol(sr)),
            "Benchmark": fp(bmt) if not np.isnan(bmt) else "—",
            "Excess":    fp(excess_s) if not np.isnan(excess_s) else "—",
        })

    if results:
        st.dataframe(pd.DataFrame(results).set_index("Scenario"), use_container_width=True)
        vals, labs, bvals = [], [], []
        for r in results:
            try:
                vals.append(float(r["Portfolio"].replace("%",""))/100)
                labs.append(r["Scenario"][:35])
                try: bvals.append(float(r["Benchmark"].replace("%",""))/100)
                except: bvals.append(None)
            except: pass
        if vals:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labs, y=vals, name="Portfolio",
                                 marker_color=[GREEN if v > 0 else RED for v in vals],
                                 text=[fp(v) for v in vals], textposition="outside"))
            if any(v is not None for v in bvals):
                fig.add_trace(go.Bar(x=labs, y=[v or 0 for v in bvals], name=bm_ticker,
                                     marker_color=[BLUE if (v or 0) > 0 else PURPLE for v in bvals],
                                     opacity=0.6, text=[fp(v) if v is not None else "" for v in bvals],
                                     textposition="outside"))
            fig.update_layout(**BBG_LAYOUT, title="Total Return by Historical Scenario",
                              yaxis_tickformat=".1%", barmode="group",
                              xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — ROLLING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-title">Rolling Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        rs = rolling_sharpe(pr, roll_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs.index, y=rs.values, line=dict(color=ORANGE, width=2),
                                 fill="tozeroy", fillcolor="rgba(245,166,35,0.08)", name="Sharpe"))
        fig.add_hline(y=0, line_color=RED, line_dash="dot")
        fig.add_hline(y=1, line_color=GREEN, line_dash="dot", annotation_text="Sharpe=1")
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling {roll_days}d Sharpe")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rvol = pr.rolling(roll_days).std() * np.sqrt(TRADING_DAYS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rvol.index, y=rvol.values, line=dict(color=BLUE, width=2),
                                 fill="tozeroy", fillcolor="rgba(79,195,247,0.08)", name="Vol"))
        if br is not None:
            bm_rvol = br.rolling(roll_days).std() * np.sqrt(TRADING_DAYS)
            fig.add_trace(go.Scatter(x=bm_rvol.index, y=bm_rvol.values,
                                     line=dict(color=PURPLE, width=1.5, dash="dot"), name="BM Vol"))
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling {roll_days}d Annualised Volatility",
                          yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        rret = pr.rolling(roll_days).apply(lambda r: (1+r).prod()-1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rret.index, y=rret.values, line=dict(color=GREEN, width=2),
                                 fill="tozeroy", fillcolor="rgba(0,230,118,0.06)", name="Return"))
        fig.add_hline(y=0, line_color=RED, line_dash="dot")
        fig.update_layout(**BBG_LAYOUT, title=f"Rolling {roll_days}d Return",
                          yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        if br is not None:
            rbeta = (pr.rolling(roll_days).corr(br) *
                     pr.rolling(roll_days).std() / br.rolling(roll_days).std())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rbeta.index, y=rbeta.values,
                                     line=dict(color=PURPLE, width=2), name="Beta"))
            fig.add_hline(y=1, line_color="#333", line_dash="dot", annotation_text="β=1")
            fig.update_layout(**BBG_LAYOUT, title=f"Rolling {roll_days}d Beta vs {bm_ticker}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No benchmark for beta.")

    # Monthly heatmap
    st.markdown('<div class="sec-title">Monthly Return Heatmap</div>', unsafe_allow_html=True)
    monthly = pr.resample("ME").apply(lambda r: (1+r).prod()-1)
    mdf = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month,
                         "Return": monthly.values})
    if not mdf.empty:
        pivot = mdf.pivot(index="Year", columns="Month", values="Return")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure(go.Heatmap(
            z=pivot.values*100, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#3d0000"],[0.5,"#111"],[1,"#003d15"]],
            text=np.round(pivot.values*100, 1), texttemplate="%{text}%",
            textfont=dict(size=9, family="IBM Plex Mono"),
            colorbar=dict(title="%", tickfont=dict(size=9)),
        ))
        fig.update_layout(**BBG_LAYOUT, title="Monthly Return Heatmap (%)",
                          xaxis=dict(side="top"), height=max(200, len(pivot)*40+100))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sec-title">Position Detail</div>', unsafe_allow_html=True)
    rows = []
    for p in positions:
        tk = p["ticker"]
        lp = prices[tk].iloc[-1] if tk in prices.columns else np.nan
        fp0 = prices[tk].iloc[0] if tk in prices.columns else np.nan
        hr = (lp / fp0 - 1) if not np.isnan(fp0) and fp0 > 0 else np.nan
        rows.append({
            "Ticker": tk, "Weight %": f"{p['weight']/total_w*100:.2f}%",
            "Buy Date": str(p["buy_date"]),
            "Sell Date": str(p["sell_date"]) if p["sell_date"] else "—",
            "Status": "Closed" if p["sell_date"] else "Active",
            "First Price": f"${fp0:.2f}" if not np.isnan(fp0) else "—",
            "Last Price":  f"${lp:.2f}"  if not np.isnan(lp)  else "—",
            "Period Return": fp(hr) if not np.isnan(hr) else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    # Normalised price chart
    st.markdown('<div class="sec-title">Normalised Price Chart (Base = 100)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for i, p in enumerate(positions):
        tk = p["ticker"]
        if tk in prices.columns:
            norm = prices[tk] / prices[tk].iloc[0] * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=tk,
                                     line=dict(color=PALETTE[i%len(PALETTE)], width=1.5)))
    if bm_ticker in prices.columns:
        norm_bm = prices[bm_ticker] / prices[bm_ticker].iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm_bm.index, y=norm_bm.values, name=bm_ticker,
                                 line=dict(color="#555", width=1.5, dash="dot")))
    fig.update_layout(**BBG_LAYOUT, title="Normalised Price (Base = 100)")
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown('<div class="sec-title">Export</div>', unsafe_allow_html=True)
    export = prices.copy()
    export["Portfolio_Return"] = port_rets
    st.download_button("⬇  Download CSV", export.to_csv().encode("utf-8"),
                       f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
