from __future__ import annotations

TRADING_DAYS = 252
ORANGE = "#f5a623"
GREEN = "#00e676"
RED = "#ff3d3d"
BLUE = "#4fc3f7"
PURPLE = "#b39ddb"
YELLOW = "#fff176"
TEAL = "#80cbc4"
PALETTE = [ORANGE, BLUE, GREEN, PURPLE, RED, YELLOW, TEAL, "#ef9a9a", "#a5d6a7", "#ffe082", "#ce93d8", "#80deea"]

BENCHMARK_CATALOGUE = {
    "🇺🇸 Equity — USA": {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Dow Jones (DIA)": "DIA", "Russell 2000 (IWM)": "IWM", "US Total Market (VTI)": "VTI"},
    "🌐 Equity — Global": {"MSCI ACWI (ACWI)": "ACWI", "MSCI World (URTH)": "URTH", "All World Vanguard (VT)": "VT", "Intl Developed ex-US (EFA)": "EFA"},
    "🏭 Equity — Sectors USA": {"Technology (XLK)": "XLK", "Financials (XLF)": "XLF", "Healthcare (XLV)": "XLV", "Energy (XLE)": "XLE", "Utilities (XLU)": "XLU"},
    "💵 Fixed Income": {"US Agg Bond (AGG)": "AGG", "Total Bond (BND)": "BND", "Long Treasury (TLT)": "TLT", "Corp Bond IG (LQD)": "LQD", "Cash T-Bill (SGOV)": "SGOV"},
    "🪙 Commodities": {"Gold (GLD)": "GLD", "Silver (SLV)": "SLV", "Oil WTI (USO)": "USO", "Broad Commodities (PDBC)": "PDBC"},
    "₿ Crypto": {"Bitcoin (BTC-USD)": "BTC-USD", "Ethereum (ETH-USD)": "ETH-USD", "Solana (SOL-USD)": "SOL-USD"},
}

HISTORICAL_SCENARIOS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-01-15", "2020-03-23"),
    "COVID Recovery (Mar–Dec 2020)": ("2020-03-23", "2020-12-31"),
    "2022 Global Bear Market": ("2022-01-01", "2022-12-31"),
    "GFC Crisis (2008–2009)": ("2008-01-01", "2009-03-09"),
    "Dot-com Crash (2000–2002)": ("2000-03-10", "2002-10-09"),
    "Custom Period": None,
}

BBG_LAYOUT = dict(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(family="IBM Plex Mono, monospace", color="#666", size=11),
    xaxis=dict(gridcolor="#141414", linecolor="#1f1f1f", zerolinecolor="#222", tickfont=dict(size=10, color="#555")),
    yaxis=dict(gridcolor="#141414", linecolor="#1f1f1f", zerolinecolor="#222", tickfont=dict(size=10, color="#555")),
    legend=dict(bgcolor="#0d0d0d", bordercolor="#1a1a1a", borderwidth=1, font=dict(size=10)),
    margin=dict(l=50, r=20, t=50, b=40),
    hovermode="x unified",
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {font-family:'IBM Plex Sans',sans-serif;background-color:#050505;color:#d4d4d4;}
.stApp { background-color:#050505; }
.block-container { padding-top:1rem; max-width:100%; }
.bbg-topbar {background:#f5a623;padding:8px 24px;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;color:#000;letter-spacing:2px;display:flex;justify-content:space-between;margin-bottom:20px;}
.m-card{background:#0d0d0d;border:1px solid #1f1f1f;border-top:2px solid #f5a623;padding:14px 18px;margin-bottom:4px;}
.m-label{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#555;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:5px;}
.m-value{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;line-height:1.1;}
.m-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;margin-top:3px;}
.pos{color:#00e676;}.neg{color:#ff3d3d;}.neu{color:#d4d4d4;}.orange{color:#f5a623;}
.sec-title{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#f5a623;letter-spacing:2.5px;text-transform:uppercase;border-bottom:1px solid #1a1a1a;padding-bottom:8px;margin:24px 0 14px 0;}
[data-testid="stSidebar"]{background:#080808!important;border-right:1px solid #141414;}
.stTextInput input,.stNumberInput input,.stDateInput input{background:#0d0d0d!important;color:#d4d4d4!important;border:1px solid #222!important;font-family:'IBM Plex Mono',monospace!important;}
.stSelectbox > div > div {background:#0d0d0d !important;border:1px solid #222 !important;color:#d4d4d4 !important;}
.stTabs [aria-selected="true"]{color:#f5a623!important;border-bottom:2px solid #f5a623!important;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
"""
