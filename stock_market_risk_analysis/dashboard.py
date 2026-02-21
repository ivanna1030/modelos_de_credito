"""
dashboard.py
============
Dashboard de Streamlit para el análisis de riesgo bursátil.
Estética terminal financiera — oscuro, preciso, profesional.

Uso:
    streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from credit_decision import analyze_portfolio, credit_decision
from altman import run_altman, classify_zscore
from merton import run_merton, classify_merton
from data import fetch_ticker, get_hist_volatility

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Risk Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: estética terminal financiera ────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #fdf0f4;
    color: #5a1e35;
}
.stApp { background-color: #fdf0f4; }

[data-testid="stSidebar"] {
    background-color: rgba(191,146,162,0.2);
    border-right: 1px solid rgb(146,66,94);
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }

.stTextInput > div > div > input {
    background-color: rgba(191,146,162,0.15) !important;
    border: 1px solid rgb(146,66,94) !important;
    border-radius: 4px !important;
    color: #5a1e35 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px;
}
.stTextInput > div > div > input::placeholder { color: #b06080 !important; }
.stTextInput > div > div > input:focus {
    border-color: #5a1e35 !important;
    box-shadow: 0 0 8px rgba(146,66,94,0.25) !important;
}

.stButton > button {
    background: rgb(146,66,94) !important;
    border: 1px solid #5a1e35 !important;
    color: #fdf0f4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    font-size: 12px !important;
    border-radius: 4px !important;
    padding: 0.5rem 2rem !important;
    text-transform: uppercase;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #5a1e35 !important;
    box-shadow: 0 0 14px rgba(146,66,94,0.35) !important;
}

[data-testid="metric-container"] {
    background: rgba(191,146,162,0.2) !important;
    border: 1px solid rgb(146,66,94) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #5a1e35 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #92425e !important;
}

[data-testid="stDataFrame"] { border: 1px solid rgb(146,66,94) !important; }
.dataframe { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; }
hr { border-color: rgba(146,66,94,0.4) !important; }

[data-testid="stExpander"] {
    border: 1px solid rgba(146,66,94,0.5) !important;
    background: rgba(191,146,162,0.12) !important;
    border-radius: 4px !important;
}
            
[data-testid="stExpander"] summary {
    color: #92425e !important;
    font-family: 'IBM Plex Mono' !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
}
            
[data-testid="stExpander"] summary:hover {
    color: #d4789a !important;
}

[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #92425e !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5a1e35 !important;
    border-bottom-color: rgb(146,66,94) !important;
    font-weight: 700 !important;
}

.stSpinner > div { border-top-color: rgb(146,66,94) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #fdf0f4; }
::-webkit-scrollbar-thumb { background: rgba(146,66,94,0.4); border-radius: 4px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 2rem 0 1rem 0; border-bottom: 1px solid rgba(146,66,94,0.4); margin-bottom: 2rem;">
    <div style="display:flex; align-items:baseline; gap:1rem;">
        <span style="font-family:'IBM Plex Mono'; font-size:1.8rem; font-weight:700;
                     color:#92425e; letter-spacing:3px;">RISK TERMINAL</span>
        <span style="font-family:'IBM Plex Mono'; font-size:0.75rem; color:#92425e;
                     letter-spacing:2px;">ALTMAN Z-SCORE + MERTON MODEL</span>
    </div>
    <div style="font-family:'IBM Plex Mono'; font-size:0.7rem; color:#b06080;
                letter-spacing:1px; margin-top:0.3rem;">
        CREDIT RISK ANALYSIS SYSTEM  //  DATA: YAHOO FINANCE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; letter-spacing:3px;
                color:#92425e; margin-bottom:1.5rem; border-bottom:1px solid rgba(146,66,94,0.4);
                padding-bottom:1rem;">
        ◈ INPUT PARAMETERS
    </div>
    """, unsafe_allow_html=True)

    raw_input = st.text_input(
        "TICKER SYMBOLS",
        value="AAPL, TSLA, DIS",
        help="Ingresa los tickers separados por comas. Ej: AAPL, TSLA, MSFT, F",
        placeholder="AAPL, TSLA, MSFT..."
    )

    st.markdown("""
    <div style="font-family:'IBM Plex Mono'; font-size:0.6rem; color:#b06080;
                margin: 0.5rem 0 1.5rem 0; letter-spacing:1px;">
        SEPARATE WITH COMMAS
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  RUN ANALYSIS")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; letter-spacing:3px;
                color:#92425e; margin-bottom:1rem; border-bottom:1px solid rgba(146,66,94,0.4);
                padding-bottom:0.5rem;">
        ◈ DECISION THRESHOLDS
    </div>
    <div style="font-size:0.65rem; color:#b06080; line-height:2; letter-spacing:0.5px;">
        Z &gt; 3.0 ............ SAFE ZONE<br>
        1.8 &lt; Z &lt; 3.0 ...... GREY ZONE<br>
        Z &lt; 1.8 ............ DISTRESS ZONE<br>
        <br>
        PD &lt; 5% ............ LOW RISK<br>
        5% ≤ PD &lt; 20% ...... MED RISK<br>
        PD ≥ 20% ........... HIGH RISK<br>
        <br>
        APPROVED: Z &gt; 1.8 AND PD &lt; 20%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; letter-spacing:3px;
                color:#92425e; margin-bottom:1rem; border-bottom:1px solid rgba(146,66,94,0.4);
                padding-bottom:0.5rem;">
        ◈ ALTMAN FORMULA (1968)
    </div>
    <div style="font-size:0.6rem; color:#b06080; line-height:2;">
        Z = 1.2X₁ + 1.4X₂ + 3.3X₃ + 0.6X₄ + 1.0X₅
    </div>
    <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; letter-spacing:3px;
                color:#92425e; margin-top:1rem; margin-bottom:0.5rem;
                border-bottom:1px solid rgba(146,66,94,0.3); padding-bottom:0.3rem;">
        ◈ MERTON FORMULA
    </div>
    <div style="font-size:0.6rem; color:#b06080; line-height:2;">
        DD = [ln(V/D)+(r−σ²/2)T] / σ√T<br>
        PD = 1 − N(DD)
    </div>
    """, unsafe_allow_html=True)

# ── Color helpers ─────────────────────────────────────────────────────────────

def z_color(z):
    return "#5a1e35"

def pd_color(pd):
    return "#5a1e35"

def decision_color(d):
    return "#5a1e35"

def pink_table(df):

    # Detectar columnas numéricas
    numeric_cols = df.select_dtypes(include="number").columns

    styled_df = df.style \
        .format({col: "{:.2f}" for col in numeric_cols}) \
        .set_table_styles([
            {
                "selector": "thead",
                "props": [
                    ("background-color", "#92425e"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "rgba(191,146,162,0.15)")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "rgba(191,146,162,0.05)")],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("color", "#92425e"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                ],
            },
        ]) \
        .hide(axis="index")

    st.markdown(styled_df.to_html(), unsafe_allow_html=True)

def gauge_chart(value, title, min_val, max_val, thresholds, colors, fmt=".2f"):
    """Crea un gauge chart con Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"family": "IBM Plex Mono", "size": 28, "color": "#5a1e35"},
                "suffix": "%" if "%" in title else "",
                "valueformat": fmt},
        title={"text": title,
               "font": {"family": "IBM Plex Mono", "size": 11,
                        "color": "#92425e"}},
        gauge={
            "axis": {"range": [min_val, max_val],
                     "tickfont": {"family": "IBM Plex Mono", "size": 8, "color": "#b06080"},
                     "tickcolor": "#92425e"},
            "bar": {"color": colors[0], "thickness": 0.25},
            "bgcolor": "rgba(191,146,162,0.2)",
            "borderwidth": 1,
            "bordercolor": "#92425e",
            "steps": [
                {"range": [min_val, thresholds[0]], "color": "#200d16"},
                {"range": [thresholds[0], thresholds[1]], "color": "#2a1520"},
                {"range": [thresholds[1], max_val], "color": "#200d16"},
            ],
            "threshold": {
                "line": {"color": colors[0], "width": 2},
                "thickness": 0.75,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        font={"family": "IBM Plex Mono"},
    )
    return fig

def price_chart(ticker_obj, symbol):
    """Gráfico de precio histórico con área."""
    hist = ticker_obj.history(period="1y")
    if hist.empty:
        return None

    color = "#92425e"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(146,66,94,0.08)",
        name="Close",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        title=dict(text=f"{symbol} — 1Y PRICE", font=dict(family="IBM Plex Mono", size=10, color="#92425e")),
        xaxis=dict(showgrid=False, tickfont=dict(family="IBM Plex Mono", size=8, color="#b06080"),
                   tickcolor="#92425e", linecolor="#92425e"),
        yaxis=dict(showgrid=True, gridcolor="#200d16", tickfont=dict(family="IBM Plex Mono", size=8, color="#b06080"),
                   tickprefix="$", tickcolor="#92425e", linecolor="#92425e"),
        showlegend=False,
    )
    return fig

def radar_chart(ratios_list, symbols):
    """Radar chart comparando los ratios X1-X5 de todas las empresas."""
    categories = ["X1", "X2", "X3", "X4", "X5"]
    colors_list = ["#92425e", "#d4789a", "#bf92a2", "#6b1a2e", "#d4789a"]

    fig = go.Figure()
    for i, (ratios, sym) in enumerate(zip(ratios_list, symbols)):
        vals = [ratios["X1"], ratios["X2"], ratios["X3"], ratios["X4"], ratios["X5"]]
        # cerrar el polígono
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        color = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color=color, width=2),
            name=sym,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(191,146,162,0.15)",
            radialaxis=dict(visible=True, showticklabels=True,
                            tickfont=dict(family="IBM Plex Mono", size=7, color="#b06080"),
                            gridcolor="#92425e", linecolor="#92425e"),
            angularaxis=dict(tickfont=dict(family="IBM Plex Mono", size=10, color="#92425e"),
                             gridcolor="#92425e", linecolor="#92425e"),
        ),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        height=360,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#5a1e35"),
                    bgcolor="rgba(191,146,162,0.15)", bordercolor="rgb(146,66,94)", borderwidth=1),
        title=dict(text="ALTMAN RATIOS COMPARISON", font=dict(family="IBM Plex Mono", size=11, color="#92425e")),
    )
    return fig

def zscore_bar_chart(results):
    """Bar chart de Z-scores con líneas de umbral."""
    symbols = [r["symbol"] for r in results]
    zscores = [r["Z_score"] for r in results]
    bar_colors = ["#92425e" for z in zscores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols, y=zscores,
        marker=dict(color=bar_colors, line=dict(color="#1a0a10", width=1)),
        text=[f"{z:.2f}" for z in zscores],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
        hovertemplate="<b>%{x}</b><br>Z-Score: %{y:.2f}<extra></extra>",
    ))
    # Líneas de umbral
    fig.add_hline(y=3.0, line=dict(color="#5a1e35", width=1, dash="dot"),
                  annotation_text="SAFE (3.0)", annotation_font=dict(family="IBM Plex Mono", size=9, color="#5a1e35"))
    fig.add_hline(y=1.8, line=dict(color="#6b1a2e", width=1, dash="dot"),
                  annotation_text="DISTRESS (1.8)", annotation_font=dict(family="IBM Plex Mono", size=9, color="#6b1a2e"))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        title=dict(text="Z-SCORE COMPARISON", font=dict(family="IBM Plex Mono", size=11, color="#92425e")),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
                   gridcolor="#200d16", linecolor="#92425e"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                   gridcolor="#200d16", linecolor="#92425e", title="Z-Score",
                   titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
        showlegend=False,
    )
    return fig

def pd_bar_chart(results):
    """Bar chart de probabilidades de default."""
    symbols = [r["symbol"] for r in results]
    pds = [r["Default_Prob"] * 100 for r in results]
    bar_colors = ["#92425e" for p in pds]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols, y=pds,
        marker=dict(color=bar_colors, line=dict(color="#1a0a10", width=1)),
        text=[f"{p:.2f}%" for p in pds],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
        hovertemplate="<b>%{x}</b><br>Default Prob: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=20, line=dict(color="#6b1a2e", width=1, dash="dot"),
                  annotation_text="HIGH RISK (20%)", annotation_font=dict(family="IBM Plex Mono", size=9, color="#6b1a2e"))
    fig.add_hline(y=5, line=dict(color="#b06080", width=1, dash="dot"),
                  annotation_text="LOW RISK (5%)", annotation_font=dict(family="IBM Plex Mono", size=9, color="#b06080"))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        title=dict(text="DEFAULT PROBABILITY (%)", font=dict(family="IBM Plex Mono", size=11, color="#92425e")),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
                   gridcolor="#200d16", linecolor="#92425e"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                   gridcolor="#200d16", linecolor="#92425e", title="PD (%)",
                   titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
        showlegend=False,
    )
    return fig

def risk_scatter(results):
    """Scatter plot Z-Score vs PD para ver el riesgo combinado."""
    fig = go.Figure()
    for r in results:
        z = r["Z_score"]
        pd = r["Default_Prob"] * 100
        color = decision_color(r["Decision"])
        fig.add_trace(go.Scatter(
            x=[z], y=[pd],
            mode="markers+text",
            marker=dict(size=18, color=color, line=dict(color="#1a0a10", width=2),
                        symbol="diamond"),
            text=[r["symbol"]],
            textposition="top center",
            textfont=dict(family="IBM Plex Mono", size=10, color=color),
            name=r["symbol"],
            hovertemplate=f"<b>{r['symbol']}</b><br>Z-Score: {z:.2f}<br>PD: {pd:.2f}%<br>{r['Decision']}<extra></extra>",
        ))

    # Zonas de color
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(191,146,162,0.08)", line_width=0)
    fig.add_hrect(y0=20, y1=100, fillcolor="rgba(146,66,94,0.06)", line_width=0)
    fig.add_vrect(x0=1.8, x1=3.0, fillcolor="rgba(146,66,94,0.06)", line_width=0)
    fig.add_vline(x=1.8, line=dict(color="#6b1a2e", width=1, dash="dot"))
    fig.add_vline(x=3.0, line=dict(color="#5a1e35", width=1, dash="dot"))
    fig.add_hline(y=20, line=dict(color="#6b1a2e", width=1, dash="dot"))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        title=dict(text="RISK MAP: Z-SCORE vs DEFAULT PROBABILITY",
                   font=dict(family="IBM Plex Mono", size=11, color="#92425e")),
        xaxis=dict(title="Z-Score", tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                   gridcolor="#200d16", linecolor="#92425e",
                   titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
        yaxis=dict(title="Default Prob (%)", tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                   gridcolor="#200d16", linecolor="#92425e",
                   titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
        legend=dict(font=dict(family="IBM Plex Mono", size=9, color="#5a1e35"),
                    bgcolor="rgba(191,146,162,0.15)", bordercolor="rgb(146,66,94)"),
        showlegend=False,
    )
    return fig

def vol_comparison(ticker_objs, symbols):
    """Boxplot de distribución de retornos diarios por empresa."""
    fig = go.Figure()
    colors_list = ["#92425e", "#d4789a", "#bf92a2", "#6b1a2e", "#d4789a"]
    for i, (t, sym) in enumerate(zip(ticker_objs, symbols)):
        hist = t.history(period="1y")
        if hist.empty:
            continue
        returns = (np.log(hist["Close"] / hist["Close"].shift(1)).dropna() * 100).values
        color = colors_list[i % len(colors_list)]
        fig.add_trace(go.Box(
            y=returns, name=sym,
            marker_color=color,
            line=dict(color=color, width=1.5),
            fillcolor=color.replace("ff", "1a") if "ff" in color else "#92425e",
            boxmean=True,
            hovertemplate="<b>" + sym + "</b><br>Return: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#fdf0f4",
        plot_bgcolor="rgba(191,146,162,0.1)",
        title=dict(text="DAILY RETURNS DISTRIBUTION (1Y)",
                   font=dict(family="IBM Plex Mono", size=11, color="#92425e")),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=10, color="#5a1e35"),
                   gridcolor="#200d16", linecolor="#92425e"),
        yaxis=dict(title="Daily Return (%)", tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                   gridcolor="#200d16", linecolor="#92425e",
                   titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
        showlegend=False,
    )
    return fig

# ── Badge HTML helpers ────────────────────────────────────────────────────────

def badge(text, color):
    bg = color + "22"
    return f"""<span style="background:{bg}; color:{color}; border:1px solid {color};
    font-family:'IBM Plex Mono'; font-size:0.65rem; font-weight:600;
    padding:2px 10px; border-radius:2px; letter-spacing:1px;">{text}</span>"""

def section_header(title):
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; letter-spacing:3px;
                color:#92425e; margin: 2rem 0 1rem 0; border-bottom:1px solid rgba(146,66,94,0.4);
                padding-bottom:0.5rem;">
        ◈ {title}
    </div>""", unsafe_allow_html=True)

# ── Main logic ────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.tickers_done = []
    st.session_state.ticker_objs = []

if run_btn:
    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
    if not tickers:
        st.error("Ingresa al menos un ticker.")
    else:
        with st.spinner():
            try:
                results = analyze_portfolio(tickers)
                ticker_objs = [fetch_ticker(sym) for sym in tickers]
                st.session_state.results = results
                st.session_state.tickers_done = tickers
                st.session_state.ticker_objs = ticker_objs
            except Exception as e:
                st.error(f"Error al obtener datos: {e}")

results = st.session_state.results
tickers = st.session_state.tickers_done
ticker_objs = st.session_state.ticker_objs

if results is None:
    # Estado vacío — instrucciones
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                height:50vh; gap:1.5rem; text-align:center;">
        <div style="font-family:'IBM Plex Mono'; font-size:3rem; color:#92425e;">◈</div>
        <div style="font-family:'IBM Plex Mono'; font-size:0.75rem; letter-spacing:3px; color:#b06080;">
            ENTER TICKER SYMBOLS IN THE SIDEBAR
        </div>
        <div style="font-family:'IBM Plex Mono'; font-size:0.65rem; color:#92425e; letter-spacing:1px;">
            SUPPORTS ANY VALID YAHOO FINANCE TICKER
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Summary cards ─────────────────────────────────────────────────────────
    section_header("CREDIT DECISION SUMMARY")

    for i in range(0, len(results), 3):
        chunk = results[i:i+3]
        cols = st.columns(3)
        for col, r in zip(cols, chunk):
            with col:
                decision_text = "APPROVED" if "APPROVED" in r["Decision"] else "DENIED"

                st.markdown(f"""
                <div style="background:rgba(191,146,162,0.2); border:1px solid #92425e; border-top: 2px solid #92425e;
                            padding:1.2rem; border-radius:2px; margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:center; align-items:center; gap:0.965rem;
                                font-family:'IBM Plex Mono'; font-size:1.3rem; color:#92425e;
                                letter-spacing:2px; margin-bottom:0.8rem;">
                        <span style="font-weight:700;">{r["symbol"]}</span>
                        <span style="font-size:1rem; font-weight:400;">${r["Price"]:.2f}</span>
                        <span style="font-size:0.7rem; font-weight:400; color:#92425e; margin-left:0.5rem;">
                            {decision_text}
                        </span>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; justify-items:center;">
                        <div>
                            <div style="font-size:0.55rem; letter-spacing:2px; color:#b06080; text-align:center; ">Z-SCORE</div>
                            <div style="font-size:1.4rem; font-weight:700; color:#92425e; text-align:center; ">{r["Z_score"]:.2f}</div>
                            <div style="font-size:0.55rem; color:#92425e; text-align:center; ">{r["Z_class"]}</div>
                        </div>
                        <div>
                            <div style="font-size:0.55rem; letter-spacing:2px; color:#b06080; text-align:center; ">DEFAULT PROB</div>
                            <div style="font-size:1.4rem; font-weight:700; color:#92425e; text-align:center; ">{r["Default_Prob"]:.2%}</div>
                            <div style="font-size:0.55rem; color:#92425e; text-align:center; ">{r["PD_class"]}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tabs principales ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "  OVERVIEW  ",
        "  ALTMAN Z-SCORE  ",
        "  MERTON MODEL  ",
        "  PRICE & VOLATILITY  ",
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(zscore_bar_chart(results), use_container_width=True)
        with col_right:
            st.plotly_chart(pd_bar_chart(results), use_container_width=True)

        st.plotly_chart(risk_scatter(results), use_container_width=True)

        # Tabla resumen
        section_header("FULL DATA TABLE")
        df = pd.DataFrame([{
            "Ticker":    r["symbol"],
            "Z-Score":   round(r["Z_score"], 2),
            "Z Zone":    r["Z_class"],
            "Asset Value": f"${r['Asset_Value_B']:.2f}B",
            "Asset Vol": f"{r['Asset_Vol']:.2%}",
            "Default Prob": f"{r['Default_Prob']:.2%}",
            "Risk":      r["PD_class"],
            "Decision":  r["Decision"],
        } for r in results])
        pink_table(df)

    # ── TAB 2: ALTMAN ─────────────────────────────────────────────────────────
    with tab2:
        section_header("ALTMAN Z-SCORE DETAIL")
    
        ratios_df = pd.DataFrame([{
            "Ticker": r["symbol"],
            "X1":     round(r["X1"], 2),
            "X2": round(r["X2"], 2),
            "X3":   round(r["X3"], 2),
            "X4":      round(r["X4"], 2),
            "X5":    round(r["X5"], 2),
            "Z-Score":            round(r["Z_score"], 2),
        } for r in results])
        pink_table(ratios_df)

        # Explicación de ratios
        section_header("RATIO DEFINITIONS")
        st.markdown("""
        <div style="background:rgba(191,146,162,0.2); border:1px solid #92425e; padding:1rem;
                    border-radius:2px; margin-top:1rem; max-width:393px">
            <div style="font-size:0.75rem; color:#b06080; line-height:2;">
                <strong>X1 = Working Capital / Total Assets :</strong> It measures a company's liquidity.<br>
                <strong>X2 = Retained Earnings / Total Assets :</strong> It measures profitability.<br>
                <strong>X3 = EBIT / Total Assets :</strong> It measures operating efficiency.<br>
                <strong>X4 = Market Cap / Total Liabilities :</strong> It measures leverage.<br>
                <strong>X5 = Revenue / Total Assets :</strong> It measures asset turnover.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Explicación del modelo
        with st.expander("▸ ABOUT ALTMAN Z-SCORE (1968)"):
            st.markdown("""
            <div style="font-family:'IBM Plex Mono'; font-size:0.7rem; color:#92425e; line-height:2;">
            <b style="color:#92425e">Altman Z-Score (1968)</b> is a formula developed by Edward Altman 
            to predict the likelihood that a company will go bankrupt within two years, assessing the 
            overall financial health of a company. It analyzes several financial ratios and combines five 
            of them using a weighting system to calculate the overall Z-Score.<br><br>
            <b>Applications:</b><br>
            &nbsp;• <b>Financial status:</b> indicates financial solvency and predicts bankruptcy.<br>
            &nbsp;• <b>Investment decisions:</b> helps investors evaluate investment risks.<br>
            &nbsp;• <b>Lending decisions:</b> banks assess creditworthiness for loans.<br>
            &nbsp;• <b>Audits:</b> auditors evaluate overall financial health.<br><br>
            <b>Limitations:</b><br>
            &nbsp;• <b>Qualitative factors:</b> Does not consider qualitative factors (management, industry trends, competition).<br>
            &nbsp;• <b>Industry biases:</b> originally based on manufacturing companies.<br>
            &nbsp;• <b>Backward-looking:</b> relies on historical financial data.<br>
            &nbsp;• <b>Private companies:</b> limited applicability due to lack of disclosed financials.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: MERTON ─────────────────────────────────────────────────────────
    with tab3:
        section_header("MERTON MODEL DETAIL")

        merton_df = pd.DataFrame([{
            "Ticker":          r["symbol"],
            "Equity Value":    f"${r['Asset_Value_B'] - (r.get('debt_b', 0)):.2f}B",
            "Asset Value":     f"${r['Asset_Value_B']:.2f}B",
            "Asset Volatility":f"{r['Asset_Vol']:.2%}",
            "Default Prob":    f"{r['Default_Prob']:.2%}",
            "Risk Level":      r["PD_class"],
        } for r in results])
        pink_table(merton_df)

        # Asset Volatility comparison
        section_header("ASSET VOLATILITY COMPARISON")
        vols = [r["Asset_Vol"] * 100 for r in results]
        syms = [r["symbol"] for r in results]
        vc = ["#92425e" for v in vols]

        fig_vol = go.Figure(go.Bar(
            x=syms, y=vols,
            marker=dict(color=vc, line=dict(color="#1a0a10", width=1)),
            text=[f"{v:.2f}%" for v in vols],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
        ))
        fig_vol.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#fdf0f4",
            plot_bgcolor="rgba(191,146,162,0.1)",
            xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color="#5a1e35"),
                       gridcolor="#200d16", linecolor="#92425e"),
            yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=9, color="#b06080"),
                       gridcolor="#200d16", linecolor="#92425e", title="σ_V (%)",
                       titlefont=dict(family="IBM Plex Mono", size=9, color="#92425e")),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Explicación del modelo
        with st.expander("▸ ABOUT THE MERTON MODEL"):
            st.markdown("""
            <div style="font-family:'IBM Plex Mono'; font-size:0.7rem; color:#92425e; line-height:2;">
            <b style="color:#92425e">Merton Model</b> uses option pricing theory to estimate a company's probability of default 
            by modeling equity as a call option on the firm's assets, with debt as the strike price.<br><br>
            <b>Core Concept:</b><br>
            &nbsp;• <b>Assets (A):</b> total firm value.<br>
            &nbsp;• <b>Liabilities (L):</b> debt with a future maturity.<br>
            &nbsp;• <b>Equity (E):</b> treated as a call option on assets: E = max(A - L, 0).<br><br>
            <b>Logic:</b><br>
            &nbsp;• If assets exceed debt at maturity, shareholders keep A - L.<br>
            &nbsp;• If assets are less than debt, the firm defaults and creditors take over.<br><br>
            <b>Distance to Default (DD):</b> Number of standard deviations between expected asset value at debt maturity and the liability threshold.<br>
            &nbsp;• DD = (ln(V/D) + (r + σ²/2)·T) / (σ·√T)<br>
            &nbsp;• A lower distance to default indicates a higher probability of default.<br><br>
            <b>Probability of Default (PD):</b> Probability of the asset value falling below the liability threshold.<br>
            &nbsp;• PD = 1 - N(DD)<br><br>
            <b>Assumptions:</b><br>
            &nbsp;• <b>Continuous-time framework:</b> the value of a company's assets and debt evolves continuously.<br>
            &nbsp;• <b>Geometric Brownian motion:</b> the company's assets follow a geometric Brownian motion.<br>
            &nbsp;• <b>Tradeable debt and equity:</b> both the debt and equity of the company are tradeable.<br>
            &nbsp;• <b>Risk-neutral valuation:</b> assumes market participants are risk-neutral and make decisions based on risk-free interest rates.<br>
            &nbsp;• <b>No taxes or transaction costs:</b> the model ignores taxes and transaction costs.<br>
            &nbsp;• <b>Constant asset volatility:</b> the volatility of the company's asset value (σ) remains constant over time.<br><br>
            <b>Observations:</b><br>
            &nbsp;• Equity is observable from market prices; assets and liabilities are typically unobservable.<br>
            &nbsp;• Liability threshold often chosen based on short-term and total liabilities.<br>
            &nbsp;• Assumes lognormal distribution for asset returns to handle non-negativity, skewness, and heavy tails.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: PRICE & VOLATILITY ─────────────────────────────────────────────
    with tab4:
        section_header("HISTORICAL PRICE CHARTS")

        # Precio histórico por empresa
        if ticker_objs:
            price_cols = st.columns(min(len(ticker_objs), 2))
            for i, (t, sym) in enumerate(zip(ticker_objs, tickers)):
                with price_cols[i % 2]:
                    fig_p = price_chart(t, sym)
                    if fig_p:
                        st.plotly_chart(fig_p, use_container_width=True)

            section_header("RETURNS DISTRIBUTION")
            st.plotly_chart(vol_comparison(ticker_objs, tickers), use_container_width=True)

            # Info de mercado
            section_header("MARKET INFO")
            info_rows = []
            for t, sym in zip(ticker_objs, tickers):
                info = t.info
                info_rows.append({
                    "Ticker":      sym,
                    "Company":     info.get("longName", "N/A"),
                    "Sector":      info.get("sector", "N/A"),
                    "Industry":    info.get("industry", "N/A"),
                    "Market Cap":  f"${info.get('marketCap', 0)/1e9:.2f}B",
                    "P/E Ratio":   round(info.get("trailingPE", 0), 2),
                    "52W High":    f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
                    "52W Low":     f"${info.get('fiftyTwoWeekLow', 0):.2f}",
                    "Beta":        round(info.get("beta", 0), 2),
                })
            pink_table(pd.DataFrame(info_rows))