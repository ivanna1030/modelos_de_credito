import numpy as np

from data import fetch_ticker
from altman import run_altman
from merton import run_merton

def credit_decision(z_score: float, default_prob: float) -> str:
    z_ok = z_score > 1.8
    pd_ok = default_prob < 0.20
    return "APPROVED ✅" if (z_ok and pd_ok) else "DENIED ❌"

def analyze_company(symbol: str) -> dict:
    ticker = fetch_ticker(symbol)

    try:
        price = ticker.history(period="1d")["Close"].iloc[-1]
    except:
        price = np.nan

    altman_result = run_altman(ticker)
    merton_result = run_merton(ticker)

    z = altman_result["z_score"]
    pd = merton_result["default_probability"]

    return {
        "symbol": symbol,
        "Price": price,

        # Altman
        "X1": altman_result["X1"],
        "X2": altman_result["X2"],
        "X3": altman_result["X3"],
        "X4": altman_result["X4"],
        "X5": altman_result["X5"],
        "Z_score": z,
        "Z_class": altman_result["z_class"],

        # Merton
        "Asset_Value_B": merton_result["asset_value"] / 1e9,
        "Asset_Vol": merton_result["asset_volatility"],
        "Default_Prob": pd,
        "PD_class": merton_result["pd_class"],

        # Decisión
        "Decision": credit_decision(z, pd),
    }

def analyze_portfolio(symbols: list) -> list:
    return list(map(analyze_company, symbols))