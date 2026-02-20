import numpy as np
from scipy.stats import norm
import yfinance as yf
from data import (
    get_balance_sheet,
    get_market_cap,
    get_hist_volatility,
    safe_get,
)

def extract_merton_vars(ticker: yf.Ticker,
                        risk_free_rate: float = 0.05,
                        horizon: float = 1.0) -> dict:
    bs = get_balance_sheet(ticker)

    total_debt = safe_get(bs, [
        "Total Debt", "TotalDebt",
        "Long Term Debt", "LongTermDebt",
        "Total Liabilities Net Minority Interest",
    ])

    return {
        "equity_value": get_market_cap(ticker),
        "debt_face_value": total_debt,
        "sigma_equity": get_hist_volatility(ticker),
        "r": risk_free_rate,
        "T": horizon,
    }

def merton_asset_value(E: float, D: float, sigma_e: float,
                       r: float, T: float,
                       tol: float = 1e-6,
                       max_iter: int = 1000) -> tuple:
    if D == 0 or E == 0:
        return (E, sigma_e)

    V = E + D
    sigma_V = sigma_e * E / V

    for _ in range(max_iter):
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        V_new = E + D * np.exp(-r * T) * N_d2
        denom = N_d1 * V_new
        sigma_V_new = (sigma_e * E / denom) if denom != 0 else sigma_e

        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            V, sigma_V = V_new, sigma_V_new
            break
        V, sigma_V = V_new, sigma_V_new

    return (V, sigma_V)

def merton_default_probability(V: float, sigma_V: float,
                               D: float, r: float, T: float) -> float:
    if D == 0 or V == 0 or sigma_V == 0:
        return 0.0

    d2 = (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    return float(norm.cdf(-d2))


def classify_merton(pd: float) -> str:
    if pd < 0.05:
        return "Low Risk ✅"
    elif pd < 0.20:
        return "Medium Risk ⚠️"
    else:
        return "High Risk ❌"

def run_merton(ticker: yf.Ticker) -> dict:
    vars = extract_merton_vars(ticker)

    E, D = vars["equity_value"], vars["debt_face_value"]
    sigma_e = vars["sigma_equity"]
    r, T = vars["r"], vars["T"]

    V, sigma_V = merton_asset_value(E, D, sigma_e, r, T)
    pd = merton_default_probability(V, sigma_V, D, r, T)

    return {
        "asset_value": V,
        "asset_volatility": sigma_V,
        "default_probability": pd,
        "pd_class": classify_merton(pd),
    }