import yfinance as yf
from data import (
    get_balance_sheet,
    get_income_stmt,
    get_market_cap,
    safe_get,
)

def extract_altman_vars(ticker: yf.Ticker) -> dict:    
    bs = get_balance_sheet(ticker)
    inc = get_income_stmt(ticker)

    total_assets = safe_get(bs, ["Total Assets", "TotalAssets"])
    total_liabilities = safe_get(bs, ["Total Liabilities Net Minority Interest",
                                      "TotalLiabilitiesNetMinorityInterest",
                                      "Total Liabilities"])
    current_assets = safe_get(bs, ["Current Assets", "TotalCurrentAssets"])
    current_liabilities = safe_get(bs, ["Current Liabilities", "TotalCurrentLiabilities"])
    retained_earnings = safe_get(bs, ["Retained Earnings", "RetainedEarnings"])
    ebit = safe_get(inc, ["EBIT", "Operating Income", "OperatingIncome"])
    revenue = safe_get(inc, ["Total Revenue", "TotalRevenue"])
    market_cap = get_market_cap(ticker)

    return {
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "working_capital": current_assets - current_liabilities,
        "retained_earnings": retained_earnings,
        "ebit": ebit,
        "revenue": revenue,
        "market_cap": market_cap,
    }

def compute_altman_ratios(vars: dict) -> dict:
    ta = vars["total_assets"]
    if ta == 0:
        return {"X1": 0.0, "X2": 0.0, "X3": 0.0, "X4": 0.0, "X5": 0.0}

    return {
        "X1": vars["working_capital"] / ta,
        "X2": vars["retained_earnings"] / ta,
        "X3": vars["ebit"] / ta,
        "X4": vars["market_cap"] / vars["total_liabilities"]
              if vars["total_liabilities"] != 0 else 0.0,
        "X5": vars["revenue"] / ta,
    }

def altman_zscore(ratios: dict) -> float:
    return (1.2 * ratios["X1"] +
            1.4 * ratios["X2"] +
            3.3 * ratios["X3"] +
            0.6 * ratios["X4"] +
            1.0 * ratios["X5"])

def classify_zscore(z: float) -> str:
    if z > 3.0:
        return "Safe Zone ✅"
    elif z > 1.8:
        return "Grey Zone ⚠️"
    else:
        return "Distress Zone ❌"

def run_altman(ticker: yf.Ticker) -> dict:
    vars = extract_altman_vars(ticker)
    ratios = compute_altman_ratios(vars)
    z = altman_zscore(ratios)

    return {
        **ratios,               # X1, X2, X3, X4, X5
        "z_score": z,
        "z_class": classify_zscore(z),
    }