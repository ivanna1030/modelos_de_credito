import yfinance as yf
import numpy as np
import pandas as pd

def fetch_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)

def get_balance_sheet(ticker: yf.Ticker) -> pd.DataFrame:
    return ticker.balance_sheet

def get_income_stmt(ticker: yf.Ticker) -> pd.DataFrame:
    return ticker.income_stmt

def get_market_cap(ticker: yf.Ticker) -> float:
    return float(ticker.info.get("marketCap", 0))

def get_stock_price(ticker: yf.Ticker) -> float:
    info = ticker.info
    return float(info.get("currentPrice", info.get("regularMarketPrice", 0)))

def get_shares_outstanding(ticker: yf.Ticker) -> float:
    return float(ticker.info.get("sharesOutstanding", 0))

def get_hist_volatility(ticker: yf.Ticker, period: str = "1y") -> float:
    hist = ticker.history(period=period)
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))

def safe_get(df: pd.DataFrame, keys: list,
             col_idx: int = 0, default: float = 0.0) -> float:
    for key in keys:
        if key in df.index:
            val = df.loc[key].iloc[col_idx]
            if pd.notna(val):
                return float(val)
    return default