import warnings
warnings.filterwarnings("ignore")

from tabulate import tabulate
from credit_decision import analyze_portfolio

TICKERS = ["AAPL", "TSLA", "DIS"]

def print_results(results: list) -> None:
    
    print("\n" + "=" * 82)
    print("                              ALTMAN Z-SCORE RESULTS")
    print("=" * 82)
    altman_rows = [
        [r["symbol"],
         f"{r['X1']:.2f}", f"{r['X2']:.2f}", f"{r['X3']:.2f}",
         f"{r['X4']:.2f}", f"{r['X5']:.2f}",
         f"{r['Z_score']:.2f}", r["Z_class"]]
        for r in results
    ]
    print(tabulate(altman_rows,
                   headers=["Ticker", "X1", "X2", "X3", "X4", "X5", "Z-Score", "Zone"],
                   tablefmt="rounded_outline",
                   colalign=("center",) * 8))

    print("\n" + "=" * 82)
    print("                               MERTON MODEL RESULTS")
    print("=" * 82)
    merton_rows = [
        [r["symbol"],
         f"${r['Asset_Value_B']:.2f}B",
         f"{r['Asset_Vol']:.2%}",
         f"{r['Default_Prob']:.2%}",
         r["PD_class"]]
        for r in results
    ]
    print(tabulate(merton_rows,
                   headers=["Ticker", "Asset Value", "Asset Volatility",
                             "Default Prob.", "Risk Level"],
                   tablefmt="rounded_outline",
                   colalign=("center",) * 5))

    print("\n" + "=" * 85)
    print("                               CREDIT DECISION SUMMARY")
    print("=" * 85)
    decision_rows = [
        [r["symbol"], f"{r['Z_score']:.2f}", r["Z_class"],
         f"{r['Default_Prob']:.2%}", r["PD_class"], r["Decision"]]
        for r in results
    ]
    print(tabulate(decision_rows,
                   headers=["Ticker", "Z-Score", "Zone",
                             "Default Prob.", "Risk", "Decision"],
                   tablefmt="rounded_outline",
                   colalign=("center",) * 6))

if __name__ == "__main__":
    
    results = analyze_portfolio(TICKERS)

    print_results(results)