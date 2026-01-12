"""
validation2_costs_fixed_fee.py

Validation phase (no optimisation):
Apply fixed per-trade fees (e.g., £12/order) to an EXISTING walk-forward return series,
then recompute performance metrics.

Assumes CSV has at minimum:
  - date
  - strategy_ret   (daily return, decimal: 0.01 = +1%)
  - position       (e.g., 0/1 or -1/0/1)

Optional:
  - strategy_equity (ignored for costed equity; we rebuild from returns for clean accounting)

Usage:
  python3 validation2_costs_fixed_fee.py --csv gold_walkforward_backtest.csv
  python3 validation2_costs_fixed_fee.py --csv gold_walkforward_backtest.csv --fee 12 --capital 100000
"""

import argparse
import numpy as np
import pandas as pd


# ----------------------------
# Metrics
# ----------------------------
def max_drawdown(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 2:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())  # negative


def annualised_vol(daily_ret: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(daily_ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(daily_ret: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(daily_ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((r.mean() / std) * np.sqrt(periods))


def cagr(equity: pd.Series, dates: pd.Series) -> float:
    tmp = pd.DataFrame({"date": pd.to_datetime(dates), "equity": pd.to_numeric(equity, errors="coerce")})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 2:
        return np.nan
    d0 = tmp["date"].iloc[0]
    d1 = tmp["date"].iloc[-1]
    years = (d1 - d0).days / 365.25
    if years <= 0:
        return np.nan
    start = float(tmp["equity"].iloc[0])
    end = float(tmp["equity"].iloc[-1])
    if start <= 0 or end <= 0:
        return np.nan
    return float((end / start) ** (1.0 / years) - 1.0)


def time_in_market(position: pd.Series) -> float:
    p = pd.to_numeric(position, errors="coerce").fillna(0.0)
    return float((p != 0).mean())


def count_switches(position: pd.Series) -> int:
    p = pd.to_numeric(position, errors="coerce").fillna(0.0)
    return int((p.diff().fillna(0) != 0).sum())


# ----------------------------
# Cost application
# ----------------------------
def orders_from_position_changes(position: pd.Series) -> pd.Series:
    """
    Orders per day based on change in position.
    For positions in {-1,0,1}:
      0->1 = 1 order
      1->0 = 1 order
      -1->+1 = 2 orders
    For fractional positions, abs(diff) is still a reasonable proxy for "how much traded".
    """
    p = pd.to_numeric(position, errors="coerce").fillna(0.0)
    return p.diff().abs().fillna(0.0)


def build_equity_with_fixed_fees(
    dates: pd.Series,
    daily_ret: pd.Series,
    position: pd.Series,
    start_capital: float,
    fee_per_order: float,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ret": pd.to_numeric(daily_ret, errors="coerce").fillna(0.0),
        "position": pd.to_numeric(position, errors="coerce").fillna(0.0),
    }).sort_values("date").reset_index(drop=True)

    d["orders"] = orders_from_position_changes(d["position"])
    d["fee"] = d["orders"] * float(fee_per_order)

    equity = []
    equity_net = []
    fee_paid = []

    eq = float(start_capital)
    eq_net = float(start_capital)

    for i in range(len(d)):
        r = float(d.loc[i, "ret"])
        f = float(d.loc[i, "fee"])

        # Gross equity (no costs)
        eq = eq * (1.0 + r)

        # Net equity: apply return, then subtract fees in currency terms
        eq_net = eq_net * (1.0 + r) - f

        equity.append(eq)
        equity_net.append(eq_net)
        fee_paid.append(f)

    d["equity_gross"] = equity
    d["equity_net"] = equity_net
    d["fee_paid"] = fee_paid

    # Implied net daily return series (for Sharpe/vol on net performance)
    d["ret_net"] = d["equity_net"].pct_change().fillna(0.0)

    return d


def summarise(df: pd.DataFrame, equity_col: str, ret_col: str) -> dict:
    out = {}
    out["CAGR"] = cagr(df[equity_col], df["date"])
    out["AnnVol"] = annualised_vol(df[ret_col])
    out["Sharpe"] = sharpe_ratio(df[ret_col])
    out["MaxDD"] = max_drawdown(df[equity_col])
    out["EndEquity"] = float(df[equity_col].iloc[-1])
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to walk-forward backtest CSV")
    ap.add_argument("--fee", type=float, default=12.0, help="Fixed fee per order (currency)")
    ap.add_argument("--capital", type=float, default=100000.0, help="Starting capital (currency)")
    ap.add_argument("--out", default="gold_validation2_costs_fixed_fee_summary.csv", help="Output summary CSV")
    ap.add_argument("--curve_out", default="gold_validation2_costs_fixed_fee_curve.csv", help="Output equity curve CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise KeyError("CSV must include a 'date' column.")
    if "strategy_ret" not in df.columns:
        raise KeyError("CSV must include 'strategy_ret' (daily decimal returns).")
    if "position" not in df.columns:
        raise KeyError("CSV must include 'position'.")

    curve = build_equity_with_fixed_fees(
        dates=df["date"],
        daily_ret=df["strategy_ret"],
        position=df["position"],
        start_capital=args.capital,
        fee_per_order=args.fee,
    )

    # Core behavioural stats (unchanged by costs)
    switches = count_switches(curve["position"])
    tim = time_in_market(curve["position"])
    total_orders = float(curve["orders"].sum())
    total_fees = float(curve["fee_paid"].sum())

    gross = summarise(curve, "equity_gross", "ret")
    net = summarise(curve, "equity_net", "ret_net")

    summary = pd.DataFrame([{
        "start_date": str(curve["date"].iloc[0].date()),
        "end_date": str(curve["date"].iloc[-1].date()),
        "start_capital": args.capital,
        "fee_per_order": args.fee,
        "switches": switches,
        "time_in_market": tim,
        "total_orders": total_orders,
        "total_fees_paid": total_fees,

        "gross_CAGR": gross["CAGR"],
        "gross_AnnVol": gross["AnnVol"],
        "gross_Sharpe": gross["Sharpe"],
        "gross_MaxDD": gross["MaxDD"],
        "gross_EndEquity": gross["EndEquity"],

        "net_CAGR": net["CAGR"],
        "net_AnnVol": net["AnnVol"],
        "net_Sharpe": net["Sharpe"],
        "net_MaxDD": net["MaxDD"],
        "net_EndEquity": net["EndEquity"],

        "fee_drag_pct_of_end_equity": (total_fees / net["EndEquity"]) if net["EndEquity"] else np.nan,
        "fee_drag_pct_of_start_capital": (total_fees / args.capital) if args.capital else np.nan,
    }])

    # Pretty console output
    def pct(x): return f"{x*100:6.2f}%" if pd.notna(x) else "  nan "
    def num(x): return f"{x:7.2f}" if pd.notna(x) else "  nan "

    print("\n=== Validation 2: fixed fee costs on walk-forward ===")
    print(f"Period: {summary.loc[0,'start_date']} → {summary.loc[0,'end_date']}")
    print(f"Start capital: £{args.capital:,.0f} | Fee per order: £{args.fee:.2f}")
    print(f"Time in market: {pct(tim)} | Switches: {switches} | Total orders: {total_orders:.0f}")
    print(f"Total fees paid: £{total_fees:,.2f}")

    print("\n--- GROSS (no costs) ---")
    print(f"CAGR={pct(gross['CAGR'])} | Vol={pct(gross['AnnVol'])} | Sharpe={num(gross['Sharpe'])} | MaxDD={pct(gross['MaxDD'])} | EndEquity=£{gross['EndEquity']:,.2f}")

    print("\n--- NET (with fixed fees) ---")
    print(f"CAGR={pct(net['CAGR'])} | Vol={pct(net['AnnVol'])} | Sharpe={num(net['Sharpe'])} | MaxDD={pct(net['MaxDD'])} | EndEquity=£{net['EndEquity']:,.2f}")

    summary.to_csv(args.out, index=False)
    curve.to_csv(args.curve_out, index=False)
    print(f"\nSaved summary: {args.out}")
    print(f"Saved curve:   {args.curve_out}")


if __name__ == "__main__":
    main()
