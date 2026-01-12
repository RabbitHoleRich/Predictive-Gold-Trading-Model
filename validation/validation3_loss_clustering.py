"""
validation3_loss_clustering.py

Validation phase (no optimisation):
Loss clustering + survivability diagnostics on an EXISTING walk-forward series.

Inputs (CSV must include):
  - date
  - strategy_ret (daily decimal returns, e.g. 0.01 = +1%)
Optional:
  - equity_net / equity_gross (if you want to use your Validation2 curve output)
  - position (used only for switch/trade-level streaks if present)

Outputs:
  - gold_validation3_summary.csv
  - gold_validation3_worst_drawdowns.csv
  - gold_validation3_underwater_periods.csv
  - gold_validation3_rolling_windows.csv
  - gold_validation3_yearly_returns.csv

Usage:
  python3 validation3_loss_clustering.py --csv gold_walkforward_backtest.csv
  python3 validation3_loss_clustering.py --csv gold_validation2_costs_fixed_fee_curve.csv --equity_col equity_net --ret_col ret_net
"""

import argparse
import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def build_equity_from_returns(dates: pd.Series, daily_ret: pd.Series, start_capital: float = 100000.0) -> pd.Series:
    r = pd.to_numeric(daily_ret, errors="coerce").fillna(0.0).astype(float)
    eq = (1.0 + r).cumprod() * float(start_capital)
    eq.index = pd.RangeIndex(len(eq))
    return eq


def max_drawdown_details(equity: pd.Series, dates: pd.Series):
    """
    Returns:
      - max_dd (negative float)
      - peak_date, trough_date, recovery_date (may be NaT if not recovered)
      - peak_equity, trough_equity
      - dd_duration_days, time_to_recover_days (NaN if not recovered)
    """
    eq = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 2:
        return {}

    d = pd.DataFrame({"date": _to_dt(dates), "equity": eq}).dropna().reset_index(drop=True)
    d["peak"] = d["equity"].cummax()
    d["dd"] = d["equity"] / d["peak"] - 1.0

    trough_idx = int(d["dd"].idxmin())
    max_dd = float(d.loc[trough_idx, "dd"])

    # peak before trough
    peak_idx = int(d.loc[:trough_idx, "equity"].idxmax())

    peak_date = d.loc[peak_idx, "date"]
    trough_date = d.loc[trough_idx, "date"]
    peak_equity = float(d.loc[peak_idx, "equity"])
    trough_equity = float(d.loc[trough_idx, "equity"])

    # recovery: first date after trough where equity >= prior peak_equity
    rec = d.loc[trough_idx:].copy()
    rec_idx = rec.index[rec["equity"] >= peak_equity]
    if len(rec_idx) > 0:
        recovery_idx = int(rec_idx[0])
        recovery_date = d.loc[recovery_idx, "date"]
        time_to_recover_days = float((recovery_date - trough_date).days)
    else:
        recovery_idx = None
        recovery_date = pd.NaT
        time_to_recover_days = np.nan

    dd_duration_days = float((trough_date - peak_date).days)

    return {
        "max_dd": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "dd_duration_days": dd_duration_days,
        "time_to_recover_days": time_to_recover_days,
    }


def underwater_periods(equity: pd.Series, dates: pd.Series):
    """
    Underwater periods: intervals where equity is below its previous peak.
    Returns dataframe of periods with depth + duration.
    """
    d = pd.DataFrame({"date": _to_dt(dates), "equity": pd.to_numeric(equity, errors="coerce")}).dropna().sort_values("date")
    d["peak"] = d["equity"].cummax()
    d["underwater"] = d["equity"] < d["peak"]
    d["dd"] = d["equity"] / d["peak"] - 1.0

    periods = []
    in_uw = False
    start_i = None
    peak_equity = None
    peak_date = None

    for i in range(len(d)):
        uw = bool(d.iloc[i]["underwater"])
        if uw and not in_uw:
            in_uw = True
            start_i = i
            # peak point is last non-underwater day (or first day)
            j = max(i - 1, 0)
            peak_equity = float(d.iloc[j]["peak"])
            peak_date = d.iloc[j]["date"]
        elif (not uw) and in_uw:
            # end underwater at i-1
            end_i = i - 1
            seg = d.iloc[start_i:end_i + 1].copy()
            trough_i = int(seg["dd"].idxmin())
            trough_row = d.loc[trough_i]
            periods.append({
                "peak_date": peak_date,
                "recovery_date": d.iloc[i]["date"],
                "trough_date": trough_row["date"],
                "max_dd": float(seg["dd"].min()),
                "days_underwater": float((d.iloc[i]["date"] - peak_date).days),
                "days_peak_to_trough": float((trough_row["date"] - peak_date).days),
                "days_trough_to_recovery": float((d.iloc[i]["date"] - trough_row["date"]).days),
            })
            in_uw = False
            start_i = None

    # if still underwater at end
    if in_uw and start_i is not None:
        seg = d.iloc[start_i:].copy()
        trough_i = int(seg["dd"].idxmin())
        trough_row = d.loc[trough_i]
        periods.append({
            "peak_date": peak_date,
            "recovery_date": pd.NaT,
            "trough_date": trough_row["date"],
            "max_dd": float(seg["dd"].min()),
            "days_underwater": float((d.iloc[-1]["date"] - peak_date).days),
            "days_peak_to_trough": float((trough_row["date"] - peak_date).days),
            "days_trough_to_recovery": np.nan,
        })

    return pd.DataFrame(periods).sort_values(["max_dd", "days_underwater"]).reset_index(drop=True)


def rolling_window_stats(df: pd.DataFrame, equity_col: str, ret_col: str, windows_days=(63, 126, 252)):
    """
    Rolling stats:
      - rolling return over window
      - rolling max drawdown within window
      - rolling volatility (annualised)
    """
    out = df[["date"]].copy()
    r = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0).astype(float)
    eq = pd.to_numeric(df[equity_col], errors="coerce").fillna(method="ffill").fillna(method="bfill").astype(float)

    for w in windows_days:
        # rolling compounded return
        out[f"roll_{w}d_return"] = (1.0 + r).rolling(w).apply(lambda x: np.prod(x) - 1.0, raw=True)

        # rolling vol (ann.)
        out[f"roll_{w}d_vol"] = r.rolling(w).std(ddof=0) * np.sqrt(252)

        # rolling max drawdown inside window (compute on rolling equity)
        def _roll_mdd(x):
            x = np.asarray(x, dtype=float)
            peak = np.maximum.accumulate(x)
            dd = x / peak - 1.0
            return float(np.min(dd))

        out[f"roll_{w}d_maxdd"] = eq.rolling(w).apply(_roll_mdd, raw=True)

    return out


def longest_losing_streak(daily_ret: pd.Series):
    r = pd.to_numeric(daily_ret, errors="coerce").fillna(0.0).astype(float)
    neg = (r < 0).astype(int)
    # lengths of consecutive 1s
    max_len = 0
    cur = 0
    for v in neg:
        if v == 1:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return int(max_len)


def trade_streaks_from_position(df: pd.DataFrame):
    """
    If position exists, approximate "trades" as intervals of non-zero position.
    Return longest consecutive losing trades, win rate, avg trade return, etc.
    """
    if "position" not in df.columns:
        return {}

    d = df.copy()
    d["position"] = pd.to_numeric(d["position"], errors="coerce").fillna(0.0)

    # identify trade segments by entry/exit
    pos = d["position"].values
    in_trade = False
    start = None
    trades = []

    for i in range(len(d)):
        if not in_trade and pos[i] != 0:
            in_trade = True
            start = i
        elif in_trade and pos[i] == 0:
            end = i - 1
            seg = d.iloc[start:end + 1]
            tr = (1.0 + pd.to_numeric(seg["strategy_ret"], errors="coerce").fillna(0.0)).prod() - 1.0
            trades.append(float(tr))
            in_trade = False
            start = None

    # if still in trade at end
    if in_trade and start is not None:
        seg = d.iloc[start:]
        tr = (1.0 + pd.to_numeric(seg["strategy_ret"], errors="coerce").fillna(0.0)).prod() - 1.0
        trades.append(float(tr))

    if len(trades) == 0:
        return {}

    trades = np.array(trades, dtype=float)
    wins = trades[trades > 0]
    losses = trades[trades <= 0]

    # longest consecutive losing trades
    longest = 0
    cur = 0
    for t in trades:
        if t <= 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    return {
        "n_trades": int(len(trades)),
        "trade_win_rate": float((trades > 0).mean()),
        "avg_trade_return": float(trades.mean()),
        "median_trade_return": float(np.median(trades)),
        "worst_trade_return": float(trades.min()),
        "best_trade_return": float(trades.max()),
        "longest_consec_losing_trades": int(longest),
        "avg_win_trade": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_trade": float(losses.mean()) if len(losses) else np.nan,
    }

def yearly_returns(df: pd.DataFrame, ret_col: str):
    d = df.copy()
    d["date"] = _to_dt(d["date"])
    d = d.dropna(subset=["date"])
    d["year"] = d["date"].dt.year

    r = pd.to_numeric(d[ret_col], errors="coerce").fillna(0.0)

    yr = (1.0 + r).groupby(d["year"]).prod() - 1.0
    yr = yr.rename("year_return").reset_index()   # <-- this is the key line
    return yr


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (walk-forward backtest or validation2 curve)")
    ap.add_argument("--equity_col", default="", help="Equity column to use (optional). If blank, rebuild from ret_col.")
    ap.add_argument("--ret_col", default="strategy_ret", help="Return column to use (default: strategy_ret)")
    ap.add_argument("--start_capital", type=float, default=100000.0, help="Starting capital if rebuilding equity")
    ap.add_argument("--out_prefix", default="gold_validation3", help="Output file prefix")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise KeyError("CSV must include a 'date' column.")
    if args.ret_col not in df.columns:
        raise KeyError(f"CSV must include return column '{args.ret_col}'.")

    df["date"] = _to_dt(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Choose equity series
    if args.equity_col and args.equity_col in df.columns:
        df["equity_used"] = pd.to_numeric(df[args.equity_col], errors="coerce")
        df["ret_used"] = pd.to_numeric(df[args.ret_col], errors="coerce").fillna(0.0)
        # ensure equity has no gaps
        df["equity_used"] = df["equity_used"].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    else:
        df["ret_used"] = pd.to_numeric(df[args.ret_col], errors="coerce").fillna(0.0)
        df["equity_used"] = build_equity_from_returns(df["date"], df["ret_used"], start_capital=args.start_capital)

    # Overall DD details
    dd = max_drawdown_details(df["equity_used"], df["date"])

    # Underwater periods
    uw = underwater_periods(df["equity_used"], df["date"])
    uw_top = uw.sort_values("max_dd").head(10).copy()

    # Rolling window worsts
    roll = rolling_window_stats(df, "equity_used", "ret_used", windows_days=(63, 126, 252))
    # Worst rolling returns
    worst_3m = float(roll["roll_63d_return"].min())
    worst_6m = float(roll["roll_126d_return"].min())
    worst_1y = float(roll["roll_252d_return"].min())
    # Worst rolling maxDD
    worst_3m_mdd = float(roll["roll_63d_maxdd"].min())
    worst_6m_mdd = float(roll["roll_126d_maxdd"].min())
    worst_1y_mdd = float(roll["roll_252d_maxdd"].min())

    # Daily losing streak
    longest_daily_ls = longest_losing_streak(df["ret_used"])

    # Trade streaks (if position exists) — build a clean minimal DF to avoid duplicate columns
    if "position" in df.columns:
        tmp_for_trades = df[["date", "position"]].copy()
        tmp_for_trades["strategy_ret"] = df["ret_used"].values  # always a 1D series
        trade_stats = trade_streaks_from_position(tmp_for_trades)
    else:
        trade_stats = {}


    # Yearly returns
    yr = yearly_returns(df, "ret_used")
    best_year = yr.loc[yr["year_return"].idxmax()]
    worst_year = yr.loc[yr["year_return"].idxmin()]

    # Summary
    summary = {
        "start_date": str(df["date"].iloc[0].date()),
        "end_date": str(df["date"].iloc[-1].date()),
        "n_days": int(len(df)),
        "max_dd": dd.get("max_dd", np.nan),
        "dd_peak_date": dd.get("peak_date", pd.NaT),
        "dd_trough_date": dd.get("trough_date", pd.NaT),
        "dd_recovery_date": dd.get("recovery_date", pd.NaT),
        "dd_duration_days": dd.get("dd_duration_days", np.nan),
        "dd_time_to_recover_days": dd.get("time_to_recover_days", np.nan),
        "longest_consec_losing_days": longest_daily_ls,
        "worst_3m_return": worst_3m,
        "worst_6m_return": worst_6m,
        "worst_1y_return": worst_1y,
        "worst_3m_maxdd": worst_3m_mdd,
        "worst_6m_maxdd": worst_6m_mdd,
        "worst_1y_maxdd": worst_1y_mdd,
        "best_year": int(best_year["year"]),
        "best_year_return": float(best_year["year_return"]),
        "worst_year": int(worst_year["year"]),
        "worst_year_return": float(worst_year["year_return"]),
        "top10_underwater_count": int(min(10, len(uw))),
        "longest_underwater_days": float(uw["days_underwater"].max()) if len(uw) else np.nan,
        "deepest_underwater_dd": float(uw["max_dd"].min()) if len(uw) else np.nan,
    }
    summary.update(trade_stats)

    # Save outputs
    prefix = args.out_prefix
    pd.DataFrame([summary]).to_csv(f"{prefix}_summary.csv", index=False)
    uw.to_csv(f"{prefix}_underwater_periods.csv", index=False)
    uw_top.to_csv(f"{prefix}_worst_drawdowns.csv", index=False)
    roll.to_csv(f"{prefix}_rolling_windows.csv", index=False)
    yr.to_csv(f"{prefix}_yearly_returns.csv", index=False)

    # Console output (tight + useful)
    def pct(x): return f"{x*100:6.2f}%" if pd.notna(x) else "  nan "
    def days(x): return f"{x:7.0f}" if pd.notna(x) else "   nan "

    print("\n=== Validation 3: loss clustering & survivability ===")
    print(f"Period: {summary['start_date']} → {summary['end_date']} | days={summary['n_days']}")
    print(f"MaxDD: {pct(summary['max_dd'])} | Peak={summary['dd_peak_date']} | Trough={summary['dd_trough_date']} | Recovery={summary['dd_recovery_date']}")
    print(f"DD duration (peak→trough): {days(summary['dd_duration_days'])} days | Time to recover: {days(summary['dd_time_to_recover_days'])} days")
    print(f"Longest consecutive losing days: {summary['longest_consec_losing_days']}")

    print("\n--- Worst rolling returns ---")
    print(f"Worst 3M (63d):  {pct(summary['worst_3m_return'])} | Worst 6M (126d): {pct(summary['worst_6m_return'])} | Worst 1Y (252d): {pct(summary['worst_1y_return'])}")

    print("\n--- Worst rolling drawdowns inside window ---")
    print(f"Worst 3M MDD:    {pct(summary['worst_3m_maxdd'])} | Worst 6M MDD:   {pct(summary['worst_6m_maxdd'])} | Worst 1Y MDD:   {pct(summary['worst_1y_maxdd'])}")

    print("\n--- Yearly return extremes ---")
    print(f"Best year:  {summary['best_year']}  ({pct(summary['best_year_return'])})")
    print(f"Worst year: {summary['worst_year']} ({pct(summary['worst_year_return'])})")

    print("\n--- Underwater periods ---")
    print(f"Deepest underwater: {pct(summary['deepest_underwater_dd'])} | Longest underwater: {days(summary['longest_underwater_days'])} days")
    print(f"Saved: {prefix}_summary.csv, {prefix}_worst_drawdowns.csv, {prefix}_underwater_periods.csv, {prefix}_rolling_windows.csv, {prefix}_yearly_returns.csv")

    if "n_trades" in summary:
        print("\n--- Trade-level stats (approx from position) ---")
        print(f"Trades: {summary['n_trades']} | Win rate: {pct(summary['trade_win_rate'])} | Longest losing trades streak: {summary['longest_consec_losing_trades']}")
        print(f"Avg trade: {pct(summary['avg_trade_return'])} | Worst trade: {pct(summary['worst_trade_return'])} | Best trade: {pct(summary['best_trade_return'])}")


if __name__ == "__main__":
    main()
