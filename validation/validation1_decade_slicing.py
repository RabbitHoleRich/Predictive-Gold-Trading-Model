"""
validation_decade_slicing.py

Validation (no optimisation): decade / regime slicing on an EXISTING walk-forward backtest file.

Assumes your walk-forward output CSV contains (at minimum):
    - date (YYYY-MM-DD)
    - strategy_equity
    - strategy_ret   (daily strategy return, e.g. +0.002 = +0.2%)
    - position       (0/1 or -1/0/1 etc)
Optional:
    - bh_equity, bh_ret

Usage:
    python3 validation_decade_slicing.py --csv gold_walkforward_backtest.csv
    python3 validation_decade_slicing.py --csv gold_walkforward_backtest.csv --out my_slices.csv
"""

import argparse
import numpy as np
import pandas as pd


# ----------------------------
# Metrics helpers
# ----------------------------
def max_drawdown(equity: pd.Series) -> float:
    equity = equity.astype(float).copy()
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())  # negative number


def annualised_vol(daily_ret: pd.Series, periods: int = 252) -> float:
    r = daily_ret.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(daily_ret: pd.Series, periods: int = 252) -> float:
    r = daily_ret.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((r.mean() / std) * np.sqrt(periods))


def cagr_from_equity(equity: pd.Series, dates: pd.Series) -> float:
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


def count_switches(position: pd.Series) -> int:
    p = position.fillna(0).astype(float)
    return int((p.diff().fillna(0) != 0).sum())


def time_in_market(position: pd.Series) -> float:
    p = position.fillna(0).astype(float)
    return float((p != 0).mean())


# ----------------------------
# Slicing logic
# ----------------------------
def default_decade_slices(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Creates decade buckets spanning the data range, aligned to calendar decades:
        2000-01-01..2009-12-31, 2010-01-01..2019-12-31, etc.
    """
    slices = []
    start_year = int(start_date.year)
    end_year = int(end_date.year)

    # Start at the beginning of the decade containing start_year
    decade_start_year = (start_year // 10) * 10
    y = decade_start_year

    while y <= end_year:
        s = pd.Timestamp(year=y, month=1, day=1)
        e = pd.Timestamp(year=y + 9, month=12, day=31)
        slices.append((f"{y}s", s, e))
        y += 10

    return slices


def compute_slice_metrics(df: pd.DataFrame, label: str) -> dict:
    d = df.copy()
    out = {"slice": label}

    out["start"] = str(d["date"].iloc[0].date())
    out["end"] = str(d["date"].iloc[-1].date())
    out["n_days"] = int(len(d))

    # Strategy metrics
    out["cagr"] = cagr_from_equity(d["strategy_equity"], d["date"])
    out["ann_vol"] = annualised_vol(d["strategy_ret"])
    out["sharpe"] = sharpe_ratio(d["strategy_ret"])
    out["max_dd"] = max_drawdown(d["strategy_equity"])
    out["time_in_mkt"] = time_in_market(d["position"])
    out["switches"] = count_switches(d["position"])

    # Optional B&H metrics if present
    if "bh_equity" in d.columns and "bh_ret" in d.columns:
        out["bh_cagr"] = cagr_from_equity(d["bh_equity"], d["date"])
        out["bh_ann_vol"] = annualised_vol(d["bh_ret"])
        out["bh_sharpe"] = sharpe_ratio(d["bh_ret"])
        out["bh_max_dd"] = max_drawdown(d["bh_equity"])
    else:
        out["bh_cagr"] = np.nan
        out["bh_ann_vol"] = np.nan
        out["bh_sharpe"] = np.nan
        out["bh_max_dd"] = np.nan

    # Slice return (simple equity ratio)
    out["slice_return_pct"] = float((d["strategy_equity"].iloc[-1] / d["strategy_equity"].iloc[0] - 1.0) * 100.0)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to walk-forward backtest CSV")
    ap.add_argument("--out", default="gold_validation_decade_slices.csv", help="Output CSV filename")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise KeyError("CSV must include a 'date' column.")
    required = ["strategy_equity", "strategy_ret", "position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]

    slices = default_decade_slices(start_date, end_date)

    rows = []
    for label, s, e in slices:
        sub = df[(df["date"] >= s) & (df["date"] <= e)].copy()
        if len(sub) < 20:  # ignore tiny fragments
            continue
        rows.append(compute_slice_metrics(sub, label))

    # Add an "ALL" row for reference
    rows.append(compute_slice_metrics(df, "ALL"))

    res = pd.DataFrame(rows)

    # Pretty print
    show_cols = [
        "slice", "start", "end", "n_days",
        "cagr", "ann_vol", "sharpe", "max_dd",
        "time_in_mkt", "switches", "slice_return_pct",
        "bh_cagr", "bh_sharpe", "bh_max_dd",
    ]
    res_show = res[show_cols].copy()

    def fmt_pct(x):
        return f"{x*100:6.2f}%" if pd.notna(x) else "   nan "

    def fmt_num(x):
        return f"{x:7.2f}" if pd.notna(x) else "   nan "

    print("\n=== Validation: decade slicing (walk-forward) ===")
    for _, r in res_show.iterrows():
        print(
            f"{r['slice']:>6} | {r['start']} → {r['end']} | days={int(r['n_days']):4d} | "
            f"CAGR={fmt_pct(r['cagr'])} | vol={fmt_pct(r['ann_vol'])} | Sharpe={fmt_num(r['sharpe'])} | "
            f"MaxDD={fmt_pct(r['max_dd'])} | TiM={fmt_pct(r['time_in_mkt'])} | sw={int(r['switches']):4d} | "
            f"sliceRet={r['slice_return_pct']:8.2f}%"
        )

    res.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
