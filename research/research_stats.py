"""
research_stats.py

Statistical computation layer for the research engine.

This is a direct modularisation of the original stats logic from research_layer.py.

It computes:
- PerformanceSummary
- Max drawdown
- Trade extraction
- Rolling stats (Sharpe, max DD, CAGR)
- Welch-style IS/OOS t-test
- Sharpe t-stats

All functions are PURE: they do not read/write files or plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# ================================================================
# Dataclass: PerformanceSummary
# ================================================================
@dataclass
class PerformanceSummary:
    """
    High-level performance snapshot for the strategy and Buy & Hold.
    All returns are expressed in decimal form (e.g. 0.25 = +25%).
    """

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    n_years: float

    strat_cagr: float
    bh_cagr: float

    strat_total_return: float
    bh_total_return: float

    strat_max_dd: float
    bh_max_dd: float

    strat_sharpe: float
    bh_sharpe: float

    time_in_market: float
    hit_rate: float
    avg_trade_return: float
    n_trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_days": self.n_days,
            "n_years": self.n_years,
            "strat_cagr": self.strat_cagr,
            "bh_cagr": self.bh_cagr,
            "strat_total_return": self.strat_total_return,
            "bh_total_return": self.bh_total_return,
            "strat_max_dd": self.strat_max_dd,
            "bh_max_dd": self.bh_max_dd,
            "strat_sharpe": self.strat_sharpe,
            "bh_sharpe": self.bh_sharpe,
            "time_in_market": self.time_in_market,
            "hit_rate": self.hit_rate,
            "avg_trade_return": self.avg_trade_return,
            "n_trades": self.n_trades,
        }


# ================================================================
# Core helpers
# ================================================================
def max_drawdown(equity: np.ndarray) -> float:
    """
    Max drawdown on an equity curve (same as _max_drawdown in original code).
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return np.nan
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple trade extraction:

    - A trade starts when position goes from 0 -> 1 (or -1)
    - A trade ends when position goes back to 0 or flips sign.

    Assumes:
        - 'position' column exists
        - 'strategy_ret' is daily percentage return (e.g. 0.01 = +1%)
    """
    if "position" not in df.columns or "strategy_ret" not in df.columns:
        return pd.DataFrame()

    pos = df["position"].values
    rets = df["strategy_ret"].values
    dates = df["date"].values

    trades = []
    current_pos = 0
    entry_idx: Optional[int] = None
    cum_return = 0.0

    for i in range(len(df)):
        if current_pos == 0 and pos[i] != 0:
            # Enter new trade
            current_pos = pos[i]
            entry_idx = i
            cum_return = rets[i] * current_pos
        elif current_pos != 0:
            if pos[i] == current_pos:
                cum_return += rets[i] * current_pos
            else:
                # Position closed or flipped
                exit_idx = i
                trades.append(
                    {
                        "entry_date": dates[entry_idx],
                        "exit_date": dates[exit_idx],
                        "direction": int(current_pos),
                        "holding_days": exit_idx - entry_idx + 1,
                        "trade_return": cum_return,
                    }
                )
                # New trade if new position is non-zero
                if pos[i] != 0:
                    current_pos = pos[i]
                    entry_idx = i
                    cum_return = rets[i] * current_pos
                else:
                    current_pos = 0
                    entry_idx = None
                    cum_return = 0.0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    return trades_df


# ================================================================
# Performance summary (direct port from _compute_performance_summary)
# ================================================================
def compute_performance_summary(
    backtest_df: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> PerformanceSummary:
    """
    Compute a PerformanceSummary from the backtest DataFrame.

    Expects columns:
        - 'date'
        - 'strategy_equity'
        - 'bh_equity'
        - 'strategy_ret'
        - 'bh_ret'
        - 'position'
    """

    df = backtest_df.copy()
    df = df.sort_values("date")
    start = df["date"].iloc[0]
    end = df["date"].iloc[-1]

    n_days = (end - start).days
    n_years = n_days / 365.25 if n_days > 0 else 0.0

   # Normalise equity curves (MUST match chart logic)
   
    strat_eq_start = float(df["strategy_equity"].iloc[0])
    strat_eq_end = float(df["strategy_equity"].iloc[-1])
    bh_eq_start = float(df["bh_equity"].iloc[0])
    bh_eq_end = float(df["bh_equity"].iloc[-1])

    strat_total_return = strat_eq_end / strat_eq_start - 1.0
    bh_total_return = bh_eq_end / bh_eq_start - 1.0

    def safe_cagr(eq_start: float, eq_end: float, years: float) -> float:
        if eq_start <= 0 or eq_end <= 0 or years <= 0:
            return np.nan
        return (eq_end / eq_start) ** (1.0 / years) - 1.0

    strat_cagr = safe_cagr(strat_eq_start, strat_eq_end, n_years)
    bh_cagr = safe_cagr(bh_eq_start, bh_eq_end, n_years)

    # Daily returns
    strat_ret = df["strategy_ret"].values
    bh_ret = df["bh_ret"].values

    def annualised_sharpe(rets: np.ndarray) -> float:
        rets = np.asarray(rets, dtype=float)
        rets = rets[~np.isnan(rets)]
        if rets.size < 2:
            return np.nan
        mu = rets.mean()
        sigma = rets.std(ddof=1)
        if sigma == 0:
            return np.nan
        return (mu / sigma) * np.sqrt(trading_days_per_year)

    strat_sharpe = annualised_sharpe(strat_ret)
    bh_sharpe = annualised_sharpe(bh_ret)

    # Max drawdowns (on provided equity curve)
    strat_max_dd = max_drawdown(df["strategy_equity"].values)
    bh_max_dd = max_drawdown(df["bh_equity"].values)

    # Time in market (long-only assumption, non-zero position means in market)
    time_in_market = float((df["position"] != 0).mean()) if "position" in df.columns else np.nan

    # Hit rate and trade stats
    trades_df = extract_trades(df)
    if not trades_df.empty:
        hit_rate = float((trades_df["trade_return"] > 0).mean())
        avg_trade_return = float(trades_df["trade_return"].mean())
        n_trades = int(len(trades_df))
    else:
        hit_rate = np.nan
        avg_trade_return = np.nan
        n_trades = 0

    return PerformanceSummary(
        start_date=start,
        end_date=end,
        n_days=n_days,
        n_years=n_years,
        strat_cagr=strat_cagr,
        bh_cagr=bh_cagr,
        strat_total_return=strat_total_return,
        bh_total_return=bh_total_return,
        strat_max_dd=strat_max_dd,
        bh_max_dd=bh_max_dd,
        strat_sharpe=strat_sharpe,
        bh_sharpe=bh_sharpe,
        time_in_market=time_in_market,
        hit_rate=hit_rate,
        avg_trade_return=avg_trade_return,
        n_trades=n_trades,
    )


# ================================================================
# Rolling stats (ported from _compute_rolling_stats)
# ================================================================
def compute_rolling_stats(
    backtest_df: pd.DataFrame,
    trading_days_per_year: int = 252,
    window: int = 189,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling diagnostics:

        - strat_rolling_sharpe / bh_rolling_sharpe
        - strat_rolling_sharpe_smooth / bh_rolling_sharpe_smooth
        - strat_rolling_max_dd / bh_rolling_max_dd
        - strat_rolling_cagr_approx / bh_rolling_cagr_approx
        - strat_rolling_cagr_smooth / bh_rolling_cagr_smooth

    Expects:
        - 'date', 'strategy_ret', 'bh_ret', 'strategy_equity', 'bh_equity'
    """

    df = backtest_df.sort_values("date").reset_index(drop=True)

    # ------------------------
    # Inner helper functions
    # ------------------------
    def rolling_sharpe(x: pd.Series) -> float:
        mu = x.mean()
        sigma = x.std(ddof=1)
        if sigma == 0:
            return np.nan
        return (mu / sigma) * np.sqrt(trading_days_per_year)

    def rolling_max_dd_on_equity_window(equity_window: pd.Series) -> float:
        if equity_window.isna().all():
            return np.nan
        arr = equity_window.values
        roll_max = np.maximum.accumulate(arr)
        dd = arr / roll_max - 1.0
        return dd.min()

    def rolling_cagr(x: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        total_ret = (1.0 + x).prod() - 1.0
        years = len(x) / trading_days_per_year
        if years <= 0:
            return np.nan
        return (1.0 + total_ret) ** (1.0 / years) - 1.0

    # ------------------------
    # Rolling computations
    # ------------------------
    out = pd.DataFrame({"date": df["date"].values})

    # Rolling Sharpe
    out["strat_rolling_sharpe"] = (
        df["strategy_ret"].rolling(window).apply(rolling_sharpe, raw=False)
    )
    out["bh_rolling_sharpe"] = (
        df["bh_ret"].rolling(window).apply(rolling_sharpe, raw=False)
    )

    # Smooth Sharpe
    out["strat_rolling_sharpe_smooth"] = (
        out["strat_rolling_sharpe"].rolling(smooth_window).mean()
    )
    out["bh_rolling_sharpe_smooth"] = (
        out["bh_rolling_sharpe"].rolling(smooth_window).mean()
    )

    # Rolling Max Drawdown
    out["strat_rolling_max_dd"] = (
        df["strategy_equity"].rolling(window).apply(
            rolling_max_dd_on_equity_window, raw=False
        )
    )
    out["bh_rolling_max_dd"] = (
        df["bh_equity"].rolling(window).apply(
            rolling_max_dd_on_equity_window, raw=False
        )
    )

    # Rolling CAGR
    out["strat_rolling_cagr_approx"] = (
        df["strategy_ret"].rolling(window).apply(rolling_cagr, raw=False)
    )
    out["bh_rolling_cagr_approx"] = (
        df["bh_ret"].rolling(window).apply(rolling_cagr, raw=False)
    )

    # Smooth CAGR
    out["strat_rolling_cagr_smooth"] = (
        out["strat_rolling_cagr_approx"].rolling(smooth_window).mean()
    )
    out["bh_rolling_cagr_smooth"] = (
        out["bh_rolling_cagr_approx"].rolling(smooth_window).mean()
    )

    return out


# ================================================================
# IS/OOS Welch t-test & Sharpe t-stats
# ================================================================
def welch_ttest_is_oos(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    ret_col: str = "strategy_ret",
) -> Dict[str, float]:
    """
    Perform a Welch-style two-sample t-test comparing mean daily strategy
    returns in-sample vs out-of-sample.

    Returns a dict with:
        - t_stat:   t statistic (OOS - IS)
        - df:       approximate degrees of freedom
        - p_value:  approximate two-sided p-value (normal approx if df large)
        - mean_is:  mean IS daily return
        - mean_oos: mean OOS daily return
        - n_is:     number of IS days
        - n_oos:    number of OOS days
    """

    rets_is = df_is[ret_col].dropna().astype(float).values
    rets_oos = df_oos[ret_col].dropna().astype(float).values

    n1, n2 = len(rets_is), len(rets_oos)
    if n1 < 2 or n2 < 2:
        return {
            "t_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "mean_is": np.nan,
            "mean_oos": np.nan,
            "n_is": n1,
            "n_oos": n2,
        }

    m1, m2 = rets_is.mean(), rets_oos.mean()
    v1, v2 = rets_is.var(ddof=1), rets_oos.var(ddof=1)

    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        t_stat = np.nan
        df = np.nan
    else:
        t_stat = (m2 - m1) / np.sqrt(se2)

        # Welch–Satterthwaite df
        num = se2**2
        den = (v1**2) / ((n1**2) * (n1 - 1)) + (v2**2) / ((n2**2) * (n2 - 1))
        df = num / den if den > 0 else np.nan

    # Approximate two-sided p-value using normal approximation if df large
    if np.isnan(t_stat):
        p_value = np.nan
    else:
        z = abs(t_stat)
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        p_value = 2.0 * (1.0 - cdf)

    return {
        "t_stat": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "mean_is": float(m1),
        "mean_oos": float(m2),
        "n_is": int(n1),
        "n_oos": int(n2),
    }


def sharpe_tstat_from_perf(perf: PerformanceSummary) -> float:
    """
    Compute a t-statistic for the annualised Sharpe using a PerformanceSummary.

        t ≈ Sharpe_ann * sqrt(N / 252)
    """
    sharpe = perf.strat_sharpe
    n_days = perf.n_days

    if n_days is None or n_days < 2 or not np.isfinite(sharpe):
        return np.nan

    return sharpe * np.sqrt(n_days / 252.0)


def compute_sharpe_tstat(cagr: float, sharpe: float, n_days: float) -> float:
    """
    Compute the t-statistic for an annualised Sharpe estimate:

        t ≈ Sharpe * sqrt(N / 252)
    """
    if n_days <= 2 or sharpe is None or np.isnan(sharpe):
        return np.nan

    return sharpe * np.sqrt(n_days / 252.0)

# ================================================================
# IS/OOS SUMMARY METRICS (MODULAR)
# ================================================================

def compute_is_oos_stats(df_is: pd.DataFrame, df_oos: pd.DataFrame) -> dict:
    """
    Compute simple IS vs OOS performance summaries using the same
    logic as compute_performance_summary but without the full dataclass
    overhead.

    Both dataframes must contain:
        - date
        - strategy_equity
        - strategy_ret
        - position
    """

    def metrics(df: pd.DataFrame) -> dict:
        df = df.sort_values("date")

        eq0 = df["strategy_equity"].iloc[0]
        eq1 = df["strategy_equity"].iloc[-1]
        total_ret = eq1 / eq0 - 1

        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        years = days / 365.25 if days > 0 else np.nan
        cagr = (eq1 / eq0) ** (1 / years) - 1 if years > 0 else np.nan

        rets = df["strategy_ret"].dropna()
        if len(rets) > 1 and rets.std(ddof=1) > 0:
            sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252)
        else:
            sharpe = np.nan

        dd = df["strategy_equity"] / df["strategy_equity"].cummax() - 1
        max_dd = dd.min()

        hit_rate = (rets > 0).mean() * 100 if len(rets) else np.nan
        time_in_mkt = (df["position"] != 0).mean() * 100

        return {
            "CAGR": float(cagr * 100),
            "Total_Return": float(total_ret * 100),
            "Sharpe": float(sharpe),
            "Max_DD": float(max_dd * 100),
            "Hit_Rate": float(hit_rate),
            "Time_in_Market": float(time_in_mkt),
        }

    return {
        "is": metrics(df_is),
        "oos": metrics(df_oos),
    }

def compute_is_oos_profile(df_is: pd.DataFrame, df_oos: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side In-sample vs Out-of-sample performance table.
    Mirrors compute_is_oos_stats but returns a DataFrame.
    """

    def summarise(df: pd.DataFrame) -> dict:
        df = df.sort_values("date")

        eq0 = df["strategy_equity"].iloc[0]
        eq1 = df["strategy_equity"].iloc[-1]
        total_return = eq1 / eq0 - 1.0

        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        years = days / 365.25 if days > 0 else float("nan")
        cagr = (eq1 / eq0) ** (1 / years) - 1 if years > 0 else float("nan")

        rets = df["strategy_ret"].dropna()
        if len(rets) > 1 and rets.std(ddof=1) > 0:
            sharpe = (rets.mean() / rets.std(ddof=1)) * (252 ** 0.5)
        else:
            sharpe = float("nan")

        dd = df["strategy_equity"] / df["strategy_equity"].cummax() - 1
        max_dd = dd.min()

        hit_rate = (rets > 0).mean() * 100 if len(rets) else float("nan")
        time_in_market = (df["position"] != 0).mean() * 100

        return {
            "CAGR (%)": cagr * 100,
            "Total Return (%)": total_return * 100,
            "Sharpe": sharpe,
            "Max DD (%)": max_dd * 100,
            "Hit Rate (%)": hit_rate,
            "Time in Market (%)": time_in_market,
        }

    table = {
        "In-sample": summarise(df_is),
        "Out-of-sample": summarise(df_oos),
    }

    return pd.DataFrame(table).T


# ================================================================
# SUBPERIOD PERFORMANCE (MODULAR)
# ================================================================

def compute_subperiod_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance across major multi-year subperiods.
    Returns a DataFrame indexed by period label.
    """

    df = df.sort_values("date")

    def eval_period(start, end):
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        sub = df.loc[mask]

        if len(sub) < 10:
            return {"CAGR": np.nan, "Sharpe": np.nan, "Total_Return": np.nan}

        eq0 = sub["strategy_equity"].iloc[0]
        eq1 = sub["strategy_equity"].iloc[-1]
        total_ret = eq1 / eq0 - 1

        days = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days
        years = days / 365.25 if days > 0 else np.nan
        cagr = (eq1 / eq0) ** (1 / years) - 1 if years > 0 else np.nan

        rets = sub["strategy_ret"].dropna()
        sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252) if len(rets) > 1 else np.nan

        return {
            "CAGR": float(cagr * 100),
            "Sharpe": float(sharpe),
            "Total_Return": float(total_ret * 100),
        }

    periods = [
        ("2001-01-01", "2007-12-31"),
        ("2008-01-01", "2013-12-31"),
        ("2014-01-01", "2019-12-31"),
        ("2020-01-01", "2025-12-31"),
    ]

    result = {f"{s[:4]}-{e[:4]}": eval_period(s, e) for s, e in periods}

    return pd.DataFrame(result).T



    







