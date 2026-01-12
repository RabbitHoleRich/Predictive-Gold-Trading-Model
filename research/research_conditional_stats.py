# research_conditional_stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolRegimeResult:
    table: pd.DataFrame
    thresholds: Dict[str, float]


def compute_oos_volatility_regime_profile(
    oos_df: pd.DataFrame,
    window: int = 63,
    q: Tuple[float, float] = (1/3, 2/3),
    ret_col: str = "strategy_ret",
    equity_col: str = "strategy_equity",
    pos_col: str = "position",
    date_col: str = "date",
    annualise: int = 252,
) -> VolRegimeResult:
    """
    OOS-only conditional performance by realised-volatility regime.

    Regime definition:
      - vol_t = rolling std(ret_col, window) * sqrt(annualise)
      - thresholds computed on OOS vol_t (excluding NaNs)
      - low / medium / high buckets by OOS quantiles (q1, q2)

    Returns:
      - VolRegimeResult(table=DataFrame, thresholds=dict)
    """

    df = oos_df.copy()
    if date_col in df.columns:
        df = df.sort_values(date_col)

    # --- Validate required columns
    missing = [c for c in [ret_col, equity_col, pos_col, date_col] if c not in df.columns]
    if missing:
        raise KeyError(f"compute_oos_volatility_regime_profile: missing columns: {missing}")

    # --- Rolling realised vol (annualised)
    vol = df[ret_col].astype(float).rolling(window).std(ddof=1) * np.sqrt(annualise)

    vol_clean = vol.dropna()
    if vol_clean.empty:
        # Not enough data to form regimes
        out = pd.DataFrame(
            columns=["n_days", "CAGR (%)", "Total Return (%)", "Sharpe", "Max DD (%)", "Hit Rate (%)", "Time in Market (%)"]
        )
        return VolRegimeResult(table=out, thresholds={"q1": np.nan, "q2": np.nan})

    q1, q2 = vol_clean.quantile(q[0]), vol_clean.quantile(q[1])

    def label(v: float) -> str:
        if np.isnan(v):
            return np.nan
        if v <= q1:
            return "Low volatility"
        if v <= q2:
            return "Medium volatility"
        return "High volatility"

    df["_vol"] = vol
    df["_vol_regime"] = df["_vol"].apply(label)

    # Drop rows without a regime label (early window NaNs)
    dfg = df.dropna(subset=["_vol_regime"]).copy()

    # --- Metric calculator (mirrors your existing IS/OOS logic)
    def summarise(d: pd.DataFrame) -> dict:
        d = d.sort_values(date_col)

        eq0 = float(d[equity_col].iloc[0])
        eq1 = float(d[equity_col].iloc[-1])
        total_return = (eq1 / eq0 - 1.0) if eq0 > 0 else np.nan

        days = (pd.to_datetime(d[date_col].iloc[-1]) - pd.to_datetime(d[date_col].iloc[0])).days
        years = days / 365.25 if days > 0 else np.nan
        cagr = (eq1 / eq0) ** (1 / years) - 1 if (years and years > 0 and eq0 > 0 and eq1 > 0) else np.nan

        rets = d[ret_col].dropna().astype(float)
        if len(rets) > 1 and rets.std(ddof=1) > 0:
            sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(annualise)
        else:
            sharpe = np.nan

        dd = d[equity_col] / d[equity_col].cummax() - 1
        max_dd = float(dd.min()) if len(dd) else np.nan

        hit_rate = float((rets > 0).mean() * 100) if len(rets) else np.nan
        time_in_mkt = float((d[pos_col] != 0).mean() * 100)

        return {
            "n_days": int(len(d)),
            "CAGR (%)": float(cagr * 100) if np.isfinite(cagr) else np.nan,
            "Total Return (%)": float(total_return * 100) if np.isfinite(total_return) else np.nan,
            "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "Max DD (%)": float(max_dd * 100) if np.isfinite(max_dd) else np.nan,
            "Hit Rate (%)": float(hit_rate) if np.isfinite(hit_rate) else np.nan,
            "Time in Market (%)": float(time_in_mkt) if np.isfinite(time_in_mkt) else np.nan,
        }

    order = ["Low volatility", "Medium volatility", "High volatility"]
    rows = {k: summarise(dfg[dfg["_vol_regime"] == k]) for k in order if (dfg["_vol_regime"] == k).any()}
    table = pd.DataFrame(rows).T.reindex(order)

    return VolRegimeResult(
        table=table,
        thresholds={"q1": float(q1), "q2": float(q2), "window": float(window)},
    )
