# research_stress_stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WorstRollingResult:
    table: pd.DataFrame


def _rolling_compound_return(rets: pd.Series, window: int) -> pd.Series:
    """
    Rolling compounded return over `window` days:
        Π(1+r) - 1

    Uses log1p for numerical stability.
    """
    r = rets.astype(float)
    # log(1+r) then rolling sum then exp-1
    log1p = np.log1p(r.replace([np.inf, -np.inf], np.nan))
    roll = log1p.rolling(window=window).sum()
    return np.expm1(roll)


def compute_oos_worst_rolling_periods(
    oos_df: pd.DataFrame,
    windows: Tuple[int, int, int] = (63, 126, 252),
    strat_ret_col: str = "strategy_ret",
    bh_ret_col: str = "bh_ret",
    date_col: str = "date",
) -> WorstRollingResult:
    """
    Compute worst rolling returns for Strategy vs Buy & Hold over
    3M/6M/12M windows (63/126/252 trading days), OOS only.

    Returns a table with:
      - Window (label)
      - Strategy worst return (%)
      - BH worst return (%)
      - Strategy worst period start/end dates
      - BH worst period start/end dates
    """

    df = oos_df.copy()

    # Validate columns
    need = [date_col, strat_ret_col, bh_ret_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"compute_oos_worst_rolling_periods missing columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    rows: List[Dict] = []

    for w in windows:
        # Rolling compounded returns
        strat_roll = _rolling_compound_return(df[strat_ret_col], w)
        bh_roll = _rolling_compound_return(df[bh_ret_col], w)

        # Strategy worst
        s_min = strat_roll.min(skipna=True)
        s_end_idx = strat_roll.idxmin(skipna=True) if np.isfinite(s_min) else None
        s_start_idx = (s_end_idx - (w - 1)) if s_end_idx is not None else None

        # BH worst
        b_min = bh_roll.min(skipna=True)
        b_end_idx = bh_roll.idxmin(skipna=True) if np.isfinite(b_min) else None
        b_start_idx = (b_end_idx - (w - 1)) if b_end_idx is not None else None

        def _range(start_idx, end_idx):
            if start_idx is None or end_idx is None:
                return (pd.NaT, pd.NaT)
            if start_idx < 0 or end_idx < 0 or end_idx >= len(df):
                return (pd.NaT, pd.NaT)
            return (df.loc[start_idx, date_col], df.loc[end_idx, date_col])

        s_start, s_end = _range(s_start_idx, s_end_idx)
        b_start, b_end = _range(b_start_idx, b_end_idx)

        # Human labels
        label = {63: "3M", 126: "6M", 252: "12M"}.get(w, f"{w}d")

        rows.append(
            {
                "Window": label,
                "Strategy worst (%)": float(s_min * 100) if np.isfinite(s_min) else np.nan,
                "BH worst (%)": float(b_min * 100) if np.isfinite(b_min) else np.nan,
                "Strategy start": s_start.date() if pd.notna(s_start) else None,
                "Strategy end": s_end.date() if pd.notna(s_end) else None,
                "BH start": b_start.date() if pd.notna(b_start) else None,
                "BH end": b_end.date() if pd.notna(b_end) else None,
                "Window_days": int(w),
            }
        )

    out = pd.DataFrame(rows).set_index("Window")
    return WorstRollingResult(table=out)


