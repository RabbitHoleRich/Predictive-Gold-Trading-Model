# research_recovery_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown_recovery_stats(
    oos_df: pd.DataFrame,
    equity_col: str = "strategy_equity",
    date_col: str = "date",
) -> dict:
    """
    Compute drawdown recovery statistics for the strategy (OOS only).

    Returns:
        {
            "episodes": DataFrame of individual drawdowns,
            "summary": dict of recovery statistics
        }
    """

    df = oos_df[[date_col, equity_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    equity = df[equity_col].astype(float).values
    dates = df[date_col].values

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    episodes = []
    in_dd = False
    peak_idx = None
    trough_idx = None

    for i in range(len(df)):
        if drawdown[i] < 0 and not in_dd:
            # Start of drawdown
            in_dd = True
            peak_idx = i - 1 if i > 0 else i
            trough_idx = i

        elif drawdown[i] < 0 and in_dd:
            # Update trough
            if drawdown[i] < drawdown[trough_idx]:
                trough_idx = i

        elif drawdown[i] == 0 and in_dd:
            # Recovery
            recovery_idx = i

            episodes.append(
                {
                    "peak_date": dates[peak_idx],
                    "trough_date": dates[trough_idx],
                    "recovery_date": dates[recovery_idx],
                    "drawdown_pct": drawdown[trough_idx] * 100,
                    "recovery_days": (dates[recovery_idx] - dates[trough_idx]).astype("timedelta64[D]").astype(int),
                }
            )

            in_dd = False
            peak_idx = None
            trough_idx = None

    # Handle unrecovered drawdown
    if in_dd and trough_idx is not None:
        episodes.append(
            {
                "peak_date": dates[peak_idx],
                "trough_date": dates[trough_idx],
                "recovery_date": None,
                "drawdown_pct": drawdown[trough_idx] * 100,
                "recovery_days": None,
            }
        )

    episodes_df = pd.DataFrame(episodes)

    # Summary stats
    recovered = episodes_df["recovery_days"].dropna()

    summary = {
        "n_drawdowns": int(len(episodes_df)),
        "n_recovered": int(recovered.count()),
        "n_unrecovered": int(episodes_df["recovery_days"].isna().sum()),
        "median_recovery_days": float(recovered.median()) if not recovered.empty else np.nan,
        "p75_recovery_days": float(recovered.quantile(0.75)) if not recovered.empty else np.nan,
        "max_recovery_days": float(recovered.max()) if not recovered.empty else np.nan,
        "max_drawdown_pct": float(episodes_df["drawdown_pct"].min()) if not episodes_df.empty else np.nan,
    }

    return {
        "episodes": episodes_df,
        "summary": summary,
    }
