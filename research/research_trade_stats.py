# research_trade_stats.py

import pandas as pd
import numpy as np

def compute_trade_duration_payoff_stats(trades: pd.DataFrame) -> dict:
    """
    Computes joint statistics that mirror the
    holding-period vs trade-return chart.
    """

    stats = {}

    # Spearman correlation (robust monotonic relationship)
    stats["spearman_corr"] = trades["holding_days"].corr(
        trades["trade_return_pct"], method="spearman"
    )

    # Bucket holding periods
    trades = trades.copy()
    trades["duration_bucket"] = pd.qcut(
        trades["holding_days"],
        q=[0, 0.5, 0.8, 1.0],
        labels=["short", "medium", "long"],
    )

    bucket = trades.groupby("duration_bucket")

    stats["bucket_summary"] = bucket.agg(
        n_trades=("trade_return_pct", "count"),
        median_return=("trade_return_pct", "median"),
        mean_return=("trade_return_pct", "mean"),
        win_rate=("trade_return_pct", lambda x: (x > 0).mean()),
    )

    # Tail contribution (top decile winners)
    top_decile = trades["trade_return_pct"].quantile(0.9)

    stats["tail_by_bucket"] = (
        trades.assign(is_top_decile=lambda df: df["trade_return_pct"] >= top_decile)
        .groupby("duration_bucket")["is_top_decile"]
        .mean()
    )

    return stats
