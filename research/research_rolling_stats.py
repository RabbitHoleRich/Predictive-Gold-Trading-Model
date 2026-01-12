import pandas as pd
import numpy as np


def compute_rolling_sharpe_profile(rolling_df: pd.DataFrame) -> dict:
    stats = {}

    strat = rolling_df["strat_rolling_sharpe"]
    bh = rolling_df["bh_rolling_sharpe"]

    stats["pct_positive"] = (strat > 0).mean()
    stats["pct_above_one"] = (strat > 1).mean()
    stats["worst_sharpe"] = strat.min()

    # Trend (simple linear slope)
    x = np.arange(len(strat.dropna()))
    y = strat.dropna().values
    stats["sharpe_trend"] = np.polyfit(x, y, 1)[0] if len(y) > 10 else 0.0

    stats["mean_excess_sharpe"] = (strat - bh).mean()

    return stats
