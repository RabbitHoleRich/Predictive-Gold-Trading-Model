import pandas as pd


def compute_underwater_drawdown_stats(df: pd.DataFrame) -> dict:
    stats = {}

    dd = df["strategy_equity"] / df["strategy_equity"].cummax() - 1
    bh_dd = df["bh_equity"] / df["bh_equity"].cummax() - 1

    # Time underwater
    stats["pct_time_underwater"] = (dd < 0).mean()
    stats["bh_pct_time_underwater"] = (bh_dd < 0).mean()

    # Recovery durations
    underwater = dd < 0
    streaks = underwater.groupby(underwater.ne(underwater.shift()).cumsum()).sum()
    stats["median_recovery_days"] = streaks.median()
    stats["max_recovery_days"] = streaks.max()

    # Drawdown “pain” (area under curve)
    stats["dd_area"] = abs(dd).sum()
    stats["bh_dd_area"] = abs(bh_dd).sum()

    return stats

