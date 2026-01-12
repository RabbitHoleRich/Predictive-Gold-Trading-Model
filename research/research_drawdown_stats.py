import pandas as pd

def compute_drawdown_depth_duration_stats(dd_df: pd.DataFrame) -> dict:
    """
    Computes statistics aligned with the drawdown depth vs duration chart.

    Expected columns in dd_df:
        - depth (negative, drawdown fraction)
        - duration (integer, days)
    """

    stats = {}

    # Relationship between depth and duration
    stats["spearman_corr"] = dd_df["depth"].corr(
        dd_df["duration"], method="spearman"
    )

    # Typical drawdown characteristics
    stats["median_depth"] = dd_df["depth"].median()
    stats["median_duration"] = dd_df["duration"].median()

    # Tail behaviour
    stats["p90_depth"] = dd_df["depth"].quantile(0.9)
    stats["p90_duration"] = dd_df["duration"].quantile(0.9)

    return stats
