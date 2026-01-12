import pandas as pd

def compute_heatmap_profile(wf_df: pd.DataFrame) -> dict:
    """
    Stats and table aligned to the walk-forward heatmap.
    Expects monthly walk-forward summary with 'month' and 'strat_ret_pct'.
    """

    df = wf_df.copy()
    df["test_month"] = pd.to_datetime(df["test_month"])

    # -----------------------------
    # Heatmap table
    # -----------------------------
    df["year"] = df["test_month"].dt.year
    df["calendar_month"] = df["test_month"].dt.month

    heatmap_table = (
        df.pivot_table(
            index="year",
            columns="calendar_month",
            values="strat_ret_pct",
            aggfunc="mean",
        )
        .sort_index()
    )

    heatmap_table.columns = heatmap_table.columns.astype(int)

    # -----------------------------
    # Summary stats
    # -----------------------------
    stats = {
        "n_periods": int(len(df)),
        "positive_months": int((df["strat_ret_pct"] > 0).sum()),
        "positive_ratio": (df["strat_ret_pct"] > 0).mean(),
        "median_month": df["strat_ret_pct"].median(),
        "p10_month": df["strat_ret_pct"].quantile(0.10),
        "p90_month": df["strat_ret_pct"].quantile(0.90),
        "worst_month": df["strat_ret_pct"].min(),
        "best_month": df["strat_ret_pct"].max(),
    }

    return {
        "table": heatmap_table,
        "stats": stats,
    }




