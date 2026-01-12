import pandas as pd
import numpy as np


def compute_annual_max_loss_profile(oos: pd.DataFrame) -> dict:
    required = {"date", "strategy_equity", "bh_equity"}
    missing = required - set(oos.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = oos.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    def max_drawdown(equity):
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        return dd.min()

    annual = (
        df.groupby("year")
        .apply(lambda g: pd.Series({
            "strat_max_dd": max_drawdown(g["strategy_equity"]),
            "bh_max_dd": max_drawdown(g["bh_equity"]),
        }))
        .dropna()
    )

    stats = {
        "worst_year_dd": annual["strat_max_dd"].min(),
        "median_year_dd": annual["strat_max_dd"].median(),
        "p90_year_dd": annual["strat_max_dd"].quantile(0.90),
        "bh_worst_year_dd": annual["bh_max_dd"].min(),
        "n_years": int(len(annual)),
        "years_dd_gt_20pct": int((annual["strat_max_dd"] <= -0.20).sum()),
        "table": annual,
    }

    return stats

