import pandas as pd


def compute_annual_max_gain_profile(oos: pd.DataFrame) -> dict:
    required = {"date", "strategy_equity", "bh_equity"}
    missing = required - set(oos.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = oos.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    annual = (
        df.groupby("year")
        .apply(lambda g: pd.Series({
            "strat_year_return": g["strategy_equity"].iloc[-1] / g["strategy_equity"].iloc[0] - 1,
            "bh_year_return": g["bh_equity"].iloc[-1] / g["bh_equity"].iloc[0] - 1,
        }))
        .dropna()
    )

    stats = {
        "best_year": annual["strat_year_return"].max(),
        "median_year": annual["strat_year_return"].median(),
        "p90_year": annual["strat_year_return"].quantile(0.90),
        "bh_best_year": annual["bh_year_return"].max(),
        "n_years": int(len(annual)),
        "years_gt_50pct": int((annual["strat_year_return"] >= 0.50).sum()),
        "table": annual,
    }

    return stats

