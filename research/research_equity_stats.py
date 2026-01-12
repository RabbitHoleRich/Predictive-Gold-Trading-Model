import numpy as np
import pandas as pd

def compute_equity_curve_stats(backtest_df: pd.DataFrame) -> dict:
    """
    Equity-curve–derived statistics aligned EXACTLY to the equity chart.
    """
    stats = {}

    df = backtest_df.copy().sort_values("date")

    # -------------------------------------------------
    # Build equity curves (MATCH chart logic)
    # -------------------------------------------------
    strat_eq = (1 + df["strategy_ret"]).cumprod()
    bh_eq = (1 + df["bh_ret"]).cumprod()

    strat_eq /= strat_eq.iloc[0]
    bh_eq /= bh_eq.iloc[0]

    # -------------------------------------------------
    # Path metrics
    # -------------------------------------------------
    rolling_max = strat_eq.cummax()
    at_highs = strat_eq == rolling_max
    underwater = strat_eq < rolling_max

    stats["n_new_highs"] = int(at_highs.sum())
    stats["pct_time_at_highs"] = float(at_highs.mean())

    stats["max_stagnation_days"] = int(
        underwater.groupby(underwater.ne(underwater.shift()).cumsum())
        .sum()
        .max()
    )

    stats["relative_outperformance"] = float((strat_eq / bh_eq).iloc[-1])

    # -------------------------------------------------
    # Growth acceleration metrics
    # -------------------------------------------------
    if len(strat_eq) < 252 * 3:
        stats["growth_acceleration_ratio"] = None
        stats["rolling_cagr_pickup_pp"] = None
        return stats

    log_eq = np.log(strat_eq.clip(lower=1e-12))
    x = np.arange(len(log_eq))

    n = len(log_eq)
    a = n // 3
    b = 2 * n // 3

    def slope(y, x):
        x_ = x - x.mean()
        return float((x_ @ (y - y.mean())) / (x_ @ x_))

    slope_early = slope(log_eq.iloc[:a].values, x[:a])
    slope_late = slope(log_eq.iloc[b:].values, x[b:])

    stats["growth_acceleration_ratio"] = (
        float(slope_late / slope_early) if slope_early > 0 else None
    )

    window = 252 * 3
    roll_cagr = (strat_eq / strat_eq.shift(window)) ** (252 / window) - 1
    roll_cagr = roll_cagr.dropna()

    if len(roll_cagr) > 10:
        m = len(roll_cagr)
        med_early = roll_cagr.iloc[: m // 3].median()
        med_late = roll_cagr.iloc[2 * m // 3 :].median()
        stats["rolling_cagr_pickup_pp"] = float((med_late - med_early) * 100)
    else:
        stats["rolling_cagr_pickup_pp"] = None

    return stats


    # -------------------------------------------------
    # Growth acceleration metrics
    # -------------------------------------------------
    if len(eq) < 252 * 3:
        stats["growth_acceleration_ratio"] = None
        stats["rolling_cagr_pickup_pp"] = None
        return stats

    # --- log-equity slope acceleration ---
    log_eq = np.log(eq.clip(lower=1e-12))
    x = np.arange(len(log_eq))

    n = len(log_eq)
    a = n // 3
    b = 2 * n // 3

    def slope(y, x):
        x_ = x - x.mean()
        return float((x_ @ (y - y.mean())) / (x_ @ x_))

    slope_early = slope(log_eq.iloc[:a].values, x[:a])
    slope_late = slope(log_eq.iloc[b:].values, x[b:])

    stats["growth_acceleration_ratio"] = (
        float(slope_late / slope_early) if slope_early > 0 else None
    )

    # --- rolling CAGR pickup (3-year window) ---
    window = 252 * 3
    roll_cagr = (eq / eq.shift(window)) ** (252 / window) - 1
    roll_cagr = roll_cagr.dropna()

    if len(roll_cagr) > 10:
        m = len(roll_cagr)
        med_early = roll_cagr.iloc[: m // 3].median()
        med_late = roll_cagr.iloc[2 * m // 3 :].median()
        stats["rolling_cagr_pickup_pp"] = float((med_late - med_early) * 100)
    else:
        stats["rolling_cagr_pickup_pp"] = None

    return stats
