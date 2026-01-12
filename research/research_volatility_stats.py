import pandas as pd
import numpy as np


def compute_rolling_volatility_profile(df: pd.DataFrame, window: int = 126) -> dict:
    """
    Rolling vol stats for commentary + diagnostics.
    - window must match the chart window
    - high-vol regime is defined by Buy & Hold rolling vol (top decile)
    """
    stats = {}

    strat_vol = df["strategy_ret"].rolling(window).std() * np.sqrt(252)
    bh_vol    = df["bh_ret"].rolling(window).std() * np.sqrt(252)

    # Core strategy-vol stats (as before)
    stats["mean_vol"] = float(strat_vol.mean())
    stats["cv_vol"]   = float(strat_vol.std() / strat_vol.mean())
    stats["p95_vol"]  = float(strat_vol.quantile(0.95))

    # Define stress regime off BH vol (not strategy vol)
    bh_q90 = bh_vol.quantile(0.90)
    high_bh = bh_vol >= bh_q90

    stats["pct_high_vol"] = float(high_bh.mean())

    # Average vol ratio (as before, but made safe)
    vol_ratio = (strat_vol / bh_vol).replace([np.inf, -np.inf], np.nan)
    stats["mean_vol_ratio"] = float(vol_ratio.mean())

    # NEW: “goes into reverse” evidence
    stats["bh_strat_vol_corr"] = float(strat_vol.corr(bh_vol))
    stats["mean_vol_ratio_high_bh"] = float(vol_ratio[high_bh].mean())
    stats["mean_vol_ratio_normal_bh"] = float(vol_ratio[~high_bh].mean())

    return stats

