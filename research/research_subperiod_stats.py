import pandas as pd

def compute_subperiod_profile(subperiod_df: pd.DataFrame) -> dict:
    stats = {}

    best_era = subperiod_df["CAGR"].idxmax()
    worst_era = subperiod_df["CAGR"].idxmin()

    best_cagr = subperiod_df.loc[best_era, "CAGR"]
    worst_cagr = subperiod_df.loc[worst_era, "CAGR"]

    stats["best_era"] = best_era
    stats["worst_era"] = worst_era

    # NEW: expose actual return levels (decimal CAGR)
    stats["best_cagr"] = float(best_cagr)
    stats["worst_cagr"] = float(worst_cagr)

    stats["cagr_range"] = best_cagr - worst_cagr

    stats["positive_eras"] = (subperiod_df["CAGR"] > 0).sum()
    stats["total_eras"] = len(subperiod_df)

    stats["table"] = subperiod_df.copy()

    return stats



