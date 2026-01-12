"""
research_retvol_stats.py

Chart-aligned statistics for return vs volatility by regime.
"""

import numpy as np
import pandas as pd


def _ann_return_from_daily(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def _ann_vol_from_daily(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.std() * np.sqrt(252))


def compute_return_vs_vol_profile(oos: pd.DataFrame) -> dict:
    """
    Computes regime-level return/vol/sharpe comparisons for Strategy vs Buy&Hold,
    aligned to the return-vs-vol chart.
    """
    required = {"strategy_ret", "bh_ret", "regime_code", "position"}
    missing = required - set(oos.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = oos.copy()
    df = df.dropna(subset=["regime_code"])
    df["regime_code"] = df["regime_code"].astype(int)

    out_rows = []
    for reg, g in df.groupby("regime_code"):
        strat_r = g["strategy_ret"]
        bh_r = g["bh_ret"]

        ann_ret = _ann_return_from_daily(strat_r)
        ann_vol = _ann_vol_from_daily(strat_r)
        ann_sharpe = float(ann_ret / ann_vol) if np.isfinite(ann_ret) and np.isfinite(ann_vol) and ann_vol > 0 else np.nan

        bh_ann_ret = _ann_return_from_daily(bh_r)
        bh_ann_vol = _ann_vol_from_daily(bh_r)
        bh_ann_sharpe = float(bh_ann_ret / bh_ann_vol) if np.isfinite(bh_ann_ret) and np.isfinite(bh_ann_vol) and bh_ann_vol > 0 else np.nan

        # Time in market within regime (how “active” the strategy is there)
        tim = float((g["position"] != 0).mean())

        out_rows.append(
            {
                "regime_code": reg,
                "n_days": int(len(g)),
                "time_in_market": tim,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "ann_sharpe": ann_sharpe,
                "bh_ann_return": bh_ann_ret,
                "bh_ann_vol": bh_ann_vol,
                "bh_ann_sharpe": bh_ann_sharpe,
                "excess_ann_return": float(ann_ret - bh_ann_ret) if np.isfinite(ann_ret) and np.isfinite(bh_ann_ret) else np.nan,
                "vol_ratio": float(ann_vol / bh_ann_vol) if np.isfinite(ann_vol) and np.isfinite(bh_ann_vol) and bh_ann_vol > 0 else np.nan,
            }
        )

    prof = pd.DataFrame(out_rows).sort_values("regime_code").reset_index(drop=True)

    # High-level summary used for narrative
    wins = (prof["ann_sharpe"] > prof["bh_ann_sharpe"]).sum()
    out = {
        "table": prof,
        "n_regimes": int(prof.shape[0]),
        "n_regimes_sharpe_beats_bh": int(wins),
    }

    # “Best” regime for strategy Sharpe (if available)
    if prof["ann_sharpe"].notna().any():
        best = prof.loc[prof["ann_sharpe"].idxmax()]
        out["best_regime"] = int(best["regime_code"])
        out["best_regime_sharpe"] = float(best["ann_sharpe"])
        out["best_regime_ann_return"] = float(best["ann_return"])
        out["best_regime_ann_vol"] = float(best["ann_vol"])

    return out
