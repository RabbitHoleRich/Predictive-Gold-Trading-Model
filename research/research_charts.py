"""
research_charts.py

Charting layer for the research engine.

All functions are PURE and return a Matplotlib Figure.
They do not save to disk or call plt.show().

Expected columns in backtest_df:
    - date
    - strategy_ret
    - bh_ret
    - strategy_equity
    - bh_equity
    - position
    - (optional) regime_code

Expected columns in walkforward_df:
    - some date/month column (month / test_month / period)
    - some strategy return column (final_strategy_return / strat_ret_pct)
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure

from research_stats import extract_trades

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def extract_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract drawdown depth and duration from an equity curve.
    Returns a DataFrame with columns:
        - depth (negative, as fraction)
        - duration (in days)
    """

    equity = df["strategy_equity"]

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    drawdowns = []
    in_dd = False
    start_idx = None
    min_dd = 0.0

    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            if not in_dd:
                in_dd = True
                start_idx = i
                min_dd = drawdown.iloc[i]
            else:
                min_dd = min(min_dd, drawdown.iloc[i])
        else:
            if in_dd:
                duration = i - start_idx
                drawdowns.append({
                    "depth": min_dd,
                    "duration": duration
                })
                in_dd = False

    # Handle drawdown still open at end
    if in_dd:
        duration = len(drawdown) - start_idx
        drawdowns.append({
            "depth": min_dd,
            "duration": duration
        })

    return pd.DataFrame(drawdowns)



# ---------------------------------------------------------------------
# Confidence ellipse for plots (module-level, reusable)
# ---------------------------------------------------------------------
def confidence_ellipse(
    x,
    y,
    ax: plt.Axes,
    n_std: float = 2.0,
    edgecolor: str = "black",
    **kwargs: Any,
) -> None:
    """
    Draw a confidence ellipse for 2D data.
    - n_std=2.0 corresponds to ≈95% confidence region.
    - x and y must be array-like and of equal length.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size < 3 or y.size < 3:
        return  # Not enough points for covariance matrix

    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    # Angle of ellipse = angle of first eigenvector
    angle = np.rad2deg(np.arccos(v[0, 0]))

    ellipse = patches.Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=n_std * lambda_[0],
        height=n_std * lambda_[1],
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linewidth=1.5,
        alpha=0.8,
        **kwargs,
    )
    ax.add_patch(ellipse)


# ------------------------------------------------------------------
# Equity curve
# ------------------------------------------------------------------
def plot_equity_curves(backtest_df: pd.DataFrame) -> Figure:
    """
    Cumulative equity curves (Strategy vs Buy & Hold).

    Builds equity curves from daily returns and normalises both to start at 1.0.
    """

    df = backtest_df.copy()
    df = df.sort_values("date")

    df["strat_eq"] = (1 + df["strategy_ret"]).cumprod()
    df["bh_eq"] = (1 + df["bh_ret"]).cumprod()

    df["strat_eq"] /= df["strat_eq"].iloc[0]
    df["bh_eq"] /= df["bh_eq"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["date"], df["strat_eq"], label="Strategy", linewidth=2)
    ax.plot(df["date"], df["bh_eq"], label="Buy & Hold", linewidth=2, linestyle="--")

    ax.set_title("Cumulative Equity Curve: Strategy vs Buy & Hold", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (multiple of initial)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig

# ------------------------------------------------------------------
# Trade profile (holding period vs return) - Complete
# ------------------------------------------------------------------
def plot_trade_profile(backtest_df: pd.DataFrame) -> Optional[Figure]:
    """
    Trade-level diagnostics:
      - Joint distribution of holding period vs trade return
    """

    trades = extract_trades(backtest_df)

    if trades.empty:
        print("[INFO] No trades extracted; skipping trade profile plot.")
        return None

    trades = trades.copy()
    trades["holding_days"] = trades["holding_days"].astype(int)
    trades["trade_return_pct"] = trades["trade_return"].astype(float) * 100.0

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Scatter: holding period vs return
    ax.scatter(
        trades["holding_days"],
        trades["trade_return_pct"],
        alpha=0.6,
        s=25,
    )

    # Zero-return reference line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Optional: rolling median payoff vs holding period
    # (kept subtle, avoids regression signalling)
    trades_sorted = trades.sort_values("holding_days")
    rolling = (
        trades_sorted
        .groupby("holding_days")["trade_return_pct"]
        .median()
        .rolling(window=5, min_periods=3)
        .mean()
    )
    ax.plot(
        rolling.index,
        rolling.values,
        color="black",
        linewidth=1.5,
        alpha=0.8,
        label="Rolling median return",
    )

    ax.set_title("Trade Payoff vs Holding Period")
    ax.set_xlabel("Holding period (days)")
    ax.set_ylabel("Trade return (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig



# ------------------------------------------------------------------
# Underwater drawdown curve
# ------------------------------------------------------------------
def plot_drawdowns(backtest_df: pd.DataFrame) -> Figure:
    """
    Underwater drawdown curves for Strategy and Buy & Hold.
    Drawdowns shown in percent (<= 0%).
    """

    df = backtest_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    strat_eq = (1 + df["strategy_ret"]).cumprod()
    bh_eq = (1 + df["bh_ret"]).cumprod()

    def dd_series(eq: pd.Series) -> pd.Series:
        running_max = eq.cummax()
        return eq / running_max - 1.0

    df["strat_dd"] = dd_series(strat_eq) * 100.0
    df["bh_dd"] = dd_series(bh_eq) * 100.0

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["strat_dd"], label="Strategy DD", linewidth=1.5)
    ax.plot(df["date"], df["bh_dd"], label="Buy & Hold DD", linestyle="--", linewidth=1.5)

    ax.set_title("Underwater Drawdown Curve: Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    bottom = min(df["strat_dd"].min(), df["bh_dd"].min())
    ax.set_ylim(bottom * 1.05, 0.0)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Drawdown duration vs depth
# ------------------------------------------------------------------
def plot_drawdown_duration_vs_depth(backtest_df: pd.DataFrame) -> Optional[Figure]:
    """
    Drawdown duration vs depth for Strategy and Buy & Hold.

    - Each point = one drawdown episode
    - x-axis: duration in days
    - y-axis: max depth (%)
    - Bubble size = duration
    - Colour encodes time (year)
    """

    df = backtest_df.copy().sort_values("date").reset_index(drop=True)

    strat_eq = (1 + df["strategy_ret"]).cumprod().values
    bh_eq = (1 + df["bh_ret"]).cumprod().values
    dates = df["date"].values

    def extract_dd(eq: np.ndarray, dates_arr) -> pd.DataFrame:
        episodes = []
        peak = eq[0]
        dd_start = None
        dd_end = None
        max_dd_val = 0.0

        for i, val in enumerate(eq):
            if val >= peak:
                if dd_start is not None:
                    episodes.append((dd_start, dd_end, max_dd_val))
                peak = val
                dd_start = None
                max_dd_val = 0.0
            else:
                dd_val = val / peak - 1.0
                if dd_start is None:
                    dd_start = i
                dd_end = i
                max_dd_val = min(max_dd_val, dd_val)

        if dd_start is not None:
            episodes.append((dd_start, dd_end, max_dd_val))

        rows = []
        for s, e, dd in episodes:
            duration = e - s
            end_date = pd.to_datetime(dates_arr[e])
            rows.append(
                {
                    "duration": duration,
                    "depth": dd,
                    "end_date": end_date,
                }
            )
        return pd.DataFrame(rows)

    strat_dd = extract_dd(strat_eq, dates)
    bh_dd = extract_dd(bh_eq, dates)

    if strat_dd.empty or bh_dd.empty:
        print("[INFO] No drawdown episodes to plot.")
        return None

    for ddf in (strat_dd, bh_dd):
        ddf["year_float"] = ddf["end_date"].dt.year + ddf["end_date"].dt.dayofyear / 365.25

    years_all = pd.concat([strat_dd["year_float"], bh_dd["year_float"]])
    norm = plt.Normalize(vmin=years_all.min(), vmax=years_all.max())
    cmap = plt.cm.viridis

    def scale_size(dur_series: pd.Series) -> np.ndarray:
        return 20.0 + 4.0 * np.sqrt(dur_series.values)

    strat_colors = cmap(norm(strat_dd["year_float"].values))
    bh_colors = cmap(norm(bh_dd["year_float"].values))

    strat_depth_pct = strat_dd["depth"].values * 100.0
    bh_depth_pct = bh_dd["depth"].values * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Strategy panel
    ax = axes[0]
    ax.scatter(
        strat_dd["duration"],
        strat_depth_pct,
        s=scale_size(strat_dd["duration"]),
        c=strat_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Strategy Drawdowns")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Max depth (%)")
    ax.grid(alpha=0.3)

    s_deep_idx = strat_depth_pct.argmin()
    s_long_idx = strat_dd["duration"].values.argmax()

    s_deep = strat_dd.iloc[s_deep_idx]
    ax.annotate(
        "Deepest",
        xy=(s_deep["duration"], strat_depth_pct[s_deep_idx]),
        xytext=(15, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=8,
    )

    s_long = strat_dd.iloc[s_long_idx]
    ax.annotate(
        "Longest",
        xy=(s_long["duration"], strat_depth_pct[s_long_idx]),
        xytext=(15, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=8,
    )

    # Buy & Hold panel
    ax = axes[1]
    ax.scatter(
        bh_dd["duration"],
        bh_depth_pct,
        s=scale_size(bh_dd["duration"]),
        c=bh_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Buy & Hold Drawdowns")
    ax.set_xlabel("Duration (days)")
    ax.grid(alpha=0.3)

    b_deep_idx = bh_depth_pct.argmin()
    b_long_idx = bh_dd["duration"].values.argmax()

    b_deep = bh_dd.iloc[b_deep_idx]
    ax.annotate(
        "Deepest",
        xy=(b_deep["duration"], bh_depth_pct[b_deep_idx]),
        xytext=(15, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=8,
    )

    b_long = bh_dd.iloc[b_long_idx]
    ax.annotate(
        "Longest",
        xy=(b_long["duration"], bh_depth_pct[b_long_idx]),
        xytext=(15, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=8,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("End of drawdown (year)")

    fig.suptitle("Drawdown Duration vs Depth (time-encoded, bubble-scaled)")
    return fig


# ------------------------------------------------------------------
# Annual maximum drawdowns
# ------------------------------------------------------------------
def plot_annual_max_drawdowns(backtest_df: pd.DataFrame) -> Figure:
    """
    Annual maximum drawdown comparison (Strategy vs Buy & Hold).
    """

    df = backtest_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    strat_eq = (1 + df["strategy_ret"]).cumprod()
    bh_eq = (1 + df["bh_ret"]).cumprod()

    def dd_series(eq: pd.Series) -> pd.Series:
        running_max = eq.cummax()
        return eq / running_max - 1.0

    df["strat_dd"] = dd_series(strat_eq) * 100
    df["bh_dd"] = dd_series(bh_eq) * 100
    df["year"] = df["date"].dt.year

    annual_strat_dd = df.groupby("year")["strat_dd"].min()
    annual_bh_dd = df.groupby("year")["bh_dd"].min()

    years = annual_strat_dd.index.values

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.4

    ax.bar(years - 0.2, annual_strat_dd.values, width=width, label="Strategy")
    ax.bar(years + 0.2, annual_bh_dd.values, width=width, label="Buy & Hold")

    ax.set_title("Annual Maximum Drawdown Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("Max Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.xticks(years, rotation=45)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Annual maximum gains
# ------------------------------------------------------------------
def plot_annual_max_gains(backtest_df: pd.DataFrame) -> Figure:
    """
    Annual maximum gains (best run-ups) for Strategy vs Buy & Hold.
    Run-up is measured within each calendar year.
    """

    df = backtest_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    strat_eq = (1 + df["strategy_ret"]).cumprod()
    bh_eq = (1 + df["bh_ret"]).cumprod()

    df["strat_eq"] = strat_eq
    df["bh_eq"] = bh_eq
    df["year"] = df["date"].dt.year

    def annual_max_gain(series: pd.Series, years: np.ndarray) -> Dict[int, float]:
        max_gains: Dict[int, float] = {}
        for yr in years:
            s = series[series.index.year == yr]
            if len(s) == 0:
                max_gains[yr] = 0.0
                continue
            running_min = s.cummin()
            runup = (s / running_min) - 1.0
            max_gains[yr] = float(runup.max() * 100.0)
        return max_gains

    years = df["year"].unique()
    years_sorted = np.sort(years)

    strat_gain_by_year = annual_max_gain(df.set_index("date")["strat_eq"], years_sorted)
    bh_gain_by_year = annual_max_gain(df.set_index("date")["bh_eq"], years_sorted)

    strat_vals = [strat_gain_by_year[y] for y in years_sorted]
    bh_vals = [bh_gain_by_year[y] for y in years_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.4

    ax.bar([y - 0.2 for y in years_sorted], strat_vals, width=width, label="Strategy")
    ax.bar([y + 0.2 for y in years_sorted], bh_vals, width=width, label="Buy & Hold")

    ax.set_title("Annual Maximum Gains (Best Intra-Year Run-Ups)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Maximum Gain (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.xticks(years_sorted, rotation=45)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Rolling volatility
# ------------------------------------------------------------------
def plot_rolling_volatility(
    backtest_df: pd.DataFrame,
    trading_days_per_year: int = 252,
    window: int = 126,
) -> Figure:
    """
    Rolling annualised volatility with crisis period shading.
    Strategy vs Buy & Hold, ~6-month window by default.
    """

    df = backtest_df.sort_values("date").reset_index(drop=True)

    ann = np.sqrt(trading_days_per_year)
    df["strat_vol"] = df["strategy_ret"].rolling(window).std(ddof=1) * ann
    df["bh_vol"] = df["bh_ret"].rolling(window).std(ddof=1) * ann

    fig, ax = plt.subplots(figsize=(12, 5))

    crisis_periods = [
        ("Dot-com aftermath", "2001-01-01", "2003-12-31"),
        ("GFC", "2007-07-01", "2009-06-30"),
        ("Euro Debt Crisis", "2011-06-01", "2012-12-31"),
        ("Taper/China FX", "2013-05-01", "2015-12-31"),
        ("COVID Crash", "2020-02-01", "2020-05-31"),
        ("Inflation Shock", "2021-11-01", "2022-12-31"),
    ]

    for _, start, end in crisis_periods:
        ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color="grey",
            alpha=0.20,
        )

    ax.plot(df["date"], df["strat_vol"], label="Strategy", linewidth=1.6)
    ax.plot(df["date"], df["bh_vol"], label="Buy & Hold", linestyle="--", linewidth=1.6)

    ax.set_title(f"Rolling Annualised Volatility (window ≈ {window} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Rolling Sharpe & CAGR (uses rolling_stats DataFrame)
# ------------------------------------------------------------------
def plot_rolling_sharpe_and_cagr(rolling_stats_df: pd.DataFrame) -> Optional[Figure]:
    """
    Plot smoothed rolling Sharpe and rolling CAGR approximation.
    Expects columns:
        - date
        - strat_rolling_sharpe_smooth
        - bh_rolling_sharpe_smooth
        - strat_rolling_cagr_smooth
        - bh_rolling_cagr_smooth
    """

    rs = rolling_stats_df.dropna().copy()
    if rs.empty:
        print("[INFO] Rolling stats empty; skipping rolling Sharpe/CAGR plot.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Rolling Sharpe
    ax = axes[0]
    ax.plot(
        rs["date"],
        rs["strat_rolling_sharpe_smooth"],
        label="Strategy (smoothed)",
        linewidth=1.6,
    )
    ax.plot(
        rs["date"],
        rs["bh_rolling_sharpe_smooth"],
        label="Buy & Hold (smoothed)",
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_title("Rolling Sharpe (smoothed, window ~ 6 months)")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rolling CAGR
    ax = axes[1]
    ax.plot(
        rs["date"],
        rs["strat_rolling_cagr_smooth"],
        label="Strategy (smoothed)",
        linewidth=1.6,
    )
    ax.plot(
        rs["date"],
        rs["bh_rolling_cagr_smooth"],
        label="Buy & Hold (smoothed)",
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_title("Rolling CAGR Approximation (smoothed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("CAGR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Return vs vol by regime
# ------------------------------------------------------------------
def plot_return_vs_vol_by_regime(backtest_df: pd.DataFrame) -> Optional[Figure]:
    """
    Monthly Return vs Volatility scatter plots separated by regime:
    Bull, Neutral, Bear.

    Uses 'regime_code' if present in backtest_df, otherwise derives
    regimes from monthly BH returns.
    """

    df = backtest_df.copy()
    df = df.sort_values("date").set_index("date")

    strat_monthly = (1 + df["strategy_ret"]).resample("M").prod() - 1
    bh_monthly = (1 + df["bh_ret"]).resample("M").prod() - 1

    strat_vol = df["strategy_ret"].resample("M").std()
    bh_vol = df["bh_ret"].resample("M").std()

    if "regime_code" in df.columns:
        month_regime = df["regime_code"].resample("M").last()
        regime_map = {0: "Bull", 1: "Neutral", 2: "Bear"}
        regime_labels = month_regime.map(regime_map).fillna("Neutral")
    else:
        gold_monthly = bh_monthly
        regime_labels = pd.cut(
            gold_monthly,
            bins=[-999, -0.01, 0.01, 999],
            labels=["Bear", "Neutral", "Bull"],
        )

    out = pd.DataFrame(
        {
            "strat_ret": strat_monthly,
            "strat_vol": strat_vol,
            "bh_ret": bh_monthly,
            "bh_vol": bh_vol,
            "regime": regime_labels,
        }
    ).dropna()

    if out.empty:
        print("[INFO] No monthly data for regime scatter; skipping.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    regimes = ["Bull", "Neutral", "Bear"]

    for ax, reg in zip(axes, regimes):
        sub = out[out["regime"] == reg]

        ax.scatter(
            sub["strat_vol"],
            sub["strat_ret"],
            label="Strategy",
            alpha=0.7,
            s=40,
            color="tab:blue",
            edgecolor="white",
        )
        ax.scatter(
            sub["bh_vol"],
            sub["bh_ret"],
            label="Buy & Hold",
            alpha=0.5,
            s=40,
            color="tab:orange",
            edgecolor="white",
        )

        confidence_ellipse(
            sub["strat_vol"], sub["strat_ret"], ax, n_std=2.0, edgecolor="tab:blue"
        )
        confidence_ellipse(
            sub["bh_vol"], sub["bh_ret"], ax, n_std=2.0, edgecolor="tab:orange"
        )

        ax.set_title(f"{reg} Regime")
        ax.set_xlabel("Monthly Volatility")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Monthly Return")
    axes[1].legend()

    fig.suptitle("Monthly Return vs Volatility by Regime", fontsize=14)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Walkforward heatmap
# ------------------------------------------------------------------

def plot_walkforward_heatmap(walkforward_df: Optional[pd.DataFrame]) -> Optional[Figure]:
    """
    Heatmap of monthly out-of-sample (walk-forward) returns.

    Supports two input types:
    1) Pre-aggregated heatmap table (index=year, columns=1–12, values=monthly return %)
    2) Raw walk-forward summary dataframe (legacy behaviour)
    """

    if walkforward_df is None:
        print("[INFO] No walk-forward summary loaded — skipping heatmap.")
        return None

    df = walkforward_df.copy()

    # ==========================================================
    # NEW PATH: already-aggregated heatmap table
    # ==========================================================
    
    if all(isinstance(c, (int, np.integer)) and 1 <= c <= 12 for c in df.columns):
        heat = df.copy()

        # ensure columns are 1..12
        for m in range(1, 13):
            if m not in heat.columns:
                heat[m] = np.nan

        heat = heat.sort_index().sort_index(axis=1)

    # ==========================================================
    # LEGACY PATH: raw walk-forward summary
    # ==========================================================
    else:
        # Detect return column
        if "final_strategy_return" in df.columns:
            ret = df["final_strategy_return"].astype(float)
        elif "strat_ret_pct" in df.columns:
            ret = df["strat_ret_pct"].astype(float) / 100.0
        else:
            print("[INFO] No recognised strategy return column found in walkforward_df.")
            return None

        # Detect date/month column
        if "month" in df.columns:
            months = pd.to_datetime(df["month"])
        elif "test_month" in df.columns:
            months = pd.to_datetime(df["test_month"])
        elif "period" in df.columns:
            months = pd.to_datetime(df["period"])
        else:
            print("[INFO] No date column for walk-forward windows.")
            return None

        df_hm = pd.DataFrame(
            {
                "date": months,
                "ret": ret.values * 100,
            }
        )

        df_hm["year"] = df_hm["date"].dt.year
        df_hm["month"] = df_hm["date"].dt.month

        heat = df_hm.pivot(index="year", columns="month", values="ret")

        for m in range(1, 13):
            if m not in heat.columns:
                heat[m] = np.nan

        heat = heat.sort_index().sort_index(axis=1)

    # ==========================================================
    # Plotting (shared)
    # ==========================================================
    r = heat.values.flatten()
    r = r[~np.isnan(r)]

    if len(r) == 0:
        print("[WARN] No returns to plot in heatmap.")
        return None

    abs_q = np.quantile(np.abs(r), 0.99)
    scale = abs_q if abs_q > 0 else 0.1
    scale = min(scale, 5.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.get_cmap("RdYlGn")

    im = ax.imshow(
        heat,
        cmap=cmap,
        aspect="auto",
        vmin=-scale,
        vmax=scale,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Monthly Return (%)")

    ax.set_title("Walk-forward Monthly Out-of-Sample Performance (Heatmap)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)

    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)

    return fig
 

# ------------------------------------------------------------------
# Subperiod performance by era
# ------------------------------------------------------------------
def plot_subperiod_performance(backtest_df: pd.DataFrame) -> Optional[Figure]:
    """
    Sub-period CAGRs for Strategy vs Buy & Hold across major eras.
    """

    df = backtest_df.sort_values("date").reset_index(drop=True)
    if df.empty:
        print("[INFO] Empty backtest_df; skipping sub-period performance plot.")
        return None

    end_all = df["date"].iloc[-1]

    eras = [
        ("Gold bull market (2001–2007)", pd.Timestamp("2001-08-01"), pd.Timestamp("2007-06-30")),
        ("Financial crisis & recovery (2008–2013)", pd.Timestamp("2007-07-01"), pd.Timestamp("2012-12-31")),
        ("Monetary easing era (2014-2019)", pd.Timestamp("2013-01-01"), pd.Timestamp("2018-12-31")),
        ("COVID pandemic & inflation shock(2020- )", pd.Timestamp("2019-01-01"), end_all),
    ]

    labels = []
    strat_cagrs = []
    bh_cagrs = []

    for label, start, end in eras:
        sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if sub.empty:
            continue

        def era_cagr(rets: pd.Series) -> float:
            rets = rets.dropna()
            if len(rets) < 2:
                return np.nan
            total_ret = (1.0 + rets).prod() - 1.0
            years = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days / 365.25
            if years <= 0:
                return np.nan
            return (1.0 + total_ret) ** (1.0 / years) - 1.0

        labels.append(label)
        strat_cagrs.append(era_cagr(sub["strategy_ret"]))
        bh_cagrs.append(era_cagr(sub["bh_ret"]))

    if not labels:
        print("[INFO] No data in defined eras; skipping sub-period plot.")
        return None

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, np.array(strat_cagrs) * 100.0, width=width, label="Strategy")
    ax.bar(x + width / 2, np.array(bh_cagrs) * 100.0, width=width, label="Buy & Hold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Sub-period Performance by Era")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig






    

   







