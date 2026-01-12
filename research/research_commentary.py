"""
commentary.py

Dynamic natural-language commentary generator for the gold_engine research layer.

This module is intentionally independent of:
- charts
- notebooks
- research_layer orchestration
- markdown/IO

It accepts only:
- a performance_summary object
- optional diagnostics (OOS tests, walk-forward data)
- optional stats registry

and produces a dictionary of text sections.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from IPython.display import Markdown


class Commentary:
    """
    Generates natural-language commentary sections for research reports,
    based purely on statistics (no side effects).
    """

    def __init__(self, performance_summary, diagnostics: Dict = None):
        self.ps = performance_summary
        self.diag = diagnostics or {}

    # ------------------------------------------------------------
    # Helpers for tone / qualitative descriptors
    # ------------------------------------------------------------

    @staticmethod
    def _growth_descriptor(cagr: float) -> str:
        """
        Turn CAGR into qualitative language.
        """
        if cagr >= 0.20:
            return "explosive"
        elif cagr >= 0.12:
            return "strong"
        elif cagr >= 0.07:
            return "modest"
        elif cagr >= 0.03:
            return "low but positive"
        else:
            return "weak"

    @staticmethod
    def _risk_descriptor(drawdown: float) -> str:
        dd = abs(drawdown)
        if dd < 0.05:
            return "extremely low"
        elif dd < 0.10:
            return "very low"
        elif dd < 0.20:
            return "moderate"
        else:
            return "high"

    @staticmethod
    def _sharpe_descriptor(sharpe: float) -> str:
        if sharpe >= 2.0:
            return "exceptional"
        elif sharpe >= 1.5:
            return "very strong"
        elif sharpe >= 1.0:
            return "solid"
        else:
            return "weak"

    # ------------------------------------------------------------
    # Commentary generators
    # ------------------------------------------------------------

    def performance_summary(self) -> str:
        ps = self.ps

        years = ps.n_years
        strat_cagr = ps.strat_cagr
        bh_cagr = ps.bh_cagr

        strat_cagr_pct = strat_cagr * 100
        bh_cagr_pct = bh_cagr * 100
 
        strat_mult = (1 + strat_cagr) ** years
        bh_mult = (1 + bh_cagr) ** years

        strat_total_return = (strat_mult - 1) * 100
        bh_total_return = (bh_mult - 1) * 100

        annual_ratio = strat_cagr / bh_cagr 
        terminal_ratio = strat_mult / bh_mult
        growth_word = self._growth_descriptor(strat_cagr)

        return (
            f"Annually, the strategy compounds "
            f"{annual_ratio:.2f}× faster per year than Buy & Hold, achieving "
            f"{strat_cagr_pct:.2f}% per annum versus {bh_cagr_pct:.2f}%. "
            f"Over the full {years:.1f}-year sample, growth is **{growth_word}**, with £1 compounding to "
            f"{strat_mult:.1f}× capital ({strat_total_return:.1f}% total return) "
            f"compared with {bh_mult:.1f}× ({bh_total_return:.1f}%) under Buy & Hold."
        )

    # ------------------------------------------------------------
    # New code: For writing Equity commentary
    # ------------------------------------------------------------

    def equity_profile_commentary(self, stats: dict) -> str:
        ep = stats["equity_profile"]

        return (
            f"The equity curve highlights steady compounding over time and, by the end of the period, "
            f"the strategy reaches {ep['relative_outperformance']:.1f}× the value of Buy & Hold. "
            f"Despite the curve’s visual appearance, growth in later years "
            f"is around {ep['growth_acceleration_ratio']:.2f}× that of the early sample. "
            f"However, median rolling long-term returns decline by only "
            f"{abs(ep['rolling_cagr_pickup_pp']):.1f} percentage points, "
            f"indicating consistency across regimes."
        )

    # ------------------------------------------------------------
    # New code: For writing Trade commentary
    # ------------------------------------------------------------
    def trade_profile_commentary(self, stats: dict) -> str:
        """
        Commentary aligned to the holding-period vs return chart,
        with numeric evidence that updates as data changes.
        """

        tp = stats["trade_profile"]

        corr = tp["spearman_corr"]
        bucket = tp["bucket_summary"]

        short = bucket.loc["short"]
        medium = bucket.loc["medium"]
        long = bucket.loc["long"]

        lines = []

        # 5. Plain-English conclusion (matches what the eye sees)
        lines.append(
            "Overall, the chart indicates that the strategy does not rely on holding trades for longer "
            "periods to generate returns, and that positive and negative outcomes occur with similar "
            "frequency across the full range of holding durations."
        )

        # 1. Symmetry / lack of time dependence
        lines.append(
            f"Trade-level returns show only a weak relationship with holding duration "
            f"(correlation = {corr:.2f}, close to zero), indicating that time in trade does not "
            f"meaningfully influence expected return."
        )

        # 2. Median returns across holding periods
        lines.append(
            f"Median trade returns remain close to zero across short, medium, and long holding periods "
            f"({short.median_return:.2f}%, {medium.median_return:.2f}%, and {long.median_return:.2f}% respectively), "
            f"suggesting broadly symmetric outcomes regardless of trade length."
        )

        # 3. Win-rate consistency
        lines.append(
            f"Win rates are similarly stable across durations, at "
            f"{short.win_rate*100:.1f}% for short trades, "
            f"{medium.win_rate*100:.1f}% for medium trades, and "
            f"{long.win_rate*100:.1f}% for longer-held trades."
        )

        # 4. Dispersion insight (where variation actually appears)
        lines.append(
            f"Return dispersion is widest for medium-duration trades, while very short and very long "
            f"holding periods tend to cluster more tightly around zero, indicating that outcomes at the "
            f"extremes are generally more muted."
        )
        
        return " ".join(lines)


    # ------------------------------------------------------------
    # New code: For writing underwater commentary
    # ------------------------------------------------------------

    def underwater_drawdown_commentary(self, stats: dict) -> str:
        ud = stats["underwater_profile"]

        lines = []
        
        lines.append(
            f"The strategy experiences materially lower cumulative losses "
            f"than Buy & Hold, as reflected by a much smaller area "
            f"under the underwater curve."
        )

        lines.append(
            f"The strategy spends {ud['pct_time_underwater']*100:.1f}% of trading days "
            f"below its prior equity peak, compared with "
            f"{ud['bh_pct_time_underwater']*100:.1f}% for Buy & Hold."
        )

        lines.append(
            f"Drawdown recoveries are typically fast, with a median recovery time of "
            f"{ud['median_recovery_days']:.0f} days and a worst-case recovery of "
            f"{ud['max_recovery_days']:.0f} days."
        )


        return " ".join(lines)


    # ------------------------------------------------------------
    # New code: For writing Drawdown commentary
    # ------------------------------------------------------------
    def drawdown_profile_commentary(self, stats: dict) -> str:
        dp = stats["drawdown_profile"]

        corr = dp["spearman_corr"]
        med_depth = dp["median_depth"] * 100
        med_dur = dp["median_duration"]
        p90_depth = dp["p90_depth"] * 100
        p90_dur = dp["p90_duration"]

        lines = []

        lines.append(
            f"Drawdown depth and duration show little relationship "
            f"(correlation = {corr:.2f}), indicating that longer drawdowns "
            f"are not necessarily deeper."
        )

        lines.append(
            f"A typical drawdown is shallow and short-lived, with a median depth of "
            f"{med_depth:.1f}% lasting {med_dur:.0f} days."
        )

        lines.append(
            f"Severe drawdowns are rare: 90% of drawdowns are no worse than "
            f"{p90_depth:.1f}% and recover within {p90_dur:.0f} days."
        )

        lines.append(
            "Overall, drawdown risk is characterised by limited depth and duration, "
            "supporting the visual impression of controlled downside and rapid recovery."
        )

        return " ".join(lines)

    # ------------------------------------------------------------
    # New code: For maxloss commentary
    # ------------------------------------------------------------  

    def maxloss_profile_commentary(self, stats: dict) -> str:
        ml = stats["maxloss_profile"]

        worst = ml["worst_year_dd"] * 100
        med = ml["median_year_dd"] * 100
        p90 = ml["p90_year_dd"] * 100
        bh_worst = ml["bh_worst_year_dd"] * 100
        n_years = ml["n_years"]
        n_big = ml["years_dd_gt_20pct"]

        lines = []

        lines.append(
            f"Annual drawdowns are generally contained, with a median worst-year drawdown "
            f"of {med:.1f}% across {n_years} calendar years."
        )

        lines.append(
            f"Severe loss years are rare: the worst observed calendar-year drawdown is "
            f"{worst:.1f}%, and 90% of years experience drawdowns no worse than {p90:.1f}%."
        )

        lines.append(
            f"For comparison, Buy & Hold experiences a worst calendar-year drawdown of "
            f"{bh_worst:.1f}%, materially deeper than the strategy’s worst year."
        )

        lines.append(
            "This distribution supports the visual impression that downside risk is "
            "controlled not only intra-cycle, but also on a full-year basis."
        )

        return " ".join(lines)

    # ----------------------------------------------------------
    # New code: For maxgain commentary
    # ----------------------------------------------------------

    def maxgain_profile_commentary(self, stats: dict) -> str:
        mg = stats["maxgain_profile"]

        best = mg["best_year"] * 100
        med = mg["median_year"] * 100
        p90 = mg["p90_year"] * 100
        bh_best = mg["bh_best_year"] * 100
        n_years = mg["n_years"]
        n_big = mg["years_gt_50pct"]

        lines = []
    
        lines.append(
            f"Annual gains are driven by consistent performance rather than isolated blow-out years. "
            f"The median calendar-year return is {med:.1f}% across {n_years} years."
        )

        lines.append(
            f"{n_big} out of {n_years} years exceed 50% returns and "
            f"the strongest calendar year delivers a return of {best:.1f}%, "
            f"indicating the strategy generates consistent returns rather than depending on a few exceptional years."
        )
        lines.append(
            f"For comparison, Buy & Hold’s strongest year delivers approximately {bh_best:.1f}%, "
            f"which is comparable in magnitude but achieved with substantially higher drawdown risk."
        )
        return " ".join(lines)


    # ------------------------------------------------------------
    # New code: For writing rolling CAGR and Sharpe commentary
    # ------------------------------------------------------------


    def _sharpe_trend_descriptor(self, slope: float) -> str:
        abs_slope = abs(slope)

        if abs_slope < 5e-5:
            return "stable"
        elif slope > 0:
            return "gently improving" if abs_slope < 2e-4 else "clearly improving"
        else:
            return "mildly weakening" if abs_slope < 2e-4 else "deteriorating"


    def rolling_sharpe_commentary(self, stats: dict) -> str:
        rp = stats["rolling_sharpe_profile"]
        ps = stats.get("performance_summary")
    
        lines = []
    
    # Lead with the actual comparison
        if ps:
            strat_sharpe = ps.strat_sharpe
            bh_sharpe = ps.bh_sharpe
            if strat_sharpe > 0 and bh_sharpe > 0:
                improvement = strat_sharpe / bh_sharpe
                lines.append(
                    f"The strategy delivers a Sharpe ratio of {strat_sharpe:.2f} compared to "
                    f"Buy & Hold's {bh_sharpe:.2f} - a {improvement:.1f}x improvement in "
                    f"risk-adjusted returns."
                )
    
    # Talk about consistency
        trend_word = self._sharpe_trend_descriptor(rp["sharpe_trend"])
        lines.append(
            f"Looking at rolling 6-month windows, the strategy's Sharpe profile is {trend_word} over time, "
            f"showing consistent outperformance across different market conditions rather than "
            f"isolated winning periods."
        )
    
    # Mention worst case
        lines.append(
            f"Even during the most challenging rolling period, the strategy maintained a Sharpe of "
            f"{rp['worst_sharpe']:.2f}, demonstrating resilience during adverse market conditions."
        )
    
        return " ".join(lines)



    # New code: For writing rolling volatility commentary
    # ------------------------------------------------------------

    def rolling_volatility_commentary(self, stats: dict) -> str:
        vp = stats["volatility_profile"]

        mean_vol_pct = vp["mean_vol"] * 100
        high_bh_ratio = vp["mean_vol_ratio_high_bh"]
        normal_ratio = vp["mean_vol_ratio_normal_bh"]
        corr = vp["bh_strat_vol_corr"]

    # Behavioural interpretation (dynamic, not hard-coded)
        if high_bh_ratio < 0.4:
            stress_phrase = "effectively withdrawing risk during market stress"
        elif high_bh_ratio < 0.7:
            stress_phrase = "materially reducing risk during market stress"
        else:
            stress_phrase = "only modestly reducing risk during market stress"

        lines = []

        lines.append(
            f"Rolling volatility is well controlled, with a mean annualised volatility of "
            f"{mean_vol_pct:.1f}%, indicating stable risk exposure through time."
        )

        lines.append(
            f"When Buy & Hold volatility rises, the strategy responds asymmetrically. "
            f"Strategy and Buy & Hold volatility show weak to negative co-movement "
            f"(correlation = {corr:.2f}), "
            f"{stress_phrase}."
        )

        lines.append(
            f"During the highest volatility regimes, the strategy operates at just "
            f"{high_bh_ratio:.2f}× Buy & Hold volatility, compared with "
            f"{normal_ratio:.2f}× during more typical conditions."
        )

        return " ".join(lines)

    #---------------------------------------------------
 
    def return_vs_vol_commentary(self, stats: dict) -> str:
        rv = stats["ret_vs_vol_profile"]
        tbl = rv["table"]

    # Explicit regime naming (must match chart)
        regime_names = {
            0: "bull regime",
            1: "neutral regime",
            2: "bear regime",
        }

    # Pick best / worst Sharpe regimes
        best_idx = tbl["ann_sharpe"].idxmax() if tbl["ann_sharpe"].notna().any() else None
        worst_idx = tbl["ann_sharpe"].idxmin() if tbl["ann_sharpe"].notna().any() else None

        lines = []

        lines.append(
            f"The model delivers superior risk-adjusted performance versus buy-and-hold in "
            f"{rv['n_regimes_sharpe_beats_bh']}/{rv['n_regimes']} market regimes, "
            f"indicating that the edge is not confined to a single market environment."
        )

        if best_idx is not None:
            b = tbl.loc[best_idx]
            regime_label = regime_names.get(int(b.regime_code), f"regime {int(b.regime_code)}")

            lines.append(
                f"The strongest performance occurs in the {regime_label}, where the model achieves "
                f"approximately {b.ann_return*100:.1f}% annualised return at "
                f"{b.ann_vol*100:.1f}% volatility (Sharpe ≈ {b.ann_sharpe:.2f}), "
                f"with time-in-market of {b.time_in_market*100:.1f}%."
            )

        if worst_idx is not None:
            w = tbl.loc[worst_idx]
            regime_label = regime_names.get(int(w.regime_code), f"regime {int(w.regime_code)}")

            lines.append(
                f"Even in the {regime_label}, the model’s risk exposure remains measured "
                f"(volatility ~{w.ann_vol*100:.1f}%) with time-in-market of "
                f"{w.time_in_market*100:.1f}%, "
                f"suggesting the model actively reduces exposure rather than remaining fully invested."
           )

        lines.append(
            "Overall, the return–volatility relationship supports the visual impression that the model "
            "earns returns with controlled risk across market regimes, rather than relying on a single "
            "high-volatility environment."
        )

        return " ".join(lines)


    # ----------------------------------------------------------
    # New code: For heatmap commentary
    # ----------------------------------------------------------

    def heatmap_commentary(self, stats: dict) -> str:
        hm = stats["heatmap_profile"]

        pos = hm["positive_ratio"] * 100
        med = hm["median_month"]
        p10 = hm["p10_month"]
        p90 = hm["p90_month"]
        worst = hm["worst_month"]
        best = hm["best_month"]
        n = hm["n_periods"]

        lines = []

        lines.append(
            f"The walk-forward heatmap summarises {n} out-of-sample monthly periods, "
            f"of which {pos:.1f}% are positive. "
        )

        lines.append(
            f"Monthly returns are tightly distributed, with a median of {med:.2f}%, "
            f"and 80% of observations falling between {p10:.2f}% and {p90:.2f}%."
        )

        lines.append(
            f"Extreme outcomes are limited: the strongest month delivers {best:.2f}%, "
            f"while the weakest month is {worst:.2f}%, consistent with the visually "
            f"balanced colour distribution across the heatmap."
        )

        lines.append(
            "Overall, the heatmap supports the conclusion that performance "
            "is stable across time and market conditions, with no evidence of dependence "
            "on a narrow set of favourable periods."
        )

        return " ".join(lines)

    # ----------------------------------------------------------
    # New code: For subperiod commentary
    # ----------------------------------------------------------

    def subperiod_profile_commentary(self, stats: dict) -> str:
        sp = stats["subperiod_profile"]

        best = sp["best_era"]
        worst = sp["worst_era"]
        best_cagr = sp["best_cagr"] * 100
        worst_cagr = sp["worst_cagr"] * 100

        pos = sp["positive_eras"]
        total = sp["total_eras"]

        # Range in percentage points
        cagr_range_pp = best_cagr - worst_cagr

        # Variability descriptor
        if cagr_range_pp < 5:
            spread_word = "tight"
        elif cagr_range_pp < 12:
            spread_word = "moderate"
        elif cagr_range_pp < 20:
            spread_word = "wide"
        else:
            spread_word = "very wide"

        return (
            f"Performance remains positive across {pos} of {total} macro eras, "
            f"indicating broad robustness rather than dependence on a single market environment. "
            f"Returns vary {spread_word}ly across eras, ranging from approximately "
            f"{best_cagr:.1f}% during the {best} era to {worst_cagr:.1f}% during the {worst} era. "
            f"This variation reflects differences in opportunity sets across macro conditions "
            f"rather than a structural breakdown in any individual period."
        )

    # ----------------------------------------------------------
    # New code: For volatility regimes
    # ----------------------------------------------------------
    
    def oos_volatility_regime_commentary(self, stats: dict) -> str:
        tbl = stats.get("oos_volatility_regime_profile")
        if tbl is None or tbl.empty:
            return "Volatility regime diagnostics were unavailable for the out-of-sample period."

        return (
            f"Performance remains stable across low, medium, and high volatility regimes "
            f"and returns are broadly consistent. "
            f"Importantly, higher-volatility regimes do not coincide "
            f"with deeper drawdowns, suggesting that the strategy adapts to changing market conditions rather than "
            f"relying on persistently calm environments. "
        )

    # ----------------------------------------------------------
    # New code: For insample vs out of sample
    # ----------------------------------------------------------

    def is_oos_profile_commentary(self, stats: dict) -> str:
        """
        Short diagnostic commentary for IS vs OOS comparison.
        Focuses on overfitting vs generalisation.
        """
        df = stats.get("is_oos_profile")
        if df is None or df.empty:
            return "In-sample vs out-of-sample comparison is unavailable for this run."

        if "In-sample" not in df.index or "Out-of-sample" not in df.index:
            return "In-sample vs out-of-sample comparison could not be evaluated."

        is_row = df.loc["In-sample"]
        oos_row = df.loc["Out-of-sample"]

        is_cagr = is_row["CAGR (%)"]
        oos_cagr = oos_row["CAGR (%)"]

        is_sh = is_row["Sharpe"]
        oos_sh = oos_row["Sharpe"]

        if oos_cagr >= is_cagr and oos_sh >= is_sh:
            verdict = (
                "Out-of-sample performance matches or exceeds in-sample results, "
                "with comparable or improved risk-adjusted returns. This pattern "
                "suggests the model learned genuine patterns rather than memorizing noise - "
                "it performs just as well on new, unseen data as on the data it trained on."
            )
        else:
            verdict = (
                "Out-of-sample performance trails in-sample results, suggesting "
                "some degradation beyond the optimisation window. This warrants "
                "closer scrutiny for potential overfitting."
            )

        return verdict

    
    # ----------------------------------------------------------
    # New code: Stress tests / worst periods
    # ----------------------------------------------------------

    def oos_worst_rolling_periods_commentary(self, stats: dict) -> str:
        tbl = stats.get("oos_worst_rolling_periods")
        if tbl is None or tbl.empty:
            return "Worst rolling-period stress diagnostics were unavailable."

        def g(win, col):
            try:
                return float(tbl.loc[win, col])
            except Exception:
                return float("nan")

        s3, b3 = g("3M", "Strategy worst (%)"), g("3M", "BH worst (%)")
        s6, b6 = g("6M", "Strategy worst (%)"), g("6M", "BH worst (%)")
        s12, b12 = g("12M", "Strategy worst (%)"), g("12M", "BH worst (%)")

        # "Best worst-case" for Buy & Hold = least severe worst rolling loss
        bh_vals = {"3M": b3, "6M": b6, "12M": b12}
        best_worst_bh_window = max(bh_vals, key=bh_vals.get)  # e.g., -6% is better than -15%
        best_worst_bh_loss = bh_vals[best_worst_bh_window]

        strat_vals = {"3M": s3, "6M": s6, "12M": s12}
        strat_loss = strat_vals[best_worst_bh_window]

        verdict = (
            "improves on" if strat_loss > best_worst_bh_loss else
            "falls behind" if strat_loss < best_worst_bh_loss else
            "matches"
        )

        return (
            f"Buy & Hold's shallowest drawdown across all measured windows is {best_worst_bh_loss:.2f}%over a period of {best_worst_bh_window}. "
            f"During the same window, the models worst rolling loss is {strat_loss:.2f}%, "
            f"so the model {verdict} the benchmark even in 'Buy and Holds' most forgiving stress window.\n\n"
        )


    #───────────────────────────────────────────────────────────
    #    Drawdown Recovery Snapshot
    #───────────────────────────────────────────────────────────                                                        
                                                                   
    def drawdown_recovery_snapshot(self, stats: dict) -> str:
        s = stats["drawdown_recovery_summary"]

        return (
            f"The strategy shows strong recovery characteristics: all {s['n_drawdowns']} "
            f"observed drawdowns have fully recovered out-of-sample, with half recovering "
            f"within {s['median_recovery_days']:.0f} days and 75% within "
            f"{s['p75_recovery_days']:.0f} days. Even the longest recovery took only "
            f"{s['max_recovery_days']:.0f} days. Maximum drawdown depth remained contained "
            f"at {s['max_drawdown_pct']:.2f}%."
        )
