"""
MODULAR research_layer.py

Research & diagnostics orchestrator for gold_engine.
Coordinates:
    data → stats → charts → commentary → reports

NO model logic lives here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt

import research_data as rdata
import research_stats as rstats
import research_charts as rcharts
from IPython.display import Markdown

from research_commentary import Commentary
from research_stats import compute_is_oos_profile

# ============================================================
# Research Layer
# ============================================================

@dataclass
class ResearchLayer:
    backtest_path: Path
    summary_path: Optional[Path] = None

    data: Dict = field(default_factory=dict)
    stats: Dict = field(default_factory=dict)
    figs: Dict = field(default_factory=dict)
    comments: Dict = field(default_factory=dict)

    # --------------------------------------------------------
    @classmethod
    def from_defaults(cls):
        engine_root = Path(__file__).resolve().parent
        return cls(
            backtest_path=engine_root / "gold_walkforward_current_backtest.csv",
            summary_path=engine_root / "gold_walkforward_current_summary.csv",
        )

    # --------------------------------------------------------
    def load_all(self):
        rd = rdata.load_research_data(
            backtest_path=self.backtest_path,
            walkforward_summary_path=self.summary_path,
        )

        self.data["oos"] = rd.backtest_df
        self.data["summary"] = rd.walkforward_df

        if rd.insample_backtest_path:
            df_is = pd.read_csv(rd.insample_backtest_path)
            if "date" in df_is.columns:
                df_is["date"] = pd.to_datetime(df_is["date"])
            self.data["is"] = df_is
        else:
            self.data["is"] = None


    def compute_rolling_data(self):
        """
        Compute derived rolling dataframes required by charts and stats.
        """
        if "oos" not in self.data:
            raise RuntimeError("OOS data not loaded")
    
        self.data["rolling"] = rstats.compute_rolling_stats(
            self.data["oos"]
        )

    # -------------------------------------------------------

    def compute_stats(self):
        if "oos" not in self.data:
            raise RuntimeError("Must run load_all() first.")

        oos = self.data["oos"]

        # ==================================================
        # Core stats
        # ==================================================
        if "rolling" not in self.data:
            raise RuntimeError("Rolling data not computed")
        rolling = self.data["rolling"]
 
        self.stats["performance_summary"] = rstats.compute_performance_summary(oos)
        self.stats["subperiods"] = rstats.compute_subperiod_performance(oos)

        # ==================================================
        # IS / OOS
        # ==================================================
      
        if self.data.get("is") is not None:
            self.stats["is_oos"] = rstats.compute_is_oos_stats(self.data["is"], oos)
            self.stats["is_oos_profile"] = compute_is_oos_profile(self.data["is"], oos)
        else:
            self.stats["is_oos"] = None
            self.stats["is_oos_profile"] = None

        # ==================================================
        # Profile stats
        # ==================================================
        from research_trade_stats import compute_trade_duration_payoff_stats
        from research_drawdown_stats import compute_drawdown_depth_duration_stats
        from research_equity_stats import compute_equity_curve_stats
        from research_underwater_stats import compute_underwater_drawdown_stats
        from research_rolling_stats import compute_rolling_sharpe_profile
        from research_volatility_stats import compute_rolling_volatility_profile
        from research_retvol_stats import compute_return_vs_vol_profile
        from research_maxgain_stats import compute_annual_max_gain_profile
        from research_maxloss_stats import compute_annual_max_loss_profile
        from research_subperiod_stats import compute_subperiod_profile
        from research_conditional_stats import compute_oos_volatility_regime_profile
        from research_stress_stats import compute_oos_worst_rolling_periods
        from research_recovery_stats import compute_drawdown_recovery_stats
        from data_diagnostics import build_data_coverage_diagnostics, load_data_cleaning_aggregate
        from research_data import load_data_cleaning_summary

        # Trades
        trades = rcharts.extract_trades(oos)
        if not trades.empty:
            trades = trades.copy()
            trades["holding_days"] = trades["holding_days"].astype(int)
            trades["trade_return_pct"] = trades["trade_return"] * 100.0
            self.stats["trade_profile"] = compute_trade_duration_payoff_stats(trades)
        else:
            self.stats["trade_profile"] = None

        # Drawdowns
        dd_df = rcharts.extract_drawdowns(oos)
        self.stats["drawdown_profile"] = (
            compute_drawdown_depth_duration_stats(dd_df) if not dd_df.empty else None
        )

        # Equity curve stats (KEY FIX)
        self.stats["equity_profile"] = compute_equity_curve_stats(oos)

        # Other profiles
        self.stats["underwater_profile"] = compute_underwater_drawdown_stats(oos)
        self.stats["rolling_sharpe_profile"] = compute_rolling_sharpe_profile(rolling)
        self.stats["volatility_profile"] = compute_rolling_volatility_profile(oos)
        self.stats["ret_vs_vol_profile"] = compute_return_vs_vol_profile(oos)
        self.stats["maxgain_profile"] = compute_annual_max_gain_profile(oos)
        self.stats["maxloss_profile"] = compute_annual_max_loss_profile(oos)
        self.stats["subperiod_profile"] = compute_subperiod_profile(self.stats["subperiods"])

        # ==================================================
        # OOS diagnostics
        # ==================================================
        vol_res = compute_oos_volatility_regime_profile(oos, window=63)
        self.stats["oos_volatility_regime_profile"] = vol_res.table
        self.stats["oos_volatility_regime_thresholds"] = vol_res.thresholds

        wr = compute_oos_worst_rolling_periods(oos, windows=(63, 126, 252))
        self.stats["oos_worst_rolling_periods"] = wr.table

        recovery = compute_drawdown_recovery_stats(oos)
        self.stats["drawdown_recovery_episodes"] = recovery["episodes"]
        self.stats["drawdown_recovery_summary"] = recovery["summary"]

        # ==================================================
        # Data diagnostics
        # ==================================================
        self.stats["data_coverage_diagnostics"] = build_data_coverage_diagnostics()

        # ---------- Data coverage summary ----------
        cov = self.stats["data_coverage_diagnostics"]

        summary = {
            "Total days": len(cov),

            "Price coverage (%)": cov["has_price"].mean() * 100,
            "Price coverage (days)": int(cov["has_price"].sum()),

            "Sentiment coverage (%)": cov["has_sentiment"].mean() * 100,
            "Sentiment coverage (days)": int(cov["has_sentiment"].sum()),

            "COT coverage (%)": cov["has_cot"].mean() * 100,
            "COT coverage (days)": int(cov["has_cot"].sum()),

            "Full coverage (%)": (
                (cov["has_price"] & cov["has_sentiment"] & cov["has_cot"]).mean() * 100
            ),
            "Full coverage (days)": int(
                (cov["has_price"] & cov["has_sentiment"] & cov["has_cot"]).sum()
            ),
        }

        self.stats["data_coverage_summary"] = (
            pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
            .round(2)
        )


        self.stats["data_cleaning_aggregate"] = load_data_cleaning_aggregate()

        cleaning_df = load_data_cleaning_summary()
        self.stats["data_cleaning_summary_raw"] = cleaning_df

        if cleaning_df is not None and not cleaning_df.empty:
            total_before = cleaning_df["rows_before"].sum()
            total_removed = cleaning_df["rows_removed"].sum()

            self.stats["data_cleaning_summary"] = pd.DataFrame([{
                "total_rows_before": total_before,
                "total_rows_removed": total_removed,
                "pct_rows_removed": (total_removed / total_before) * 100 if total_before > 0 else 0.0,
                "invalid_date_rows": cleaning_df["invalid_date_rows"].sum(),
                "duplicate_date_rows": cleaning_df["duplicate_date_rows"].sum(),
                "non_positive_close_rows": cleaning_df["non_positive_close_rows"].sum(),
                "negative_volume_rows": cleaning_df["negative_volume_rows"].sum(),
                "assets_processed": cleaning_df["asset_table"].nunique(),
            }])
        else:
            self.stats["data_cleaning_summary"] = None

    # Heatmap code

        if self.data.get("summary") is not None:
            from research_heatmap_stats import compute_heatmap_profile
            heatmap = compute_heatmap_profile(self.data["summary"])

            self.data["heatmap"] = heatmap["table"]
            self.stats["heatmap_profile"] = heatmap["stats"]

     # ==================================================
     # Commentary (NO generate_all)
     # ==================================================

    def generate_commentary(self):
        """
        Generate natural-language commentary from computed stats.
        """
        commentary = Commentary(
            performance_summary=self.stats["performance_summary"],
            diagnostics=None,
        )

        self.comments = {}

        self.comments["performance_summary"] = commentary.performance_summary()

        if self.stats.get("trade_profile"):
            self.comments["trade_profile"] = commentary.trade_profile_commentary(self.stats)

        if self.stats.get("drawdown_profile"):
            self.comments["drawdown_profile"] = commentary.drawdown_profile_commentary(self.stats)

        if self.stats.get("equity_profile"):
            self.comments["equity_profile"] = commentary.equity_profile_commentary(self.stats)

        if self.stats.get("underwater_profile"):
            self.comments["drawdowns"] = commentary.underwater_drawdown_commentary(self.stats)

        if self.stats.get("rolling_sharpe_profile"):
            self.comments["rolling_profile"] = commentary.rolling_sharpe_commentary(self.stats)

        if self.stats.get("volatility_profile"):
            self.comments["volatility_profile"] = commentary.rolling_volatility_commentary(self.stats)

        if self.stats.get("ret_vs_vol_profile"):
            self.comments["ret_vs_vol_profile"] = commentary.return_vs_vol_commentary(self.stats)

        if self.stats.get("maxloss_profile"):
            self.comments["max_losses"] = commentary.maxloss_profile_commentary(self.stats)

        if self.stats.get("maxgain_profile"):
            self.comments["maxgain_profile"] = commentary.maxgain_profile_commentary(self.stats)
    
        if self.stats.get("heatmap_profile"):
            self.comments["heatmap_profile"] = commentary.heatmap_commentary(self.stats)

        if self.stats.get("subperiod_profile"):
            self.comments["subperiod_profile"] = commentary.subperiod_profile_commentary(self.stats)
    
        df = self.stats.get("is_oos_profile")
        if df is not None and not df.empty:
            self.comments["is_oos_profile"] = commentary.is_oos_profile_commentary(self.stats)

        if self.stats.get("oos_volatility_regime_profile") is not None:
            self.comments["vol_regime"] = commentary.oos_volatility_regime_commentary(self.stats)

        if self.stats.get("oos_worst_rolling_periods") is not None:
            self.comments["oos_worst_rolling_periods"] = (
                commentary.oos_worst_rolling_periods_commentary(self.stats)
            )
        
        if self.stats.get("drawdown_recovery_summary") is not None:
            self.comments["drawdown_recovery_snapshot"] = (
                commentary.drawdown_recovery_snapshot(self.stats)
            )
  
             
    # --------------------------------------------------------
    # NOTEBOOK DISPLAY HELPERS
    # --------------------------------------------------------
    def show_performance_summary(self):
        """
        Display a side-by-side performance summary
        comparing Strategy vs Buy & Hold.
        """
        perf = self.stats.get("performance_summary")
        if perf is None:
            raise RuntimeError("Performance summary not computed. Run rl.run() first.")

        df = pd.DataFrame([
            {
                "Metric": "CAGR (%)",
                "Strategy": perf.strat_cagr * 100,
                "Buy & Hold": perf.bh_cagr * 100,
            },
            {
                "Metric": "Total Return (%)",
                "Strategy": perf.strat_total_return * 100,
                "Buy & Hold": perf.bh_total_return * 100,
            },
            {
                "Metric": "Max DD (%)",
                "Strategy": perf.strat_max_dd * 100,
                "Buy & Hold": perf.bh_max_dd * 100,
            },
            {
                "Metric": "Sharpe",
                "Strategy": perf.strat_sharpe,
                "Buy & Hold": perf.bh_sharpe,
            },
            {
                "Metric": "Time in Market (%)",
                "Strategy": perf.time_in_market * 100,
                "Buy & Hold": 100.0,
            },
            {
                "Metric": "Hit Rate (%)",
                "Strategy": perf.hit_rate * 100,
                "Buy & Hold": None,
            },
            {
                "Metric": "Trades",
                "Strategy": perf.n_trades,
                "Buy & Hold": None,
            },
        ])

        display(
            df.style.format({
                "Strategy": "{:.2f}",
                "Buy & Hold": "{:.2f}",
            })
        )

    def show_data_cleaning_summary(self):
        df = self.stats.get("data_cleaning_aggregate")
        if df is None:
            print("No data cleaning summary available.")
            return

        display(
            df.style.format({
                "rows_before": "{:,.0f}",
                "rows_removed": "{:,.0f}",
            })
        )

        
    # --------------------------------------------------------
    def run(self, show=False, save_dir=None):
        self.load_all()
        self.compute_rolling_data()
        self.compute_stats()
        self.generate_commentary()
        
        return self

     
