"""
signal_monitor.py

Real-time monitoring dashboard for gold signal quality.
Tracks calibration, contradictions, and performance over time.

Usage:
    python3 signal_monitor.py --days 30        # Monitor last 30 days
    python3 signal_monitor.py --alert          # Check for issues and alert
    python3 signal_monitor.py --dashboard      # Full dashboard
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    from live_signal import generate_signal_for_model
    from five_model_gold import build_gold_dataset
except ImportError:
    print("Warning: Could not import live_signal modules")


class SignalMonitor:
    """Monitor signal quality over time"""
    
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.alerts = []
        self.metrics = {}
        
    def load_signal_history(self, signal_file="live_outputs/live_signals.csv"):
        """Load historical signals"""
        try:
            df = pd.read_csv(signal_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=self.lookback_days)
            df = df[df['date'] >= cutoff]
            
            return df
        except FileNotFoundError:
            print(f"Warning: Signal history file not found: {signal_file}")
            return pd.DataFrame()
    
    def load_diagnostics(self, diag_file="live_outputs/signal_diagnostics.csv"):
        """Load diagnostic data if available"""
        try:
            df = pd.read_csv(diag_file)
            df['date'] = pd.to_datetime(df['date'])
            
            cutoff = datetime.now() - timedelta(days=self.lookback_days)
            df = df[df['date'] >= cutoff]
            
            return df
        except FileNotFoundError:
            return pd.DataFrame()
    
    def check_calibration(self, signals_df):
        """Check if classifier and regressor are well calibrated"""
        print("\n" + "="*80)
        print("CALIBRATION MONITORING")
        print("="*80)
        
        if 'fwd_ret_20d' not in signals_df.columns:
            print("⚠️  Forward returns not yet available for recent signals")
            return
        
        # Filter to signals with known outcomes
        valid = signals_df[signals_df['fwd_ret_20d'].notna()].copy()
        
        if len(valid) == 0:
            print("⚠️  No signals with realized outcomes yet")
            return
        
        # Regressor calibration
        if 'reg_pred_20d' in valid.columns:
            correlation = valid['reg_pred_20d'].corr(valid['fwd_ret_20d'])
            mae = (valid['reg_pred_20d'] - valid['fwd_ret_20d']).abs().mean()
            
            print(f"\nRegressor Calibration:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  MAE: {mae:.2f}%")
            
            self.metrics['regressor_correlation'] = correlation
            self.metrics['regressor_mae'] = mae
            
            # Alert if calibration degraded
            if correlation < 0.2:
                self.alerts.append({
                    'severity': 'HIGH',
                    'type': 'CALIBRATION',
                    'message': f'Regressor correlation dropped to {correlation:.3f}'
                })
            elif mae > 5.0:
                self.alerts.append({
                    'severity': 'MEDIUM',
                    'type': 'CALIBRATION',
                    'message': f'Regressor MAE elevated: {mae:.2f}%'
                })
        
        # Classifier calibration
        if 'prob' in valid.columns:
            # Bin by probability and check actual win rate
            valid['prob_bin'] = pd.cut(valid['prob'], bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0])
            
            print(f"\nClassifier Calibration:")
            for bin_name, group in valid.groupby('prob_bin', observed=False):
                if len(group) > 0:
                    win_rate = (group['fwd_ret_20d'] > 0).mean()
                    avg_prob = group['prob'].mean()
                    expected_win = avg_prob
                    calibration_error = abs(win_rate - expected_win)
                    
                    print(f"  {bin_name}: n={len(group)}, "
                          f"actual={win_rate*100:.1f}%, "
                          f"expected={expected_win*100:.1f}%, "
                          f"error={calibration_error*100:.1f}%")
                    
                    # Alert if severe miscalibration
                    if len(group) >= 5 and calibration_error > 0.2:
                        self.alerts.append({
                            'severity': 'MEDIUM',
                            'type': 'CALIBRATION',
                            'message': f'Classifier miscalibrated in {bin_name}: {calibration_error*100:.1f}% error'
                        })
    
    def check_contradictions(self, signals_df):
        """Monitor classifier-regressor contradictions"""
        print("\n" + "="*80)
        print("CONTRADICTION MONITORING")
        print("="*80)
        
        if 'prob' not in signals_df.columns or 'rank' not in signals_df.columns:
            print("⚠️  Insufficient data for contradiction analysis")
            return
        
        # Type 1: High prob, low rank
        type1 = (signals_df['prob'] > 0.75) & (signals_df['rank'] < 0.25)
        
        # Type 2: Low prob, high rank
        type2 = (signals_df['prob'] < 0.25) & (signals_df['rank'] > 0.75)
        
        total = len(signals_df)
        type1_count = type1.sum()
        type2_count = type2.sum()
        
        print(f"\nContradictions in last {self.lookback_days} days:")
        print(f"  Type 1 (high prob, low rank): {type1_count}/{total} ({type1_count/total*100:.1f}%)")
        print(f"  Type 2 (low prob, high rank): {type2_count}/{total} ({type2_count/total*100:.1f}%)")
        
        self.metrics['type1_contradictions'] = type1_count
        self.metrics['type2_contradictions'] = type2_count
        
        # Show recent contradictions
        if type1_count > 0:
            print(f"\n  Recent Type 1 contradictions:")
            recent_type1 = signals_df[type1].tail(3)
            for _, row in recent_type1.iterrows():
                print(f"    {row['date']}: prob={row['prob']:.3f}, rank={row['rank']:.3f}, pos={row.get('model_position', '?')}")
        
        if type2_count > 0:
            print(f"\n  Recent Type 2 contradictions:")
            recent_type2 = signals_df[type2].tail(3)
            for _, row in recent_type2.iterrows():
                print(f"    {row['date']}: prob={row['prob']:.3f}, rank={row['rank']:.3f}, pos={row.get('model_position', '?')}")
        
        # Alert if contradictions are frequent
        if type1_count > total * 0.15:
            self.alerts.append({
                'severity': 'HIGH',
                'type': 'CONTRADICTION',
                'message': f'Frequent Type 1 contradictions: {type1_count}/{total}'
            })
        
        if type2_count > total * 0.15:
            self.alerts.append({
                'severity': 'MEDIUM',
                'type': 'CONTRADICTION',
                'message': f'Frequent Type 2 contradictions: {type2_count}/{total}'
            })
    
    def check_rank_stability(self, signals_df):
        """Monitor ranking stability"""
        print("\n" + "="*80)
        print("RANK STABILITY MONITORING")
        print("="*80)
        
        if 'rank' not in signals_df.columns or len(signals_df) < 2:
            print("⚠️  Insufficient data for rank stability analysis")
            return
        
        ranks = signals_df['rank'].values
        
        # Basic stats
        print(f"\nRank distribution over {len(ranks)} days:")
        print(f"  Mean: {ranks.mean():.3f}")
        print(f"  Std:  {ranks.std():.3f}")
        print(f"  Min:  {ranks.min():.3f}")
        print(f"  Max:  {ranks.max():.3f}")
        
        self.metrics['rank_mean'] = ranks.mean()
        self.metrics['rank_std'] = ranks.std()
        
        # Check for jumps
        rank_changes = np.abs(np.diff(ranks))
        large_jumps = (rank_changes > 0.5).sum()
        
        print(f"\nRank changes:")
        print(f"  Mean daily change: {rank_changes.mean():.3f}")
        print(f"  Max daily change:  {rank_changes.max():.3f}")
        print(f"  Large jumps (>0.5): {large_jumps}")
        
        self.metrics['rank_large_jumps'] = large_jumps
        
        # Alert if unstable
        if ranks.std() < 0.05:
            self.alerts.append({
                'severity': 'MEDIUM',
                'type': 'RANK',
                'message': f'Rank distribution has low variance: {ranks.std():.3f}'
            })
        
        if large_jumps > len(ranks) * 0.25:
            self.alerts.append({
                'severity': 'HIGH',
                'type': 'RANK',
                'message': f'Unstable ranks: {large_jumps} large jumps in {len(ranks)} days'
            })
    
    def check_performance(self, signals_df):
        """Monitor actual trading performance"""
        print("\n" + "="*80)
        print("PERFORMANCE MONITORING")
        print("="*80)
        
        if 'model_position' not in signals_df.columns:
            print("⚠️  No position data available")
            return
        
        # Count positions
        total = len(signals_df)
        long_count = (signals_df['model_position'] == 1).sum()
        flat_count = (signals_df['model_position'] == 0).sum()
        
        print(f"\nPosition distribution over {total} days:")
        print(f"  Long: {long_count} ({long_count/total*100:.1f}%)")
        print(f"  Flat: {flat_count} ({flat_count/total*100:.1f}%)")
        
        self.metrics['pct_long'] = long_count / total if total > 0 else 0
        
        # Alert if always long or always flat
        if long_count == total:
            self.alerts.append({
                'severity': 'HIGH',
                'type': 'PERFORMANCE',
                'message': 'Strategy stuck in long position'
            })
        elif flat_count == total:
            self.alerts.append({
                'severity': 'HIGH',
                'type': 'PERFORMANCE',
                'message': 'Strategy stuck in flat position'
            })
        
        # Check forward returns if available
        if 'fwd_ret_20d' in signals_df.columns:
            valid = signals_df[signals_df['fwd_ret_20d'].notna()]
            
            if len(valid) > 0:
                long_signals = valid[valid['model_position'] == 1]
                
                if len(long_signals) > 0:
                    win_rate = (long_signals['fwd_ret_20d'] > 0).mean()
                    avg_ret = long_signals['fwd_ret_20d'].mean()
                    
                    print(f"\nLong signal performance ({len(long_signals)} signals):")
                    print(f"  Win rate: {win_rate*100:.1f}%")
                    print(f"  Avg return: {avg_ret:+.2f}%")
                    
                    self.metrics['win_rate'] = win_rate
                    self.metrics['avg_return'] = avg_ret
                    
                    # Alert if poor performance
                    if win_rate < 0.45:
                        self.alerts.append({
                            'severity': 'HIGH',
                            'type': 'PERFORMANCE',
                            'message': f'Low win rate: {win_rate*100:.1f}%'
                        })
                    
                    if avg_ret < -0.5:
                        self.alerts.append({
                            'severity': 'HIGH',
                            'type': 'PERFORMANCE',
                            'message': f'Negative average return: {avg_ret:+.2f}%'
                        })
    
    def check_regime_behavior(self, signals_df):
        """Monitor regime-specific behavior"""
        print("\n" + "="*80)
        print("REGIME MONITORING")
        print("="*80)
        
        if 'regime' not in signals_df.columns:
            print("⚠️  No regime data available")
            return
        
        regime_names = {0: "Bull", 1: "Neutral", 2: "Defensive"}
        
        print(f"\nRegime distribution:")
        for regime, count in signals_df['regime'].value_counts().sort_index().items():
            name = regime_names.get(regime, f"Unknown({regime})")
            print(f"  {name}: {count} days ({count/len(signals_df)*100:.1f}%)")
        
        # Check position by regime
        if 'model_position' in signals_df.columns:
            print(f"\nPosition by regime:")
            for regime in sorted(signals_df['regime'].unique()):
                regime_df = signals_df[signals_df['regime'] == regime]
                long_pct = (regime_df['model_position'] == 1).mean()
                name = regime_names.get(regime, f"Unknown({regime})")
                print(f"  {name}: {long_pct*100:.1f}% long")
    
    def generate_alert_report(self):
        """Generate alert summary"""
        print("\n" + "="*80)
        print("ALERT SUMMARY")
        print("="*80)
        
        if not self.alerts:
            print("\n✅ No alerts - system healthy")
            return
        
        # Group by severity
        high = [a for a in self.alerts if a['severity'] == 'HIGH']
        medium = [a for a in self.alerts if a['severity'] == 'MEDIUM']
        low = [a for a in self.alerts if a['severity'] == 'LOW']
        
        if high:
            print(f"\n🚨 HIGH PRIORITY ALERTS ({len(high)}):")
            for alert in high:
                print(f"  [{alert['type']}] {alert['message']}")
        
        if medium:
            print(f"\n⚠️  MEDIUM PRIORITY ALERTS ({len(medium)}):")
            for alert in medium:
                print(f"  [{alert['type']}] {alert['message']}")
        
        if low:
            print(f"\nℹ️  LOW PRIORITY ALERTS ({len(low)}):")
            for alert in low:
                print(f"  [{alert['type']}] {alert['message']}")
    
    def save_metrics(self, output_file="live_outputs/monitoring_metrics.json"):
        """Save metrics for tracking over time"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'lookback_days': self.lookback_days,
            'metrics': self.metrics,
            'alert_count': len(self.alerts),
            'high_alerts': len([a for a in self.alerts if a['severity'] == 'HIGH'])
        }
        
        # Append to history
        history = []
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                history = json.load(f)
        
        history.append(record)
        
        # Keep last 90 days of history
        cutoff = datetime.now() - timedelta(days=90)
        history = [h for h in history if datetime.fromisoformat(h['timestamp']) >= cutoff]
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nMetrics saved to {output_file}")
    
    def run_full_monitor(self):
        """Run complete monitoring suite"""
        signals_df = self.load_signal_history()
        
        if signals_df.empty:
            print("No signal history available for monitoring")
            return
        
        print(f"\nMonitoring {len(signals_df)} signals from last {self.lookback_days} days")
        
        self.check_rank_stability(signals_df)
        self.check_contradictions(signals_df)
        self.check_performance(signals_df)
        self.check_regime_behavior(signals_df)
        self.check_calibration(signals_df)
        self.generate_alert_report()
        self.save_metrics()


def main():
    parser = argparse.ArgumentParser(description="Monitor gold signal quality")
    parser.add_argument("--days", type=int, default=30, help="Lookback period in days")
    parser.add_argument("--alert", action="store_true", help="Alert mode (check for issues)")
    parser.add_argument("--dashboard", action="store_true", help="Full dashboard")
    
    args = parser.parse_args()
    
    monitor = SignalMonitor(lookback_days=args.days)
    
    if args.alert:
        # Just check for alerts
        signals_df = monitor.load_signal_history()
        if not signals_df.empty:
            monitor.check_rank_stability(signals_df)
            monitor.check_contradictions(signals_df)
            monitor.check_performance(signals_df)
            monitor.generate_alert_report()
    else:
        # Full dashboard
        monitor.run_full_monitor()


if __name__ == "__main__":
    main()
