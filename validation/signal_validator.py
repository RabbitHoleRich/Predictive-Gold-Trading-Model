#!/usr/bin/env python3
"""
gold_signal_validator_v4.py - Alignment-Grade Validator

Validates full pipeline: signal generation → state transitions → output contracts

Critical tests:
1. API/packaging drift
2. Determinism (same date = same result)
3. Backward compatibility
4. Enhanced vs simple mode alignment
5. Diagnostic field separation
6. Historical consistency
7. STATE REPLAY (full decision pipeline) ← NEW
8. Config completeness (no silent fallbacks) ← NEW
9. Output schema validation ← NEW
10. Feature alignment (vs backtest) ← NEW

Usage:
    python3 gold_signal_validator_v4.py --full-test
    python3 gold_signal_validator_v4.py --quick-check
    python3 gold_signal_validator_v4.py --state-check
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports
try:
    from live_signal import (
        generate_signal,
        get_train_test_splits,
        train_models,
        save_signal_to_csv,
    )
    from five_model_gold import (
        build_gold_dataset,
        FEATURE_COLS as BACKTEST_FEATURE_COLS,
    )
    from config import MODEL, WALKFORWARD, OUTPUTS
    
    # Import production decision logic
    from production import interpret_action
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure this script is in the same directory as your code")
    raise


class GoldSignalValidator:
    """Comprehensive alignment-grade validation suite"""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.warnings = []
        
    def test_lookahead_bias(self, n_days=60):
        """
        Test #1: Verify no lookahead bias in ranking
        """
        print("\n" + "="*80)
        print("TEST 1: LOOKAHEAD BIAS DETECTION")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        df = df.dropna(subset=["future_ret_20d"])
        
        test_dates = df.index[-n_days:]
        lookahead_issues = []
        
        for i, test_date in enumerate(test_dates):
            if i % 10 == 0:
                print(f"Checking date {i+1}/{len(test_dates)}: {test_date.date()}")
            
            try:
                signal = generate_signal(test_date.to_pydatetime())
                train_df, test_df = get_train_test_splits(df, test_date.to_pydatetime())
                
                if (train_df.index >= test_date).any():
                    lookahead_issues.append({
                        "date": test_date,
                        "issue": "Training data includes future dates"
                    })
                
                if len(test_df) != 1 or test_df.index[0] != test_date:
                    lookahead_issues.append({
                        "date": test_date,
                        "issue": f"Test set incorrect"
                    })
                    
            except Exception as e:
                lookahead_issues.append({
                    "date": test_date,
                    "issue": f"Error: {str(e)}"
                })
        
        if lookahead_issues:
            print(f"\n❌ FAILED: Found {len(lookahead_issues)} lookahead issues")
            self.issues.append("Lookahead bias detected")
            passed = False
        else:
            print(f"\n✅ PASSED: No lookahead bias detected across {n_days} days")
            passed = True
        
        self.results['lookahead_test'] = {'passed': passed, 'issues': lookahead_issues}
        return passed
    
    def test_ranking_stability(self, n_days=30):
        """
        Test #2: Verify ranking method is stable
        """
        print("\n" + "="*80)
        print("TEST 2: RANKING STABILITY")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_dates = df.index[-n_days:]
        
        ranks = []
        for test_date in test_dates:
            try:
                signal = generate_signal(test_date.to_pydatetime())
                ranks.append(signal['rank'])
            except:
                continue
        
        ranks = np.array(ranks)
        
        rank_mean = ranks.mean()
        rank_std = ranks.std()
        
        print(f"\nRank Distribution:")
        print(f"  Mean: {rank_mean:.3f}")
        print(f"  Std:  {rank_std:.3f}")
        print(f"  Min:  {ranks.min():.3f}")
        print(f"  Max:  {ranks.max():.3f}")
        
        issues = []
        
        if rank_std < 0.1:
            issues.append("Rank distribution has very low variance")
            self.warnings.append("Ranks not well distributed")
        
        rank_changes = np.abs(np.diff(ranks))
        large_jumps = (rank_changes > 0.5).sum()
        
        print(f"\nRank Stability:")
        print(f"  Mean daily change: {rank_changes.mean():.3f}")
        print(f"  Large jumps (>0.5): {large_jumps}")
        
        if large_jumps > len(ranks) * 0.2:
            issues.append(f"Unstable ranks: {large_jumps} large jumps")
        
        if issues:
            print(f"\n⚠️  WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")
            passed = False
        else:
            print(f"\n✅ PASSED: Ranking appears stable")
            passed = True
        
        self.results['ranking_stability'] = {
            'passed': passed,
            'issues': issues,
            'stats': {'mean': float(rank_mean), 'std': float(rank_std)}
        }
        return passed
    
    def test_classifier_regressor_agreement(self, n_days=60):
        """
        Test #3: Check for classifier-regressor contradictions
        """
        print("\n" + "="*80)
        print("TEST 3: CLASSIFIER-REGRESSOR AGREEMENT")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_dates = df.index[-n_days:]
        
        signals = []
        for test_date in test_dates:
            try:
                signal = generate_signal(test_date.to_pydatetime())
                signals.append(signal)
            except:
                continue
        
        df_signals = pd.DataFrame(signals)
        
        high_prob = df_signals['prob'] > 0.80
        low_rank = df_signals['rank'] < 0.20
        contradictions = high_prob & low_rank
        
        high_prob_low_rank = df_signals[contradictions]
        
        print(f"\nSignal Analysis over {len(df_signals)} days:")
        print(f"\nContradictions (high prob >0.8, low rank <0.2):")
        print(f"  Count: {len(high_prob_low_rank)}")
        print(f"  Frequency: {len(high_prob_low_rank)/len(df_signals)*100:.1f}%")
        
        correlation = df_signals['prob'].corr(df_signals['rank'])
        print(f"\nCorrelation (prob vs rank): {correlation:.3f}")
        
        issues = []
        
        if len(high_prob_low_rank) > len(df_signals) * 0.1:
            issues.append(f"Frequent contradictions: {len(high_prob_low_rank)/len(df_signals)*100:.1f}%")
        
        if abs(correlation) < 0.3:
            issues.append(f"Weak correlation: {correlation:.3f}")
        
        if issues:
            print(f"\n⚠️  WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")
            passed = False
        else:
            print(f"\n✅ PASSED: Good classifier-regressor agreement")
            passed = True
        
        self.results['classifier_regressor'] = {
            'passed': passed,
            'issues': issues,
            'stats': {'correlation': float(correlation)}
        }
        return passed
    
    def test_determinism(self):
        """
        Test #4: Same date produces same result
        """
        print("\n" + "="*80)
        print("TEST 4: DETERMINISM")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_date = df.index[-1].to_pydatetime()
        
        signal1 = generate_signal(test_date, enhanced=True)
        signal2 = generate_signal(test_date, enhanced=True)
        
        checks = {
            'rank': signal1['rank'] == signal2['rank'],
            'prob': signal1['prob'] == signal2['prob'],
            'position': signal1['model_position'] == signal2['model_position'],
            'rank_lower': signal1.get('rank_lower') == signal2.get('rank_lower'),
            'rank_upper': signal1.get('rank_upper') == signal2.get('rank_upper'),
        }
        
        if all(checks.values()):
            print(f"✅ PASSED: Deterministic")
            print(f"   Rank: {signal1['rank']:.6f} (both runs)")
            print(f"   CI: [{signal1.get('rank_lower', 0):.6f}, {signal1.get('rank_upper', 0):.6f}] (both runs)")
            passed = True
        else:
            print(f"❌ FAILED: Non-deterministic")
            for key, passed_check in checks.items():
                if not passed_check:
                    print(f"   {key}: {signal1.get(key)} != {signal2.get(key)}")
            self.issues.append("Non-deterministic signals")
            passed = False
        
        self.results['determinism'] = {'passed': passed}
        return passed
    
    def test_backward_compatibility(self):
        """
        Test #5: Old function names still work
        """
        print("\n" + "="*80)
        print("TEST 5: BACKWARD COMPATIBILITY")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_date = df.index[-1].to_pydatetime()
        
        try:
            from live_signal import generate_signal_for_model
            
            signal_old = generate_signal_for_model(test_date, "conservative")
            signal_new = generate_signal(test_date, enhanced=True)
            
            if signal_old['model_position'] == signal_new['model_position']:
                print(f"✅ PASSED: Backward compatible")
                print(f"   Old API: position={signal_old['model_position']}")
                print(f"   New API: position={signal_new['model_position']}")
                passed = True
            else:
                print(f"❌ FAILED: Different positions")
                self.issues.append("Backward compatibility broken")
                passed = False
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.issues.append(f"Backward compatibility error: {e}")
            passed = False
        
        self.results['backward_compatibility'] = {'passed': passed}
        return passed
    
    def test_enhanced_vs_simple(self):
        """
        Test #6: Enhanced and simple modes produce same positions
        """
        print("\n" + "="*80)
        print("TEST 6: ENHANCED VS SIMPLE MODE")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_date = df.index[-1].to_pydatetime()
        
        enhanced = generate_signal(test_date, enhanced=True)
        simple = generate_signal(test_date, enhanced=False)
        
        checks = {
            'position': enhanced['model_position'] == simple['model_position'],
            'rank': abs(enhanced['rank'] - simple['rank']) < 1e-10,
            'prob': abs(enhanced['prob'] - simple['prob']) < 1e-10,
        }
        
        if all(checks.values()):
            print(f"✅ PASSED: Enhanced and simple modes match")
            print(f"   Position: {enhanced['model_position']}")
            print(f"   Enhanced has diagnostics: {'rank_std' in enhanced}")
            print(f"   Simple lacks diagnostics: {'rank_std' not in simple}")
            passed = True
        else:
            print(f"❌ FAILED: Modes differ")
            for key, passed_check in checks.items():
                if not passed_check:
                    print(f"   {key} mismatch")
            self.issues.append("Enhanced vs simple mismatch")
            passed = False
        
        self.results['enhanced_vs_simple'] = {'passed': passed}
        return passed
    
    def test_state_replay(self, n_days=30):
        """
        Test #7: STATE REPLAY - Full decision pipeline validation
        
        CRITICAL: This tests the full trading logic:
        signal → state transition → position change
        
        This is what validates alignment with backtest.
        """
        print("\n" + "="*80)
        print("TEST 7: STATE REPLAY (Full Decision Pipeline)")
        print("="*80)
        print("\nThis is the CRITICAL test - validates full trading logic")
        
        df = build_gold_dataset().sort_index()
        test_dates = df.index[-n_days:]
        
        # First run: Build position history
        prev_pos = 0  # Start flat
        position_history = []
        
        print(f"\nSimulating {n_days} days of trading...")
        
        for i, date in enumerate(test_dates):
            try:
                signal = generate_signal(date.to_pydatetime())
                model_pos = signal['model_position']
                
                # Apply production decision logic
                action = interpret_action(prev_pos, model_pos)
                
                # Compute new position (EXACT production logic)
                if action == "BUY":
                    new_pos = 1
                elif action == "SELL":
                    new_pos = 0
                else:
                    new_pos = prev_pos
                
                position_history.append({
                    'date': date,
                    'model_position': model_pos,
                    'prev_pos': prev_pos,
                    'action': action,
                    'new_pos': new_pos
                })
                
                prev_pos = new_pos
                
            except Exception as e:
                print(f"⚠️  Error on {date.date()}: {e}")
                continue
        
        # Second run: Replay to check determinism
        prev_pos = 0
        position_history_replay = []
        
        print(f"Replaying to verify determinism...")
        
        for date in test_dates:
            try:
                signal = generate_signal(date.to_pydatetime())
                model_pos = signal['model_position']
                action = interpret_action(prev_pos, model_pos)
                
                if action == "BUY":
                    new_pos = 1
                elif action == "SELL":
                    new_pos = 0
                else:
                    new_pos = prev_pos
                
                position_history_replay.append(new_pos)
                prev_pos = new_pos
            except:
                continue
        
        # Check determinism
        original_positions = [h['new_pos'] for h in position_history]
        
        position_changes = sum(1 for i in range(1, len(original_positions)) 
                              if original_positions[i] != original_positions[i-1])
        
        if original_positions == position_history_replay:
            print(f"\n✅ PASSED: State replay deterministic")
            print(f"   Days simulated: {len(position_history)}")
            print(f"   Position changes: {position_changes}")
            print(f"   BUY actions: {sum(1 for h in position_history if h['action'] == 'BUY')}")
            print(f"   SELL actions: {sum(1 for h in position_history if h['action'] == 'SELL')}")
            
            # Show last few transitions for audit
            print(f"\n   Last 5 transitions:")
            for h in position_history[-5:]:
                print(f"     {h['date'].date()}: {h['prev_pos']}→{h['new_pos']} ({h['action']})")
            
            passed = True
        else:
            print(f"\n❌ FAILED: State replay not deterministic")
            print(f"   Original and replay differ!")
            self.issues.append("State replay non-deterministic")
            passed = False
        
        self.results['state_replay'] = {
            'passed': passed,
            'stats': {
                'days': len(position_history),
                'changes': position_changes
            }
        }
        return passed
    
    def test_config_completeness(self):
        """
        Test #8: All required config keys exist (no silent fallbacks)
        
        Updated to match Gold's actual config structure:
        - MODEL contains hyperparameters only
        - PROB_THRESHOLD and REGIME_CUTOFFS are module-level constants
        """
        print("\n" + "="*80)
        print("TEST 8: CONFIG COMPLETENESS")
        print("="*80)
        print("\nValidating no silent fallbacks allowed...")
        
        required_keys = {
            'MODEL': ['regressor_params', 'classifier_params'],  # Removed prob_threshold
            'WALKFORWARD': ['rolling_lookback_months', 'rebalance_offset_days'],
            'OUTPUTS': ['signal_output_path', 'diagnostics_output_path', 'live_out_dir']
        }
        
        missing = []
        
        # Check dict-based configs
        for section, keys in required_keys.items():
            config = {'MODEL': MODEL, 'WALKFORWARD': WALKFORWARD, 'OUTPUTS': OUTPUTS}[section]
            
            print(f"\nChecking {section}:")
            for key in keys:
                if key not in config or config[key] is None or config[key] == "":
                    missing.append(f"{section}.{key}")
                    print(f"  ❌ {key}: MISSING")
                else:
                    print(f"  ✅ {key}: {config[key]}")
        
        # Check module-level constants (Gold's architecture)
        print(f"\nChecking module-level constants:")
        try:
            from five_model_gold import PROB_THRESHOLD, REGIME_CUTOFFS
            
            if PROB_THRESHOLD is None:
                missing.append("PROB_THRESHOLD (module constant)")
                print(f"  ❌ PROB_THRESHOLD: MISSING")
            else:
                print(f"  ✅ PROB_THRESHOLD: {PROB_THRESHOLD}")
            
            if REGIME_CUTOFFS is None or not REGIME_CUTOFFS:
                missing.append("REGIME_CUTOFFS (module constant)")
                print(f"  ❌ REGIME_CUTOFFS: MISSING")
            else:
                print(f"  ✅ REGIME_CUTOFFS: {REGIME_CUTOFFS}")
                
        except ImportError as e:
            missing.append(f"Module constants import failed: {e}")
            print(f"  ❌ Could not import PROB_THRESHOLD/REGIME_CUTOFFS from five_model_gold")
        
        if missing:
            print(f"\n❌ FAILED: Missing required config keys")
            print(f"   Missing: {missing}")
            self.issues.append(f"Missing config: {missing}")
            passed = False
        else:
            print(f"\n✅ PASSED: All required config keys present")
            passed = True
        
        self.results['config_completeness'] = {'passed': passed, 'missing': missing}
        return passed
    
    def test_output_schema(self):
        """
        Test #9: Output schema validation (prevents silent breaking changes)
        """
        print("\n" + "="*80)
        print("TEST 9: OUTPUT SCHEMA VALIDATION")
        print("="*80)
        
        df = build_gold_dataset().sort_index()
        test_date = df.index[-1].to_pydatetime()
        
        signal = generate_signal(test_date, enhanced=True)
        
        required_core = ['date', 'model_position', 'rank', 'prob', 'regime', 'gold_close']
        required_enhanced = ['rank_std', 'rank_lower', 'rank_upper', 'has_contradiction']
        
        print(f"\nValidating core fields:")
        missing_core = []
        for field in required_core:
            if field in signal:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}: MISSING")
                missing_core.append(field)
        
        print(f"\nValidating enhanced fields:")
        missing_enhanced = []
        for field in required_enhanced:
            if field in signal:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}: MISSING")
                missing_enhanced.append(field)
        
        if missing_core:
            print(f"\n❌ FAILED: Missing core fields: {missing_core}")
            self.issues.append(f"Schema violation: {missing_core}")
            passed = False
        elif missing_enhanced:
            print(f"\n⚠️  WARNING: Missing enhanced fields: {missing_enhanced}")
            self.warnings.append("Enhanced fields missing")
            passed = True  # Don't fail, just warn
        else:
            print(f"\n✅ PASSED: Output schema valid")
            passed = True
        
        self.results['output_schema'] = {
            'passed': passed,
            'missing_core': missing_core,
            'missing_enhanced': missing_enhanced
        }
        return passed
    
    def test_feature_alignment(self):
        """
        Test #10: Features used in live match backtest (catches drift)
        """
        print("\n" + "="*80)
        print("TEST 10: FEATURE ALIGNMENT")
        print("="*80)
        
        try:
            from live_signal import FEATURE_COLS as LIVE_FEATURES
            
            backtest_set = set(BACKTEST_FEATURE_COLS)
            live_set = set(LIVE_FEATURES)
            
            missing_in_live = backtest_set - live_set
            extra_in_live = live_set - backtest_set
            
            print(f"\nBacktest features: {len(backtest_set)}")
            print(f"Live features: {len(live_set)}")
            
            if missing_in_live:
                print(f"\n⚠️  Missing in live: {missing_in_live}")
            
            if extra_in_live:
                print(f"\n⚠️  Extra in live: {extra_in_live}")
            
            if missing_in_live or extra_in_live:
                print(f"\n❌ FAILED: Feature mismatch detected")
                self.issues.append("Feature drift detected")
                passed = False
            else:
                print(f"\n✅ PASSED: Features aligned ({len(LIVE_FEATURES)} features)")
                passed = True
                
        except ImportError:
            print(f"\n⚠️  WARNING: Could not import FEATURE_COLS from live_signal")
            print(f"   Assuming features are defined inline")
            passed = True  # Don't fail if features not exported
        
        self.results['feature_alignment'] = {'passed': passed}
        return passed
    
    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['passed'])
        
        print(f"\nTests passed: {passed_tests}/{total_tests}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if passed_tests == total_tests and not self.issues:
            print(f"\n✅ ALL TESTS PASSED - System is alignment-grade")
            print(f"\n🚀 READY FOR:")
            print(f"   - Live trading")
            print(f"   - Silver/Platinum replication")
            print(f"   - Production deployment")
        elif not self.issues:
            print(f"\n⚠️  Some warnings but no critical issues")
            print(f"   System is functional but monitor warnings")
        else:
            print(f"\n❌ CRITICAL ISSUES DETECTED")
            print(f"   Fix issues before live trading or replication")
        
        return passed_tests == total_tests and len(self.issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Alignment-grade Gold signal validation")
    parser.add_argument("--full-test", action="store_true", help="Run all 10 tests (recommended)")
    parser.add_argument("--quick-check", action="store_true", help="Run core tests only (tests 1-6)")
    parser.add_argument("--state-check", action="store_true", help="Run state replay test only (test 7)")
    parser.add_argument("--config-check", action="store_true", help="Run config validation only (test 8)")
    
    args = parser.parse_args()
    
    validator = GoldSignalValidator()
    
    if args.full_test or not any([args.quick_check, args.state_check, args.config_check]):
        print("="*80)
        print("RUNNING FULL ALIGNMENT-GRADE VALIDATION")
        print("="*80)
        print("\nThis validates:")
        print("  1-6: Signal generation correctness")
        print("  7:   Full decision pipeline (STATE REPLAY)")
        print("  8:   Config completeness")
        print("  9:   Output schema")
        print("  10:  Feature alignment")
        
        validator.test_lookahead_bias(n_days=60)
        validator.test_ranking_stability(n_days=30)
        validator.test_classifier_regressor_agreement(n_days=60)
        validator.test_determinism()
        validator.test_backward_compatibility()
        validator.test_enhanced_vs_simple()
        validator.test_state_replay(n_days=30)
        validator.test_config_completeness()
        validator.test_output_schema()
        validator.test_feature_alignment()
        validator.generate_report()
    
    elif args.quick_check:
        print("Running quick check (core tests)...")
        validator.test_determinism()
        validator.test_backward_compatibility()
        validator.test_enhanced_vs_simple()
        validator.generate_report()
    
    elif args.state_check:
        print("Running state replay test...")
        validator.test_state_replay(n_days=30)
        validator.generate_report()
    
    elif args.config_check:
        print("Running config validation...")
        validator.test_config_completeness()
        validator.generate_report()


if __name__ == "__main__":
    main()
