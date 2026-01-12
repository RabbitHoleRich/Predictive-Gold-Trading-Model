# signal_strength.py
#
# Human-readable signal interpretation layer
# Translates raw prob/rank into clear BUY/SELL/HOLD signals
#
# CRITICAL: This is PRESENTATION ONLY - does not change position logic

from typing import Dict


def interpret_classifier_signal(prob: float) -> str:
    """
    Interpret classifier probability as directional signal.
    
    Classifier answers: "Will price go up?"
    - prob > 0.75 → Very confident YES
    - prob > 0.60 → Moderately confident YES
    - prob > 0.50 → Slightly confident YES
    - prob < 0.30 → Very confident NO
    - prob < 0.45 → Moderately confident NO
    - else → Uncertain
    
    Args:
        prob: Classifier probability [0, 1]
        
    Returns:
        Signal string: STRONG BUY, BUY, WEAK BUY, HOLD, SELL, STRONG SELL
    """
    if prob >= 0.80:
        return "STRONG BUY"
    elif prob >= 0.65:
        return "BUY"
    elif prob >= 0.55:
        return "WEAK BUY"
    elif prob >= 0.45:
        return "HOLD"
    elif prob >= 0.30:
        return "WEAK SELL"
    else:
        return "STRONG SELL"


def interpret_regressor_signal(rank: float) -> str:
    """
    Interpret regressor rank as opportunity quality signal.
    
    Regressor answers: "How good is this opportunity relative to recent history?"
    - rank > 0.75 → Excellent opportunity
    - rank > 0.60 → Good opportunity
    - rank > 0.50 → Fair opportunity
    - rank < 0.25 → Poor opportunity
    - rank < 0.40 → Below average opportunity
    - else → Average
    
    Args:
        rank: Regressor percentile rank [0, 1]
        
    Returns:
        Signal string: STRONG BUY, BUY, WEAK BUY, HOLD, SELL, STRONG SELL
    """
    if rank >= 0.80:
        return "STRONG BUY"
    elif rank >= 0.65:
        return "BUY"
    elif rank >= 0.50:
        return "WEAK BUY"
    elif rank >= 0.40:
        return "HOLD"
    elif rank >= 0.25:
        return "WEAK SELL"
    else:
        return "STRONG SELL"


def interpret_combined_signal(
    prob: float, 
    rank: float, 
    model_position: int,
    regime: int = 1
) -> str:
    """
    Interpret combined signal considering both models and position.
    
    CRITICAL: This function is PRESENTATION ONLY.
    It does NOT change model_position - only interprets it for humans.
    
    Logic:
    - If position = 0 (flat): Use classifier to determine bearish strength
    - If position = 1 (long): Combine classifier + regressor for bullish strength
    
    Args:
        prob: Classifier probability
        rank: Regressor rank
        model_position: Actual position (0=flat, 1=long)
        regime: Market regime (for context)
        
    Returns:
        Combined signal: STRONG BUY, BUY, WEAK BUY, HOLD, SELL, STRONG SELL
    """
    
    # If flat, interpret based on why we're flat
    if model_position == 0:
        # Very bearish classifier
        if prob <= 0.30:
            return "STRONG SELL"
        # Moderately bearish classifier
        elif prob <= 0.40:
            return "SELL"
        # Weak signals or just below threshold
        elif prob <= 0.48:
            return "WEAK SELL"
        # Near neutral
        else:
            return "HOLD"
    
    # If long, combine both models
    else:
        # Both very bullish
        if prob >= 0.75 and rank >= 0.75:
            return "STRONG BUY"
        
        # Both bullish
        elif prob >= 0.60 and rank >= 0.60:
            return "BUY"
        
        # Mixed but still positive
        elif prob >= 0.55 or rank >= 0.55:
            return "WEAK BUY"
        
        # Marginal long (barely met threshold)
        else:
            return "HOLD (Long)"


def get_signal_explanation(
    classifier_signal: str,
    regressor_signal: str,
    combined_signal: str,
    prob: float,
    rank: float,
    regime: int
) -> str:
    """
    Generate human-readable explanation of the signal.
    
    Returns multi-line explanation string.
    """
    regime_names = {0: "Bull", 1: "Neutral", 2: "Defensive"}
    regime_name = regime_names.get(regime, "Unknown")
    
    explanation = []
    explanation.append("Signal Interpretation:")
    explanation.append(f"  Market Regime: {regime_name}")
    explanation.append("")
    explanation.append(f"  Classifier (Direction):  {classifier_signal} (prob={prob:.1%})")
    explanation.append(f"  Regressor (Opportunity): {regressor_signal} (rank={rank:.1%})")
    explanation.append(f"  Combined Signal:         {combined_signal}")
    explanation.append("")
    
    # Add context
    if "STRONG" in combined_signal:
        explanation.append("  → High conviction signal")
    elif "WEAK" in combined_signal:
        explanation.append("  → Low conviction signal - marginal threshold")
    elif combined_signal == "HOLD":
        explanation.append("  → Neutral - no clear opportunity")
    
    return "\n".join(explanation)


def get_all_signal_strengths(
    prob: float,
    rank: float,
    model_position: int,
    regime: int = 1
) -> Dict[str, str]:
    """
    Get all signal interpretations in one call.
    
    Returns dict with:
    - classifier_signal
    - regressor_signal
    - combined_signal
    - explanation
    """
    classifier_signal = interpret_classifier_signal(prob)
    regressor_signal = interpret_regressor_signal(rank)
    combined_signal = interpret_combined_signal(prob, rank, model_position, regime)
    
    explanation = get_signal_explanation(
        classifier_signal,
        regressor_signal,
        combined_signal,
        prob,
        rank,
        regime
    )
    
    return {
        'classifier_signal': classifier_signal,
        'regressor_signal': regressor_signal,
        'combined_signal': combined_signal,
        'explanation': explanation
    }


# ============================================================================
# Quick Formatting Helpers
# ============================================================================

def format_signal_summary(signal: Dict) -> str:
    """
    Format signal dict as human-readable summary.
    
    Args:
        signal: Signal dict from generate_signal()
        
    Returns:
        Formatted string
    """
    if 'classifier_signal' not in signal:
        # Add interpretations if not present
        strengths = get_all_signal_strengths(
            signal['prob'],
            signal['rank'],
            signal['model_position'],
            signal['regime']
        )
        signal.update(strengths)
    
    lines = []
    lines.append("="*60)
    lines.append(f"GOLD SIGNAL - {signal['date']}")
    lines.append("="*60)
    lines.append("")
    lines.append(f"Position: {signal['model_position']} (1=Long, 0=Flat)")
    lines.append(f"Price: ${signal.get('gold_close', 0):.2f}")
    lines.append("")
    lines.append(signal['explanation'])
    lines.append("")
    lines.append("Raw Metrics:")
    lines.append(f"  Probability: {signal['prob']:.1%}")
    lines.append(f"  Rank: {signal['rank']:.1%}")
    
    if 'rank_lower' in signal and signal['rank_lower'] is not None:
        lines.append(f"  Rank CI: [{signal['rank_lower']:.1%}, {signal['rank_upper']:.1%}]")
    
    if signal.get('has_contradiction'):
        lines.append("")
        lines.append(f"⚠️  Contradiction detected: {signal.get('contradiction_type')}")
    
    lines.append("="*60)
    
    return "\n".join(lines)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing signal strength interpretation...\n")
    
    # Test cases
    test_cases = [
        {"prob": 0.85, "rank": 0.82, "pos": 1, "regime": 0, "desc": "Very bullish"},
        {"prob": 0.65, "rank": 0.70, "pos": 1, "regime": 0, "desc": "Bullish"},
        {"prob": 0.55, "rank": 0.52, "pos": 1, "regime": 1, "desc": "Weak bullish"},
        {"prob": 0.48, "rank": 0.65, "pos": 0, "regime": 1, "desc": "Flat (prob failed)"},
        {"prob": 0.35, "rank": 0.25, "pos": 0, "regime": 2, "desc": "Bearish"},
        {"prob": 0.25, "rank": 0.15, "pos": 0, "regime": 2, "desc": "Very bearish"},
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"Test {i}: {tc['desc']}")
        print(f"  Prob={tc['prob']:.2f}, Rank={tc['rank']:.2f}, Pos={tc['pos']}, Regime={tc['regime']}")
        
        strengths = get_all_signal_strengths(
            tc['prob'],
            tc['rank'],
            tc['pos'],
            tc['regime']
        )
        
        print(f"  Classifier:  {strengths['classifier_signal']}")
        print(f"  Regressor:   {strengths['regressor_signal']}")
        print(f"  Combined:    {strengths['combined_signal']}")
        print()
