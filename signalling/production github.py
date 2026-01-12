from datetime import datetime, timezone
import sys
import json
import os
import socket
import traceback
from pathlib import Path
import smtplib
from email.mime.text import MIMEText

from live_signal import (
    generate_signal,
    save_signal_to_csv,
)

# Import signal strength interpreter
from signal_strength import (
    get_all_signal_strengths,
    format_signal_summary,
)

from five_model_gold import build_gold_dataset
from config import OUTPUTS

# --------------------
# OUTPUTS CONFIG (now from config.py)
# --------------------
LIVE_OUT_DIR = OUTPUTS.get("live_out_dir", "live_outputs")
POSITION_STATE_FILE = OUTPUTS.get("position_state_file", "position_state.json")

# Email settings (with env var override)
SEND_EMAIL = os.getenv("SEND_EMAIL", str(int(OUTPUTS.get("send_email", True)))) == "1"
EMAIL_FROM = OUTPUTS.get("email_from", "xxxx")
EMAIL_TO = OUTPUTS.get("email_to", "xxxx")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "xxxx")
SMTP_HOST = OUTPUTS.get("smtp_host", "smtp.mail.me.com")
SMTP_PORT = OUTPUTS.get("smtp_port", 465)


# --------------------
# Helper: Send email
# --------------------
def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    # 1) Try SSL/465
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return
    except Exception as e:
        last_err = e

    # 2) Fallback STARTTLS/587
    with smtplib.SMTP(SMTP_HOST, 587, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.send_message(msg)


# --------------------
# Helper: Position state
# --------------------
def load_position_state():
    if not os.path.exists(POSITION_STATE_FILE):
        return {"position": 0}

    with open(POSITION_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_position_state(state):
    with open(POSITION_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


# --------------------
# Execution logic (STATEFUL - the right "dumb" decision layer)
# --------------------
def interpret_action(prev_pos, model_pos):
    """
    prev_pos: yesterday's actual position (0 or 1)
    model_pos: today's suggested position (0 or 1)
    
    CRITICAL: This is the only place position changes happen.
    It's stateful, deterministic, and matches what backtest replay should use.
    """
    if prev_pos == 0 and model_pos == 1:
        return "BUY"
    if prev_pos == 1 and model_pos == 0:
        return "SELL"
    if prev_pos == 1 and model_pos == 1:
        return "HOLD (Long)"
    if prev_pos == 0 and model_pos == 0:
        return "HOLD (Flat)"
    return "UNKNOWN"


# --------------------
# Robust persistence (TIMEZONE-AWARE)
# --------------------
def persist_live_signal(payload: dict, out_dir: str = LIVE_OUT_DIR) -> dict:
    """
    Write signal locally:
    - timestamped JSON snapshot
    - append-only JSONL audit log
    
    Uses timezone-aware timestamps (fixed).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Timezone-aware timestamp
    now_utc = datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%d_%H%M%S")
    
    json_path = Path(out_dir) / f"live_signals_{ts}.json"
    jsonl_path = Path(out_dir) / "live_signals.jsonl"

    enriched = dict(payload)
    enriched["_meta"] = {
        "written_utc": now_utc.isoformat(),  # Timezone-aware
        "host": socket.gethostname(),
        "script": "production.py",
    }

    json_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")

    return {"json_path": str(json_path), "jsonl_path": str(jsonl_path)}


# --------------------
# Email formatting
# --------------------
def format_message(sig, prev_pos, action, strengths):
    """
    Format email with clear signal interpretations.
    
    Args:
        sig: Signal dict
        prev_pos: Previous position
        action: Action to take
        strengths: Signal strength dict from get_all_signal_strengths()
    """
    
    # Emoji for visual clarity
    action_emoji = {
        "BUY": "🟢",
        "SELL": "🔴",
        "HOLD (Long)": "⏸️",
        "HOLD (Flat)": "⏸️",
    }.get(action, "")
    
    regime_names = {0: "Bull", 1: "Neutral", 2: "Defensive"}
    regime_name = regime_names.get(sig.get('regime'), 'Unknown')
    
    return f"""
{action_emoji} Gold Daily Execution Signal - {sig.get('date')}
==========================================

MODEL: FIVE_MODEL_GOLD (Live Trading)

ACTION TO TAKE:          {action}

Signal Interpretation:
  Classifier (Direction):  {strengths['classifier_signal']}
  Regressor (Opportunity): {strengths['regressor_signal']}
  Combined Signal:         {strengths['combined_signal']}

Position Status:
  Your Current Position:   {prev_pos} (1=Long, 0=Flat)
  Recommended Position:    {sig.get('model_position')}

Market Context:
  Regime:      {regime_name} (code: {sig.get('regime')})
  Gold Close:  ${sig.get('gold_close'):.2f}

Raw Metrics:
  Probability: {float(sig.get('prob', 0.0)):.1%} (classifier confidence)
  Rank:        {float(sig.get('rank', 0.0)):.1%} (regressor percentile)
  
------------------------------------------
Signal Guide:
  STRONG BUY/SELL  → High conviction
  BUY/SELL         → Moderate conviction
  WEAK BUY/SELL    → Low conviction
  HOLD             → Neutral / no opportunity

Execution Logic:
  BUY:  Open new long position
  SELL: Close existing long position
  HOLD: Maintain current position

Generated automatically by gold_engine.
"""


# --------------------
# CLI: manual position override
# --------------------
def handle_cli_override() -> bool:
    """
    Usage:
        python3 production.py long
        python3 production.py flat
        python3 production.py 1
        python3 production.py 0
    """
    if len(sys.argv) != 2:
        return False

    val = sys.argv[1].strip().lower()
    if val in ("long", "1"):
        save_position_state({"position": 1})
        print("Position set to LONG (1).")
        return True

    if val in ("flat", "0"):
        save_position_state({"position": 0})
        print("Position set to FLAT (0).")
        return True

    print("Usage: python3 production.py [long|flat|1|0]")
    raise SystemExit(1)


# --------------------
# Main daily run
# --------------------
def main():
    # Find latest available market date
    df_latest = build_gold_dataset().sort_index()
    latest_date = df_latest.index[-1]
    print(f"Using latest available market date: {latest_date.date()}")

    asof_dt = latest_date.to_pydatetime()

    # Load previous real-world position
    prev_state = load_position_state()
    prev_pos = int(prev_state.get("position", 0))

    # Generate signal with enhanced diagnostics
    sig = generate_signal(asof_dt, enhanced=True)
    
    model_pos = int(sig["model_position"])
    
    # Get human-readable signal interpretations
    # CRITICAL: This is PRESENTATION ONLY - does not change position
    strengths = get_all_signal_strengths(
        prob=float(sig["prob"]),
        rank=float(sig["rank"]),
        model_position=model_pos,
        regime=sig["regime"]
    )
    
    # Action decision based on state (the ONLY place positions change)
    action = interpret_action(prev_pos, model_pos)

    # Update real-world position (only if BUY/SELL)
    new_pos = prev_pos
    if action == "BUY":
        new_pos = 1
    elif action == "SELL":
        new_pos = 0

    save_position_state({"position": new_pos})

    # Single canonical output payload
    output_row = {
        "date": sig.get("date"),
        "prev_pos": prev_pos,
        "model_position": model_pos,
        "action": action,
        
        # Signal strengths (human-readable - PRESENTATION ONLY)
        "classifier_signal": strengths['classifier_signal'],
        "regressor_signal": strengths['regressor_signal'],
        "combined_signal": strengths['combined_signal'],
        
        # Raw metrics
        "rank": float(sig.get("rank", 0.0)),
        "prob": float(sig.get("prob", 0.0)),
        
        # Context
        "regime": sig.get("regime"),
        "gold_close": sig.get("gold_close"),
        "position_after": new_pos,
        
        # Enhanced diagnostics (if present)
        "rank_std": sig.get("rank_std"),
        "rank_lower": sig.get("rank_lower"),
        "rank_upper": sig.get("rank_upper"),
        "has_contradiction": sig.get("has_contradiction", False),
        "contradiction_type": sig.get("contradiction_type", "NONE"),
    }

    # Trust-building audit prints
    print("\n" + "="*60)
    print("LIVE SIGNAL AUDIT")
    print("="*60)
    print(f"Date:           {output_row['date']}")
    print(f"Gold Close:     ${output_row['gold_close']:.2f}")
    print(f"Regime:         {output_row['regime']}")
    print()
    print("Signal Interpretation:")
    print(f"  Classifier:   {output_row['classifier_signal']} (prob={output_row['prob']:.1%})")
    print(f"  Regressor:    {output_row['regressor_signal']} (rank={output_row['rank']:.1%})")
    print(f"  Combined:     {output_row['combined_signal']}")
    print()
    print("Position & Action:")
    print(f"  Previous:     {output_row['prev_pos']}")
    print(f"  Recommended:  {output_row['model_position']}")
    print(f"  Action:       {output_row['action']}")
    print(f"  After:        {output_row['position_after']}")
    
    if output_row.get('has_contradiction'):
        print()
        print(f"⚠️  Contradiction: {output_row['contradiction_type']}")
    
    if output_row.get('rank_lower') is not None:
        print()
        print("Rank Confidence:")
        print(f"  CI: [{output_row['rank_lower']:.1%}, {output_row['rank_upper']:.1%}]")
    
    print("="*60)

    # Always persist locally
    paths = persist_live_signal(output_row)
    print(f"\n[LIVE SIGNAL WRITTEN] {paths['json_path']}")

    # Save signal to CSV
    save_signal_to_csv(output_row)

    # Email (best-effort)
    email_body = format_message(sig, prev_pos, action, strengths)

    if SEND_EMAIL:
        try:
            send_email("Gold Trading Signal", email_body)
            print("Email sent successfully.")
        except Exception as e:
            print("WARNING: Email failed to send.")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            print("[EMAIL FAILED] Signal still written locally.")
    else:
        print("Email sending disabled (SEND_EMAIL=0).")


if __name__ == "__main__":
    if handle_cli_override():
        raise SystemExit(0)
    main()
