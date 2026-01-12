"""
research_data.py

Data loading and path utilities for the research layer.

Responsibilities:
- Resolve and validate paths
- Load backtest and walk-forward CSVs
- Auto-detect in-sample backtest file (for IS/OOS comparison)

This module is intentionally *pure*: it does not compute stats or draw charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ResearchData:
    """
    Container for all data and resolved paths used by the research layer.
    """
    backtest_df: pd.DataFrame
    walkforward_df: Optional[pd.DataFrame]
    backtest_path: Path
    walkforward_path: Optional[Path]
    insample_backtest_path: Optional[Path]


def _resolve_path(path: str | Path) -> Path:
    """
    Resolve a path relative to the project root.

    Rules:
    - Absolute paths are accepted as-is.
    - Relative paths are resolved against the directory containing the engine
      (the folder where this file lives).
    """
    p = Path(path)

    # If path is already absolute (e.g., /Users/xxx/file.csv)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"[research_data] File not found: {p}")
        return p

    # Otherwise resolve relative to the project directory
    project_root = Path(__file__).resolve().parent
    candidate = (project_root / p).resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"[research_data] File not found: {candidate}")

    return candidate


def load_backtest(path: str | Path) -> pd.DataFrame:
    """
    Load the main strategy backtest CSV.

    Requirements:
    - Must contain at least a 'date' column.
    """
    p = _resolve_path(path)
    df = pd.read_csv(p)

    if "date" not in df.columns:
        raise ValueError(
            f"[research_data] Backtest CSV at {p} must contain a 'date' column."
        )

    # Ensure datetime
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"[research_data] Failed to parse 'date' column in {p}: {e}")

    return df


def load_walkforward_summary(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    """
    Load the walk-forward summary CSV if a path is provided, otherwise return None.
    """
    if path is None:
        return None

    p = _resolve_path(path)
    df = pd.read_csv(p)

    # Optional: normalise columns, parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def find_insample_file(
    search_root: str | Path,
    filename: str = "gold_backtest_results_insample.csv",
) -> Optional[Path]:
    """
    Attempt to locate an in-sample backtest file for IS/OOS comparison.

    Strategy:
    - Look for 'filename' in:
        1. The same directory as the main backtest file (search_root)
        2. One level above, in case of organised subfolders
        3. Any direct children of search_root matching the filename

    Returns:
        Path to the first match found, or None if not found.
    """
    root = Path(search_root).expanduser().resolve()
    candidates: list[Path] = []

    # 1) Directly in root
    candidates.append(root / filename)

    # 2) One level above
    candidates.append(root.parent / filename)

    # 3) Immediate children of root
    for child in root.iterdir():
        if child.is_dir():
            candidates.append(child / filename)

    for c in candidates:
        if c.exists():
            return c

    return None


def load_research_data(
    backtest_path: str | Path,
    walkforward_summary_path: Optional[str | Path] = None,
) -> ResearchData:
    """
    High-level convenience function to construct a ResearchData object
    from the provided CSV paths.
    """
    backtest_path_resolved = _resolve_path(backtest_path)

    if walkforward_summary_path is not None:
        walkforward_path_resolved: Optional[Path] = _resolve_path(
            walkforward_summary_path
        )
    else:
        walkforward_path_resolved = None

    backtest_df = load_backtest(backtest_path_resolved)
    walkforward_df = load_walkforward_summary(walkforward_path_resolved)

    insample_path = find_insample_file(backtest_path_resolved.parent)

    return ResearchData(
        backtest_df=backtest_df,
        walkforward_df=walkforward_df,
        backtest_path=backtest_path_resolved,
        walkforward_path=walkforward_path_resolved,
        insample_backtest_path=insample_path,
    )

def load_data_cleaning_summary():
    import duckdb
    from config import DB_PATH

    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute("""
            SELECT *
            FROM data_cleaning_summary
            ORDER BY run_timestamp DESC
        """).fetchdf()
    except Exception:
        df = None
    finally:
        con.close()

    return df
