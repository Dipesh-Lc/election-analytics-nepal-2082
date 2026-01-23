from __future__ import annotations
import pandas as pd

def require_columns(df: pd.DataFrame, cols: list[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def require_no_nulls(df: pd.DataFrame, cols: list[str], name: str = "df") -> None:
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise ValueError(f"{name} has nulls in columns: {bad}")
