from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Helpers functions for prediction 

def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def normalize_vote_share_within_seat(df: pd.DataFrame, seat_col: str, pred_col: str) -> pd.Series:
    raw = df[pred_col].astype(float).copy()
    raw = raw.clip(lower=0.0)

    seat_sum = raw.groupby(df[seat_col]).transform("sum")
    out = raw / seat_sum.replace(0.0, np.nan)

    is_nan = out.isna()
    if is_nan.any():
        counts = df.groupby(seat_col)[seat_col].transform("count").astype(float)
        out[is_nan] = 1.0 / counts[is_nan]

    return out


def seat_winner_from_share(df: pd.DataFrame, seat_col: str, share_col: str) -> pd.Series:
    max_in_seat = df.groupby(seat_col)[share_col].transform("max")
    return (df[share_col] == max_in_seat).astype(int)


# Load models

@dataclass(frozen=True)
class PredictVoteShareConfig:
    artifacts_dir: Path
    artifact_prefix: str = "fptp_vote_share"


def load_manifest(cfg: PredictVoteShareConfig) -> dict[str, Any]:
    path = cfg.artifacts_dir / f"{cfg.artifact_prefix}_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Train models first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(cfg: PredictVoteShareConfig) -> dict[str, Any]:
    import joblib

    manifest = load_manifest(cfg)
    model_paths = manifest.get("model_paths", {})
    if not isinstance(model_paths, dict) or not model_paths:
        raise ValueError("Manifest has no model_paths. Re-train the winners.")

    models: dict[str, Any] = {}
    for model_name, p in model_paths.items():
        models[model_name] = joblib.load(p)
    return models


# Predict

def predict_vote_share_and_append_columns(
    df_infer: pd.DataFrame,
    features: list[str],
    seat_col: str,
    models: dict[str, Any],
) -> pd.DataFrame:
    """
    Returns df_infer with prediction columns appended at the end:
      - <model>_pred_raw
      - <model>_pred_share (within-seat normalized)
      - <model>_pred_winner (argmax within seat)
    """
    _require_cols(df_infer, [seat_col], "infer_df")
    _require_cols(df_infer, features, "infer_df")

    out = df_infer.copy()
    Xp = out[features]

    # track newly-added columns to ensure appended at end
    new_cols: list[str] = []

    for name, model in models.items():
        raw_col = f"{name}_pred_raw"
        share_col = f"{name}_pred_share"
        win_col = f"{name}_pred_winner"

        out[raw_col] = model.predict(Xp)
        out[share_col] = normalize_vote_share_within_seat(out, seat_col, raw_col)
        out[win_col] = seat_winner_from_share(out, seat_col, share_col)

        new_cols.extend([raw_col, share_col, win_col])

    # Move prediction columns to the end 
    orig_cols = [c for c in df_infer.columns]
    final_cols = orig_cols + [c for c in new_cols if c not in orig_cols]
    out = out[final_cols]

    return out
