from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictConfig:
    artifacts_dir: Path
    artifact_prefix: str = "fptp"


def _require_cols(df: pd.DataFrame, cols: set[str], label: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def load_manifest(cfg: PredictConfig) -> dict[str, Any]:
    path = cfg.artifacts_dir / f"{cfg.artifact_prefix}_winners_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {path}. Run the training script first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_models_from_manifest(cfg: PredictConfig) -> dict[str, Any]:
    import joblib

    manifest = load_manifest(cfg)
    model_paths = manifest.get("model_paths", {})
    if not isinstance(model_paths, dict) or not model_paths:
        raise ValueError("Manifest has no model_paths. Re-train the winners.")

    models: dict[str, Any] = {}
    for model_name, model_path in model_paths.items():
        models[model_name] = joblib.load(model_path)
    return models


def predict_candidates_and_seat_winners(
    df_pred: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    models: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - candidate_pred_df: candidate-level rows with p_win_<model> and pred_winner_<model>
      - seat_winners_df: one row per seat per model with predicted winning candidate and p_win
    """
    required = {"seat_id", "district_id", "party_id", "candidate_key"}
    _require_cols(df_pred, required, "Prediction data")

    # Candidate-level output
    cand_out = df_pred[["seat_id", "district_id", "party_id", "candidate_key"]].copy()
    Xp = df_pred[numeric_features + categorical_features]

    for name, model in models.items():
        p = model.predict_proba(Xp)[:, 1]
        cand_out[f"p_win_{name}"] = p

        cand_out[f"pred_winner_{name}"] = 0
        idx = cand_out.groupby("seat_id")[f"p_win_{name}"].idxmax()
        cand_out.loc[idx, f"pred_winner_{name}"] = 1

    # Seat-level winners output
    seat_rows = []
    for name in models.keys():
        sub = cand_out[cand_out[f"pred_winner_{name}"] == 1].copy()
        sub = sub[["seat_id", "district_id", "party_id", "candidate_key", f"p_win_{name}"]]
        sub = sub.rename(columns={f"p_win_{name}": "p_win"})
        sub["model"] = name
        seat_rows.append(sub)

    seat_out = pd.concat(seat_rows, ignore_index=True) if seat_rows else pd.DataFrame()

    return cand_out, seat_out
