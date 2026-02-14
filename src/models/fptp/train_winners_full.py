from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except Exception:
    xgb = None


RND = 42


# Preprocessing

def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )


# Model candidates 

def build_candidates(
    pre: ColumnTransformer,
    random_state: int = RND,
) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    candidates: dict[str, tuple[Pipeline, dict[str, Any]]] = {}

    # Logistic
    logit = LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=random_state,
    )
    logit_pipe = Pipeline([("pre", pre), ("clf", logit)])
    logit_grid = {"clf__C": np.logspace(-3, 2, 12), "clf__penalty": ["l2"]}
    candidates["logistic"] = (logit_pipe, logit_grid)

    # HGB
    hgb = HistGradientBoostingClassifier(random_state=random_state)
    hgb_pipe = Pipeline([("pre", pre), ("clf", hgb)])
    hgb_grid = {
        "clf__max_depth": [3, 4],
        "clf__learning_rate": [0.03, 0.05],
        "clf__max_leaf_nodes": [31, 63],
    }
    candidates["hgb"] = (hgb_pipe, hgb_grid)

    # XGB 
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
        )
        xgb_pipe = Pipeline([("pre", pre), ("clf", xgb_clf)])
        xgb_grid = {
            "clf__max_depth": [3, 4],
            "clf__min_child_weight": [3, 5],
            "clf__subsample": [0.7, 0.8],
            "clf__colsample_bytree": [0.7, 0.8],
            "clf__reg_alpha": [0.5, 1.0],
            "clf__reg_lambda": [1.0, 2.0],
        }
        candidates["xgboost"] = (xgb_pipe, xgb_grid)

    return candidates


# Training

@dataclass(frozen=True)
class TrainWinnersConfig:
    target: str = "won_seat"
    group_col: Literal["district_id", "seat_id"] = "seat_id"
    inner_splits: int = 4
    random_state: int = RND


def _require_cols(df: pd.DataFrame, cols: set[str], label: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def load_winners_from_metrics(metrics_json_path: Path) -> list[str]:
    """
    Reads winners created by your updated CV backtest JSON:
      - winner_by_mean_logloss
      - winner_by_mean_seat_acc (optional)
    Returns 1 or 2 unique model names.
    """
    obj = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    w1 = obj.get("winner_by_mean_logloss")
    w2 = obj.get("winner_by_mean_seat_acc")

    winners: list[str] = []
    if isinstance(w1, str) and w1:
        winners.append(w1)
    if isinstance(w2, str) and w2 and w2 not in winners:
        winners.append(w2)
    return winners


def load_train_cfg_from_metrics(metrics_json_path: Path) -> TrainWinnersConfig:
    """
    Pulls config from the CV JSON (so you don't accidentally change group_col/target).
    Falls back to defaults if missing.
    """
    obj = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}

    return TrainWinnersConfig(
        target=cfg.get("target", "won_seat"),
        group_col=cfg.get("outer_group", "seat_id"),
        inner_splits=int(cfg.get("inner_splits", 4)),
        random_state=int(cfg.get("random_state", RND)),
    )


def fit_winner_on_full_data(
    df_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    cfg: TrainWinnersConfig,
    model_name: str,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Full-data refit using GroupKFold GridSearchCV (scoring = neg_log_loss),
    returning the best estimator and best params.
    """
    required = {"seat_id", "district_id", "party_id", "candidate_key", cfg.target, cfg.group_col}
    _require_cols(df_train, required, "Training data")

    df = df_train.copy()
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce").fillna(0).astype(int)

    X = df[numeric_features + categorical_features]
    y = df[cfg.target].values
    groups = df[cfg.group_col].values

    pre = build_preprocessor(numeric_features, categorical_features)
    candidates = build_candidates(pre=pre, random_state=cfg.random_state)

    if model_name not in candidates:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(candidates.keys())}")

    pipe, grid = candidates[model_name]

    inner_cv = GroupKFold(n_splits=cfg.inner_splits)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=inner_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def build_artifact_name(prefix: str, model_name: str, group_col: str) -> str:
    return f"{prefix}_full_{group_col}_{model_name}"


def train_and_save_winners(
    df_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    winners: list[str],
    cfg: TrainWinnersConfig,
    artifacts_dir: Path,
    artifact_prefix: str = "fptp",
) -> dict[str, Any]:
    """
    Trains each winner on full training data and saves:
      - model/artifacts/<prefix>_full_<group>_<model>.joblib
      - model/artifacts/<prefix>_fullfit_best_params.json
      - model/artifacts/<prefix>_winners_manifest.json  (what was trained)
    """
    import joblib

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trained_paths: dict[str, str] = {}
    best_params_by_model: dict[str, Any] = {}

    for model_name in winners:
        est, best_params = fit_winner_on_full_data(
            df_train=df_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            cfg=cfg,
            model_name=model_name,
        )

        stem = build_artifact_name(artifact_prefix, model_name, cfg.group_col)
        model_path = artifacts_dir / f"{stem}.joblib"
        joblib.dump(est, model_path)

        trained_paths[model_name] = str(model_path)
        best_params_by_model[model_name] = best_params

    # Save best params
    params_path = artifacts_dir / f"{artifact_prefix}_fullfit_best_params.json"
    params_path.write_text(json.dumps(best_params_by_model, indent=2), encoding="utf-8")

    # Save manifest (helps prediction code locate models + know winners)
    manifest = {
        "artifact_prefix": artifact_prefix,
        "group_col": cfg.group_col,
        "target": cfg.target,
        "inner_splits": cfg.inner_splits,
        "random_state": cfg.random_state,
        "winners": winners,
        "model_paths": trained_paths,
    }
    manifest_path = artifacts_dir / f"{artifact_prefix}_winners_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "winners": winners,
        "model_paths": trained_paths,
        "best_params_path": str(params_path),
        "manifest_path": str(manifest_path),
    }
