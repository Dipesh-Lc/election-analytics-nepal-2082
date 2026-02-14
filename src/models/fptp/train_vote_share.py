from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import xgboost as xgb
except Exception:
    xgb = None

RND = 42


# Config

@dataclass(frozen=True)
class TrainVoteShareConfig:
    target_col: str = "vote_share"
    group_col: Literal["seat_id", "district_id"] = "seat_id"
    inner_splits: int = 4
    random_state: int = RND
    use_xgboost: bool = True


# Helpers

def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_vote_share_winners(metrics_json_path: Path) -> list[str]:
    """
    Expects updated backtest JSON containing:
      - winners: [{type, model_name, mean_metrics}, ...]
      OR fallback:
      - winner_by_error / winner_by_seat_acc
    Returns 1 or 2 unique names.
    """
    obj = json.loads(metrics_json_path.read_text(encoding="utf-8"))

    winners: list[str] = []
    if isinstance(obj, dict) and "winners" in obj and isinstance(obj["winners"], list):
        for w in obj["winners"]:
            if isinstance(w, dict) and isinstance(w.get("model_name"), str):
                winners.append(w["model_name"])
    else:
        w1 = obj.get("winner_by_error")
        w2 = obj.get("winner_by_seat_acc")
        if isinstance(w1, str) and w1:
            winners.append(w1)
        if isinstance(w2, str) and w2 and w2 not in winners:
            winners.append(w2)

    # de-dupe keep order
    out = []
    for w in winners:
        if w not in out:
            out.append(w)
    return out


def load_train_cfg_from_backtest(metrics_json_path: Path) -> TrainVoteShareConfig:
    obj = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}

    return TrainVoteShareConfig(
        target_col=cfg.get("target_col", "vote_share"),
        group_col=cfg.get("group_col", "seat_id"),
        inner_splits=int(cfg.get("inner_splits", 4)),
        random_state=int(cfg.get("random_state", RND)),
        use_xgboost=bool(cfg.get("use_xgboost", True)),
    )


# Candidate builders

def build_model_and_grid(
    model_name: str,
    cfg: TrainVoteShareConfig,
) -> tuple[Any, dict[str, list[Any]]]:
    if model_name == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(random_state=cfg.random_state)),
        ])
        grid = {"reg__alpha": list(np.logspace(-3, 3, 20))}
        return model, grid

    if model_name == "elasticnet":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(max_iter=50_000, random_state=cfg.random_state)),
        ])
        grid = {
            "reg__alpha": list(np.logspace(-3, 2, 12)),
            "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        return model, grid

    if model_name == "hgb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            random_state=cfg.random_state,
        )
        grid = {
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [3, 4],
            "max_leaf_nodes": [31, 63],
        }
        return model, grid

    if model_name == "xgboost":
        if not cfg.use_xgboost or xgb is None:
            raise ValueError("xgboost model requested but xgboost is unavailable or use_xgboost=False.")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=cfg.random_state,
        )
        grid = {
            "max_depth": [3, 4],
            "min_child_weight": [3, 5],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "reg_alpha": [0.5, 1.0],
            "reg_lambda": [1.0, 2.0],
        }
        return model, grid

    raise ValueError(f"Unknown model_name '{model_name}'. Expected ridge/elasticnet/hgb/xgboost.")



# Fit + save

def fit_winner_on_full_data(
    df_train: pd.DataFrame,
    features: list[str],
    cfg: TrainVoteShareConfig,
    model_name: str,
) -> tuple[Any, dict[str, Any]]:
    _require_cols(df_train, [cfg.target_col, cfg.group_col], "train_df")
    _require_cols(df_train, features, "train_df")

    work = df_train.copy()
    y = work[cfg.target_col].astype(float).values
    groups = work[cfg.group_col].astype(str).values
    X = work[features]

    model, grid = build_model_and_grid(model_name, cfg)

    inner = GroupKFold(n_splits=cfg.inner_splits)
    gs = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=inner.split(X, y, groups=groups),
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def train_and_save_vote_share_winners(
    df_train: pd.DataFrame,
    features: list[str],
    cfg: TrainVoteShareConfig,
    winners: list[str],
    artifacts_dir: Path,
    artifact_prefix: str = "fptp_vote_share",
) -> dict[str, Any]:
    import joblib

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_paths: dict[str, str] = {}
    best_params: dict[str, Any] = {}

    for name in winners:
        est, params = fit_winner_on_full_data(df_train, features, cfg, name)

        stem = f"{artifact_prefix}_full_{cfg.group_col}_{name}"
        path = artifacts_dir / f"{stem}.joblib"
        joblib.dump(est, path)

        model_paths[name] = str(path)
        best_params[name] = params

    params_path = artifacts_dir / f"{artifact_prefix}_fullfit_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    manifest = {
        "artifact_prefix": artifact_prefix,
        "group_col": cfg.group_col,
        "target_col": cfg.target_col,
        "inner_splits": cfg.inner_splits,
        "random_state": cfg.random_state,
        "features": features,
        "winners": winners,
        "model_paths": model_paths,
    }
    manifest_path = artifacts_dir / f"{artifact_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "winners": winners,
        "model_paths": model_paths,
        "best_params_path": str(params_path),
        "manifest_path": str(manifest_path),
    }
