from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except Exception:
    xgb = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class VoteShareBacktestConfig:
    target_col: str = "vote_share"
    group_col: str = "seat_id"          # seat-holdout
    outer_splits: int = 5
    inner_splits: int = 4
    random_state: int = 42
    use_xgboost: bool = True


# -----------------------------
# Helpers
# -----------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def normalize_vote_share_within_seat(df: pd.DataFrame, seat_col: str, pred_col: str) -> pd.Series:
    """
    Convert raw regression outputs into valid vote shares within each seat:
      - clip negatives to 0
      - normalize so sum(pred_share)=1 per seat
      - if seat sum=0 (all clipped), fall back to uniform distribution
    """
    raw = df[pred_col].astype(float).copy()
    raw = raw.clip(lower=0.0)

    seat_sum = raw.groupby(df[seat_col]).transform("sum")
    out = raw / seat_sum.replace(0.0, np.nan)

    # fallback uniform for seats where everything became 0
    is_nan = out.isna()
    if is_nan.any():
        # uniform within each affected seat
        counts = df.groupby(seat_col)[seat_col].transform("count").astype(float)
        out[is_nan] = 1.0 / counts[is_nan]

    return out


def seat_winner_from_share(df: pd.DataFrame, seat_col: str, share_col: str) -> pd.Series:
    """
    Winner label from within-seat max predicted share.
    Returns 1 for the winner row in each seat else 0.
    """
    max_in_seat = df.groupby(seat_col)[share_col].transform("max")
    return (df[share_col] == max_in_seat).astype(int)


def seat_accuracy(df: pd.DataFrame, seat_col: str, y_true_win: str, y_pred_share: str) -> float:
    """
    Seat-level accuracy:
      - compute predicted winner (argmax of y_pred_share within seat)
      - compare to true winner (y_true_win)
    Assumes exactly one true winner row per seat.
    """
    pred_win = seat_winner_from_share(df, seat_col, y_pred_share)

    # for each seat, did we pick the true winner?
    # true winner row has y_true_win=1, predicted winner row has pred_win=1
    ok = []
    for seat_id, g in df.assign(pred_win=pred_win).groupby(seat_col):
        true_idx = g.index[g[y_true_win] == 1]
        pred_idx = g.index[g["pred_win"] == 1]
        if len(true_idx) != 1 or len(pred_idx) != 1:
            # if data is messy, treat as incorrect
            ok.append(0)
        else:
            ok.append(int(true_idx[0] == pred_idx[0]))
    return float(np.mean(ok)) if ok else 0.0


# -----------------------------
# Main backtest
# -----------------------------
def run_vote_share_seat_holdout_backtest(
    df: pd.DataFrame,
    features: list[str],
    config: VoteShareBacktestConfig = VoteShareBacktestConfig(),
) -> dict[str, Any]:
    """
    Seat-holdout (GroupKFold by seat_id) vote_share regression backtest.
    Returns a dict with per-fold results + summary + OOF predictions.
    """
    _require_cols(df, [config.target_col, config.group_col, "won_seat"], "train_df")
    _require_cols(df, features, "train_df")

    work = df.copy()
    y = work[config.target_col].astype(float).values
    groups = work[config.group_col].astype(str).values
    X = work[features]

    outer = GroupKFold(n_splits=config.outer_splits)
    inner = GroupKFold(n_splits=config.inner_splits)

    # store out-of-fold raw preds per model (optional) + winner selection
    oof_rows = []

    per_fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr = groups[tr_idx]

        # -----------------
        # Candidate models
        # -----------------
        models: list[tuple[str, Any, dict[str, list[Any]]]] = []

        # Ridge
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(random_state=config.random_state)),
        ])
        ridge_grid = {"reg__alpha": list(np.logspace(-3, 3, 20))}
        models.append(("ridge", ridge, ridge_grid))

        # ElasticNet
        enet = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(max_iter=50_000, random_state=config.random_state)),
        ])
        enet_grid = {
            "reg__alpha": list(np.logspace(-3, 2, 12)),
            "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        models.append(("elasticnet", enet, enet_grid))

        # HGB (no scaling needed)
        hgb_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            random_state=config.random_state,
        )
        hgb_grid = {
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [3, 4],
            "max_leaf_nodes": [31, 63],
        }
        models.append(("hgb", hgb_model, hgb_grid))

        # XGBoost (optional)
        if config.use_xgboost and xgb is not None:
            xgbr = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=800,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=config.random_state,
            )
            xgb_grid = {
                "max_depth": [3, 4],
                "min_child_weight": [3, 5],
                "subsample": [0.7, 0.8],
                "colsample_bytree": [0.7, 0.8],
                "reg_alpha": [0.5, 1.0],
                "reg_lambda": [1.0, 2.0],
            }
            models.append(("xgboost", xgbr, xgb_grid))

        # -----------------
        # Fit + evaluate each model on this outer fold
        # -----------------
        fold_candidates = []

        for name, model, grid in models:
            gs = GridSearchCV(
                estimator=model,
                param_grid=grid,
                scoring="neg_mean_absolute_error",
                cv=inner.split(X_tr, y_tr, groups=g_tr),
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_tr, y_tr)

            best = gs.best_estimator_
            pred_raw = best.predict(X_te)

            fold_df = work.iloc[te_idx].copy()
            fold_df[f"{name}_pred_raw"] = pred_raw
            fold_df[f"{name}_pred_share"] = normalize_vote_share_within_seat(
                fold_df, config.group_col, f"{name}_pred_raw"
            )

            mae = float(mean_absolute_error(fold_df[config.target_col], fold_df[f"{name}_pred_share"]))
            rmse = _rmse(fold_df[config.target_col].values, fold_df[f"{name}_pred_share"].values)
            sacc = seat_accuracy(fold_df, config.group_col, "won_seat", f"{name}_pred_share")

            fold_candidates.append({
                "model_name": name,
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "seat_acc": sacc,
                "best_params": gs.best_params_,
            })

            # store oof predictions for later analysis
            oof_part = fold_df[[
                "state_id", "district_id", "const_id", "seat_id",
                "party_id", "candidate_name", config.target_col, "won_seat"
            ]].copy()
            oof_part["fold"] = fold
            oof_part["model_name"] = name
            oof_part["pred_share"] = fold_df[f"{name}_pred_share"].values
            oof_rows.append(oof_part)

        per_fold_results.extend(fold_candidates)

    per_fold = pd.DataFrame(per_fold_results)

    # Summary
    summary = (
        per_fold.groupby("model_name")[["mae", "rmse", "seat_acc"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Pick winner: prioritize MAE (vote share), break ties with seat_acc
    means = per_fold.groupby("model_name")[["mae", "rmse", "seat_acc"]].mean()

    err_col = "mae"

    # Winner 1: best by error
    winner_by_error = (
        means.sort_values([err_col, "seat_acc"], ascending=[True, False])
        .head(1)
    )
    winner_by_error_name = str(winner_by_error.index[0])

    # Winner 2: best by seat accuracy
    winner_by_seat = (
        means.sort_values(["seat_acc", err_col], ascending=[False, True])
        .head(1)
    )
    winner_by_seat_name = str(winner_by_seat.index[0])

    winners = [
        {"type": f"min_{err_col}", "model_name": winner_by_error_name},
    ]
    if winner_by_seat_name != winner_by_error_name:
        winners.append({"type": "max_seat_acc", "model_name": winner_by_seat_name})

    # Attach mean metrics for each winner
    for w in winners:
        m = w["model_name"]
        w["mean_metrics"] = means.loc[m].to_dict()

    oof = pd.concat(oof_rows, ignore_index=True)

    return {
        "config": asdict(config),
        "features": features,
        "per_fold_results": per_fold_results,
        "summary_by_model": json.loads(summary.to_json(orient="records")),

        # NEW: winners list (1 or 2 items)
        "winners": winners,
        "winner_by_error": winner_by_error_name,
        "winner_by_seat_acc": (winner_by_seat_name if winner_by_seat_name != winner_by_error_name else None),

        "oof_predictions": oof,  # dataframe
    }

