from __future__ import annotations

import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except Exception:
    xgb = None

from src.config.constants import BASE_FEATURES, TARGET_COL, GROUP_COL, LAG_COL
from src.features.transforms import add_derived_features
from src.utils.validate import require_columns

RND = 42

@dataclass
class CandidateResult:
    model_name: str
    mae_change: float
    rmse_change: float
    vote_share_mae: float | None
    best_params: dict

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def district_holdout_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RND):
    X = df[BASE_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx

def _evaluate_on_holdout(df_all: pd.DataFrame, test_idx: np.ndarray, pred_change: np.ndarray) -> float | None:
    """
    Compute MAE on vote_share (interpretable) if vote_share exists.
    """
    test_df = df_all.iloc[test_idx].copy()
    test_df["pred_change"] = pred_change
    test_df["pred_vote_share"] = (test_df[LAG_COL] + test_df["pred_change"]).clip(0, 100)
    if "vote_share" in test_df.columns:
        return float(mean_absolute_error(test_df["vote_share"], test_df["pred_vote_share"]))
    return None

def run_district_holdout_backtest(
    df_train_raw: pd.DataFrame,
    test_size: float = 0.2,
    inner_splits: int = 5,
    random_state: int = RND,
) -> tuple[object, str, dict, pd.DataFrame]:
    """
    Returns:
      best_model (fitted on TRAIN PART only),
      best_name,
      metrics dict,
      heldout_predictions dataframe (rows of heldout set with preds)
    """
    df = add_derived_features(df_train_raw)

    require_columns(df, [TARGET_COL, GROUP_COL, LAG_COL, "party_id"], name="pr_train")
    require_columns(df, BASE_FEATURES, name="pr_train")

    X = df[BASE_FEATURES]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    train_idx, test_idx = district_holdout_split(df, test_size=test_size, random_state=random_state)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    gkf = GroupKFold(n_splits=inner_splits)

    
    # Core: Ridge
    
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=random_state)),
    ])
    ridge_grid = {"model__alpha": np.logspace(-3, 3, 20)}
    ridge_gs = GridSearchCV(
        ridge_pipe,
        ridge_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    ridge_gs.fit(X_train, y_train, groups=groups_train)
    ridge_pred = ridge_gs.best_estimator_.predict(X_test)

    ridge_res = CandidateResult(
        model_name="ridge",
        mae_change=float(mean_absolute_error(y_test, ridge_pred)),
        rmse_change=_rmse(y_test, ridge_pred),
        vote_share_mae=_evaluate_on_holdout(df, test_idx, ridge_pred),
        best_params=ridge_gs.best_params_
    )

    
    # Challenger: ElasticNet
    
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=20_000, random_state=random_state)),
    ])
    enet_grid = {
        "model__alpha": np.logspace(-3, 2, 10),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    enet_gs = GridSearchCV(
        enet_pipe,
        enet_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    enet_gs.fit(X_train, y_train, groups=groups_train)
    enet_pred = enet_gs.best_estimator_.predict(X_test)

    enet_res = CandidateResult(
        model_name="elasticnet",
        mae_change=float(mean_absolute_error(y_test, enet_pred)),
        rmse_change=_rmse(y_test, enet_pred),
        vote_share_mae=_evaluate_on_holdout(df, test_idx, enet_pred),
        best_params=enet_gs.best_params_
    )

    
    # Challenger: Shallow XGBoost (optional)
    
    xgb_res = None
    xgb_best = None
    if xgb is not None:
        # Shallow, regularized defaults
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
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

        # tiny grid to avoid overfitting / huge runs
        xgb_grid = {
            "max_depth": [3, 4],
            "min_child_weight": [3, 5],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "reg_alpha": [0.5, 1.0],
            "reg_lambda": [1.0, 2.0],
        }

        # Manual inner CV (because GridSearchCV doesn't pass groups to XGB well across versions)
        # I will do a small randomized-like search by iterating combinations.
        best_mae = float("inf")
        best_params = None
        best_model = None

        # build list of param combos
        from itertools import product
        keys = list(xgb_grid.keys())
        combos = list(product(*[xgb_grid[k] for k in keys]))

        for vals in combos:
            params = dict(zip(keys, vals))
            fold_maes = []
            for tr_i, va_i in gkf.split(X_train, y_train, groups_train):
                m = xgb.XGBRegressor(**{**xgb_model.get_params(), **params})
                m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i], verbose=False)
                p = m.predict(X_train.iloc[va_i])
                fold_maes.append(mean_absolute_error(y_train.iloc[va_i], p))
            avg_mae = float(np.mean(fold_maes))
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = params
                best_model = xgb.XGBRegressor(**{**xgb_model.get_params(), **params})

        best_model.fit(X_train, y_train, verbose=False)
        xgb_pred = best_model.predict(X_test)
        xgb_best = best_model

        xgb_res = CandidateResult(
            model_name="xgboost",
            mae_change=float(mean_absolute_error(y_test, xgb_pred)),
            rmse_change=_rmse(y_test, xgb_pred),
            vote_share_mae=_evaluate_on_holdout(df, test_idx, xgb_pred),
            best_params=best_params or {}
        )

    # Select winner by MAE on change (primary)
    candidates = [ridge_res, enet_res] + ([xgb_res] if xgb_res else [])
    winner = min(candidates, key=lambda r: r.mae_change)

    # Get the winner estimator (already fit on X_train)
    if winner.model_name == "ridge":
        best_estimator = ridge_gs.best_estimator_
    elif winner.model_name == "elasticnet":
        best_estimator = enet_gs.best_estimator_
    else:
        best_estimator = xgb_best

    # Build heldout rows with predictions from best estimator
    heldout = df.iloc[test_idx].copy()
    heldout["pred_change"] = best_estimator.predict(X_test)
    heldout["pred_vote_share"] = (heldout[LAG_COL] + heldout["pred_change"]).clip(0, 100)

    metrics = {
        "outer_test_size": test_size,
        "inner_splits": inner_splits,
        "ridge": ridge_res.__dict__,
        "elasticnet": enet_res.__dict__,
        "xgboost": (xgb_res.__dict__ if xgb_res else None),
        "winner": winner.model_name,
    }

    return best_estimator, winner.model_name, metrics, heldout
