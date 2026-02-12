from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

try:
    import xgboost as xgb
except Exception:
    xgb = None


RND = 42


# -----------------------------
# Helper metrics
# -----------------------------
def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p))


def seat_winners_from_probs(df: pd.DataFrame, prob_col: str = "p_win") -> pd.DataFrame:
    idx = df.groupby("seat_id")[prob_col].idxmax()
    winners = df.loc[idx].copy()
    winners["pred_winner"] = 1
    return winners


def seat_accuracy(df: pd.DataFrame, prob_col: str = "p_win") -> float | None:
    if "won_seat" not in df.columns:
        return None

    pred_winners = seat_winners_from_probs(df, prob_col=prob_col)
    true_winners = df[df["won_seat"] == 1][["seat_id", "candidate_key"]].copy()

    merged = pred_winners.merge(true_winners, on="seat_id", how="left", suffixes=("_pred", "_true"))
    merged = merged.dropna(subset=["candidate_key_true"])  # drop seats with missing true winner
    return float((merged["candidate_key_pred"] == merged["candidate_key_true"]).mean())


def _summary(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)) if len(arr) else None,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


# -----------------------------
# Preprocessing
# -----------------------------
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


@dataclass
class CVConfig:
    target: str = "won_seat"
    outer_group: Literal["district_id", "seat_id"] = "district_id"
    outer_splits: int = 5
    inner_splits: int = 4
    random_state: int = RND


@dataclass
class FoldResult:
    model_name: str
    fold: int
    logloss: float
    auc: float | None
    row_acc: float
    seat_acc: float | None
    best_params: dict


def _fit_one_model(
    name: str,
    pipe: Pipeline,
    grid: dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    inner_splits: int,
) -> tuple[Pipeline, dict]:
    inner_cv = GroupKFold(n_splits=inner_splits)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=inner_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr, groups=groups_tr)
    return gs.best_estimator_, gs.best_params_


def run_groupkfold_cv_backtest(
    df_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    cfg: CVConfig,
) -> dict:
    """
    Outer GroupKFold (district or seat).
    Inner GroupKFold for hyperparam tuning.

    Returns a dict with:
      - per_fold_results (list)
      - summary_by_model (mean/std)
      - oof_predictions_by_winner (dict of winner_name -> oof df records)
      - winner_by_mean_logloss (model name)
      - winner_by_mean_seat_acc (model name or None)
      - winner_oof_overall_seat_acc (for logloss winner)
      - winner2_oof_overall_seat_acc (for seat-acc winner, if any)
    """

    required = {"seat_id", "district_id", "party_id", "candidate_key", cfg.target, cfg.outer_group}
    missing = required - set(df_train.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")

    df = df_train.copy()
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce").fillna(0).astype(int)

    X_all = df[numeric_features + categorical_features]
    y_all = df[cfg.target].values
    outer_groups = df[cfg.outer_group].values

    pre = build_preprocessor(numeric_features, categorical_features)

    # Model candidates
    candidates: list[tuple[str, Pipeline, dict]] = []

    # Logistic baseline
    logit = LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=cfg.random_state,
    )
    logit_pipe = Pipeline([("pre", pre), ("clf", logit)])
    logit_grid = {"clf__C": np.logspace(-3, 2, 12), "clf__penalty": ["l2"]}
    candidates.append(("logistic", logit_pipe, logit_grid))

    # HGB (fast non-linear)
    hgb = HistGradientBoostingClassifier(random_state=cfg.random_state)
    hgb_pipe = Pipeline([("pre", pre), ("clf", hgb)])
    hgb_grid = {
        "clf__max_depth": [3, 4],
        "clf__learning_rate": [0.03, 0.05],
        "clf__max_leaf_nodes": [31, 63],
    }
    candidates.append(("hgb", hgb_pipe, hgb_grid))

    # XGB optional
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.random_state,
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
        candidates.append(("xgboost", xgb_pipe, xgb_grid))

    # Outer CV
    outer_cv = GroupKFold(n_splits=cfg.outer_splits)

    fold_rows: list[FoldResult] = []
    oof = df[["seat_id", "district_id", "party_id", "candidate_key", cfg.target]].copy()
    for name, _, _ in candidates:
        oof[f"p_win_{name}"] = np.nan

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all, groups=outer_groups), start=1):
        df_tr = df.iloc[tr_idx].copy()
        df_te = df.iloc[te_idx].copy()

        X_tr = df_tr[numeric_features + categorical_features]
        y_tr = df_tr[cfg.target].values
        groups_tr = df_tr[cfg.outer_group].values

        X_te = df_te[numeric_features + categorical_features]
        y_te = df_te[cfg.target].values

        for name, pipe, grid in candidates:
            best_est, best_params = _fit_one_model(
                name=name,
                pipe=pipe,
                grid=grid,
                X_tr=X_tr,
                y_tr=y_tr,
                groups_tr=groups_tr,
                inner_splits=cfg.inner_splits,
            )

            p = best_est.predict_proba(X_te)[:, 1]
            pred = (p >= 0.5).astype(int)

            ll = float(log_loss(y_te, p, labels=[0, 1]))
            auc = _safe_auc(y_te, p)
            racc = float(accuracy_score(y_te, pred))

            temp = df_te.copy()
            temp["p_win"] = p
            sacc = seat_accuracy(temp, prob_col="p_win")

            fold_rows.append(
                FoldResult(
                    model_name=name,
                    fold=fold,
                    logloss=ll,
                    auc=auc,
                    row_acc=racc,
                    seat_acc=sacc,
                    best_params=best_params,
                )
            )

            # Save OOF probs for this model
            oof.loc[oof.index.isin(df_te.index), f"p_win_{name}"] = p

        print(f"[fold {fold}/{cfg.outer_splits}] done")

    # Summaries
    results_df = pd.DataFrame([r.__dict__ for r in fold_rows])

    summary_by_model: dict[str, dict] = {}
    for name in results_df["model_name"].unique():
        sub = results_df[results_df["model_name"] == name]
        summary_by_model[name] = {
            "logloss": _summary(sub["logloss"].tolist()),
            "auc": _summary([v for v in sub["auc"].tolist() if v is not None]),
            "row_acc": _summary(sub["row_acc"].tolist()),
            "seat_acc": _summary([v for v in sub["seat_acc"].tolist() if v is not None]),
        }

    # -----------------------------
    # Select up to two winners:
    #  - by mean logloss (min)
    #  - by mean seat_acc (max)
    # -----------------------------
    mean_ll = {m: summary_by_model[m]["logloss"]["mean"] for m in summary_by_model.keys()}
    winner_by_logloss = min(mean_ll, key=lambda k: mean_ll[k])

    mean_seat = {m: summary_by_model[m]["seat_acc"]["mean"] for m in summary_by_model.keys()}
    # seat_acc may be None for some/all models (if won_seat missing etc.)
    seat_candidates = {m: v for m, v in mean_seat.items() if v is not None}
    winner_by_seat_acc = max(seat_candidates, key=lambda k: seat_candidates[k]) if seat_candidates else None

    # Helper to build "seat-winner" style OOF eval df for a given model name
    def _build_oof_eval_for_model(model_name: str) -> tuple[pd.DataFrame, float | None]:
        col = f"p_win_{model_name}"
        oof_eval = oof.dropna(subset=[col]).copy()
        oof_eval["p_win"] = oof_eval[col]
        oof_eval["pred_winner"] = 0
        oof_eval.loc[oof_eval.groupby("seat_id")["p_win"].idxmax(), "pred_winner"] = 1
        overall_sacc = seat_accuracy(
            oof_eval.rename(columns={cfg.target: "won_seat"}), prob_col="p_win"
        )
        return oof_eval, overall_sacc

    # Build OOF eval outputs for winners (so you can inspect both)
    oof_predictions_by_winner: dict[str, list[dict]] = {}
    overall_seat_acc_by_winner: dict[str, float | None] = {}

    oof_eval_1, overall_sacc_1 = _build_oof_eval_for_model(winner_by_logloss)
    oof_predictions_by_winner[winner_by_logloss] = oof_eval_1.to_dict(orient="records")
    overall_seat_acc_by_winner[winner_by_logloss] = overall_sacc_1

    if winner_by_seat_acc is not None and winner_by_seat_acc != winner_by_logloss:
        oof_eval_2, overall_sacc_2 = _build_oof_eval_for_model(winner_by_seat_acc)
        oof_predictions_by_winner[winner_by_seat_acc] = oof_eval_2.to_dict(orient="records")
        overall_seat_acc_by_winner[winner_by_seat_acc] = overall_sacc_2

    out = {
        "config": cfg.__dict__,
        "per_fold_results": results_df.to_dict(orient="records"),
        "summary_by_model": summary_by_model,
        "winner_by_mean_logloss": winner_by_logloss,
        "winner_by_mean_seat_acc": winner_by_seat_acc,
        "overall_seat_acc_by_winner": overall_seat_acc_by_winner,
        # New: OOF predictions per winner (can be 1 or 2 keys)
        "oof_predictions_by_winner": oof_predictions_by_winner,
    }
    return out

