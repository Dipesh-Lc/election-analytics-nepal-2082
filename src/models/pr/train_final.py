from __future__ import annotations
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet

try:
    import xgboost as xgb
except Exception:
    xgb = None

from src.config.constants import BASE_FEATURES, TARGET_COL
from src.features.transforms import add_derived_features
from src.utils.validate import require_columns

def train_final_model(df_train_raw: pd.DataFrame, model_name: str, best_params: dict) -> object:
    df = add_derived_features(df_train_raw)
    require_columns(df, BASE_FEATURES + [TARGET_COL], name="pr_train")

    X = df[BASE_FEATURES]
    y = df[TARGET_COL]

    if model_name == "ridge":
        alpha = best_params.get("model__alpha", 1.0)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=42)),
        ])
    elif model_name == "elasticnet":
        alpha = best_params.get("model__alpha", 0.1)
        l1 = best_params.get("model__l1_ratio", 0.5)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20_000, random_state=42)),
        ])
    elif model_name == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost not installed, cannot train xgboost final model.")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=600,
            learning_rate=0.03,
            max_depth=best_params.get("max_depth", 4),
            min_child_weight=best_params.get("min_child_weight", 5),
            subsample=best_params.get("subsample", 0.8),
            colsample_bytree=best_params.get("colsample_bytree", 0.8),
            reg_alpha=best_params.get("reg_alpha", 1.0),
            reg_lambda=best_params.get("reg_lambda", 1.0),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X, y)
    return model
