from __future__ import annotations
import pandas as pd

from src.config.constants import BASE_FEATURES, LAG_COL
from src.features.transforms import add_derived_features
from src.features.normalization import normalize_within_district
from src.utils.validate import require_columns

def predict_pr_2082(df_infer_raw: pd.DataFrame, model: object) -> pd.DataFrame:
    df = add_derived_features(df_infer_raw)
    require_columns(df, ["district_id", "party_id", LAG_COL] + BASE_FEATURES, name="pr_infer")

    X = df[BASE_FEATURES]
    pred_change = model.predict(X)

    out = df.copy()
    out["pred_change"] = pred_change
    out["pred_vote_share_raw"] = (out[LAG_COL] + out["pred_change"]).clip(0, 100)

    # Constraint layer
    out = normalize_within_district(out, pred_col="pred_vote_share_raw", out_col="pred_vote_share_norm")
    return out
