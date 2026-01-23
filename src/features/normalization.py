from __future__ import annotations
import pandas as pd

def normalize_within_district(
    df: pd.DataFrame,
    district_col: str = "district_id",
    pred_col: str = "pred_vote_share_raw",
    out_col: str = "pred_vote_share_norm",
) -> pd.DataFrame:
    """
    Clip to >=0 and force shares to sum to 100 within each district.
    """
    out = df.copy()
    out[pred_col] = out[pred_col].clip(lower=0)

    sums = out.groupby(district_col)[pred_col].transform("sum")
    out[out_col] = (out[pred_col] / sums.where(sums > 0, 1.0)) * 100.0
    return out
