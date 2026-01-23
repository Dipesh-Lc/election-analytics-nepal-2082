from __future__ import annotations
import pandas as pd

def add_predicted_votes(
    df_pred: pd.DataFrame,
    share_col: str = "pred_vote_share_norm",
    turnout_col: str = "valid_turnout",
    voters_col: str = "total_voters",
) -> pd.DataFrame:
    """
    Votes = total_voters * turnout * share
    turnout may be percent (e.g., 52.41) or fraction (0.5241); auto-detect.
    """
    out = df_pred.copy()

    turnout = out[turnout_col]
    turnout_frac = turnout.where(turnout <= 1.0, turnout / 100.0)

    out["pred_votes"] = out[voters_col] * turnout_frac * (out[share_col] / 100.0)
    return out

def national_vote_share(df_with_votes: pd.DataFrame) -> pd.DataFrame:
    g = df_with_votes.groupby("party_id", as_index=False)["pred_votes"].sum()
    total = g["pred_votes"].sum()
    g["pred_vote_share_national"] = (g["pred_votes"] / total) * 100.0
    return g.sort_values("pred_vote_share_national", ascending=False)
