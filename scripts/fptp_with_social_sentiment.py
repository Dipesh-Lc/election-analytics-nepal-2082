from pathlib import Path
import numpy as np
import pandas as pd

FB_CSV = Path("data/processed/topPARTYFacebook.csv")  
PRED_CSV = Path("data/outputs/fptp_vote_share_predictions/fptp_vote_share_predictions_2082.csv")

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std

def build_party_momentum_with_sentiment(fb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns party_id + momentum_score where momentum_score includes:
      - engagement intensity (reactions/comments/shares/views)
      - net sentiment (positive reaction rate - negative reaction rate)
      - penalty for high negative rate (optional extra)
    """
    df = fb_df.copy()

    # Safety
    df["num_posts"] = df["num_posts"].replace(0, np.nan)
    df["total_followers"] = df["avg_followers_total"].replace(0, np.nan)

    # Engagement intensity features (per post)
    df["reactions_per_post"] = df["avg_reactions_total"] 
    df["comments_per_post"] = df["avg_num_comments"]
    df["shares_per_post"] = df["avg_num_shares"]
    df["video_reach_rate"] = df["median_video_views"] / df["avg_followers_total"]
    df["engagement_rate_followers"] = (df["reactions_per_post"] + df["shares_per_post"] + df["comments_per_post"]) / df["avg_followers_total"]
    df["total_reach"] = np.log1p(df["avg_followers_total"])

    # Positive: Love + Care (+ optionally Like) Negative: Angry + Haha
    df["pos_rate"] = (df["avg_react_Love"] + df["avg_react_Care"] + 0.25 * df["avg_react_Like"]) / df["avg_reactions_total"]
    df["neg_rate"] = (df["avg_react_Angry"] + 0.5 * df["avg_react_Haha"]) / df["avg_reactions_total"]

    # if total reactions missing
    df["pos_rate"] = df["pos_rate"].fillna(0.0)
    df["neg_rate"] = df["neg_rate"].fillna(0.0)

    # Net sentiment: reward positive share, penalize negative share
    df["net_sentiment"] = df["pos_rate"] - df["neg_rate"]

    # Combine into momentum score 
    momentum = (
          0.15 * zscore(df["total_reach"]) +
          0.2 * zscore(df["video_reach_rate"]) +
          0.5 * zscore(df["engagement_rate_followers"]) +
          0.15 * zscore(df["net_sentiment"]) 
    )

    df["momentum_score"] = momentum.clip(lower=-3.0, upper=3.0)

    return df[["party_id", "momentum_score"]]


def apply_momentum_adjustment(
    pred_df: pd.DataFrame,
    party_momentum: pd.DataFrame,
    seat_col: str = "seat_id",
    pred_share_cols: list[str] = None,
    w: float = 0.5,
    k: float = 0.4,
) -> pd.DataFrame:
    """
    Adds new columns at the end:
      <col>_mom_adj (adjusted vote share)
      <col>_mom_winner (seat winner from adjusted share)
    """
    if pred_share_cols is None:
        pred_share_cols = [c for c in pred_df.columns if c.endswith("_pred_share")]

    out = pred_df.copy()

    mom = party_momentum.copy()
    mom["multiplier"] = np.exp(k * mom["momentum_score"])

    out = out.merge(mom[["party_id", "multiplier"]], on="party_id", how="left")

    # parties not in FB table (or independents) => neutral multiplier
    out["multiplier"] = out["multiplier"].fillna(1.0)

    new_cols = []
    for col in pred_share_cols:
        adj_col = col.replace("_pred_share", "_pred_share_mom_adj")
        win_col = col.replace("_pred_share", "_pred_winner_mom_adj")

        # multiplicative adjustment
        out[adj_col] = out[col].astype(float) * (out["multiplier"] ** w)

        # renormalize per seat
        denom = out.groupby(seat_col)[adj_col].transform("sum").replace(0.0, np.nan)
        out[adj_col] = (out[adj_col] / denom).fillna(0.0)

        # seat winner from adjusted share
        max_in_seat = out.groupby(seat_col)[adj_col].transform("max")
        out[win_col] = (out[adj_col] == max_in_seat).astype(int)

        new_cols.extend([adj_col, win_col])

    # put prediction additions at the end 
    base_cols = [c for c in pred_df.columns]
    extra_cols = ["multiplier"] + [c for c in new_cols if c not in base_cols]
    out = out[base_cols + [c for c in extra_cols if c not in base_cols]]

    return out

def main():
    fb_df = pd.read_csv(FB_CSV)
    pred_df = pd.read_csv(PRED_CSV)

    party_mom = build_party_momentum_with_sentiment(fb_df)

    out = apply_momentum_adjustment(
        pred_df=pred_df,
        party_momentum=party_mom[["party_id", "momentum_score"]],
        pred_share_cols=["xgboost_pred_share", "ridge_pred_share"],
        w=1,
        k=0.4,
    )

    out.to_csv("data/outputs/fptp_vote_share_predictions/fptp_vote_share_predictions_with_momentum_2082.csv", index=False, encoding="utf-8")

    print("[OK] Momentum adjustment complete.")
    print("Saved:", "data/outputs/fptp_vote_share_predictions/fptp_vote_share_predictions_with_momentum_2082.csv")


if __name__ == "__main__":
    main()