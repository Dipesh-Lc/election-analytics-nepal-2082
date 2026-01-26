from __future__ import annotations
import pandas as pd

def allocate_pr_seats(
    national_df: pd.DataFrame,
    seats: int = 110,
    threshold_pct: float = 3.0,
    share_col: str = "pred_vote_share_national",
) -> pd.DataFrame:
    """
    Nepal PR (simplified using predicted national vote share %):
      1) Eligible parties: share_col >= threshold_pct
      2) Remove ineligible parties and renormalize eligible shares to sum to 100
      3) Allocate `seats` using Sainte-Laguë (odd divisors 1,3,5,...)

    Returns full table with:
      - eligible_pr_3pct
      - eligible_share_norm (renormalized among eligible; sums to 100 across eligible)
      - seats (0 for ineligible)
    """
    df = national_df.copy()

    if "party_id" not in df.columns:
        raise ValueError("national_df must contain 'party_id'")
    if share_col not in df.columns:
        raise ValueError(f"national_df must contain '{share_col}'")

    df[share_col] = pd.to_numeric(df[share_col], errors="coerce")
    if df[share_col].isna().any():
        bad = df[df[share_col].isna()][["party_id", share_col]].head(20)
        raise ValueError(f"Non-numeric/NA values found in {share_col}. Example rows:\n{bad}")

    # Eligibility
    df["eligible_pr_3pct"] = df[share_col] >= float(threshold_pct)

    eligible = df[df["eligible_pr_3pct"]].copy()
    if eligible.empty:
        raise ValueError(f"No parties meet the {threshold_pct}% threshold.")

    # Renormalize eligible shares to sum to 100 
    eligible_total = float(eligible[share_col].sum())
    eligible["eligible_share_norm"] = eligible[share_col] / eligible_total * 100.0

    # Sainte-Laguë allocation among eligible parties (odd divisors)
    parties = eligible["party_id"].tolist()
    votes = eligible["eligible_share_norm"].tolist()

    quotients = []
    for p, v in zip(parties, votes):
        for k in range(seats):
            divisor = 2 * k + 1  # 1,3,5,...
            quotients.append((p, v / divisor))

    quotients.sort(key=lambda x: x[1], reverse=True)
    winners = quotients[:seats]

    seat_counts = (
        pd.Series([p for p, _ in winners])
        .value_counts()
        .rename_axis("party_id")
        .reset_index(name="seats")
    )

    eligible = eligible.merge(seat_counts, on="party_id", how="left")
    eligible["seats"] = eligible["seats"].fillna(0).astype(int)

    out = df.copy()
    out["eligible_share_norm"] = 0.0
    out["seats"] = 0

    out = out.merge(
        eligible[["party_id", "eligible_share_norm", "seats"]],
        on="party_id",
        how="left",
        suffixes=("", "_elig")
    )

    if "eligible_share_norm_elig" in out.columns:
        out["eligible_share_norm"] = out["eligible_share_norm_elig"].fillna(0.0)
        out = out.drop(columns=["eligible_share_norm_elig"])
    else:
        out["eligible_share_norm"] = out["eligible_share_norm"].fillna(0.0)

    if "seats_elig" in out.columns:
        out["seats"] = out["seats_elig"].fillna(0).astype(int)
        out = out.drop(columns=["seats_elig"])
    else:
        out["seats"] = out["seats"].fillna(0).astype(int)

    # Hard guarantees
    out.loc[~out["eligible_pr_3pct"], "seats"] = 0
    if int(out["seats"].sum()) != int(seats):
        raise RuntimeError(f"Seat allocation failed: seats sum to {out['seats'].sum()} not {seats}.")

    # Sort for readability
    out = out.sort_values(["seats", share_col], ascending=False).reset_index(drop=True)
    return out
