from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/FPTP")
OUT_PATH = Path("data/interim/fptp_train_long_74_79.csv")

def _standardize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["election_year"] = year

    # Ensure consistent types
    for c in ["state_id", "district_id", "const_id", "party_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Uniform schema
    if "qualification" not in df.columns:
        df["qualification"] = "unknown"

    # Basic cleaning
    df["candidate_name"] = df["candidate_name"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df

def main():
    df74 = pd.read_csv(RAW_DIR / "fptp_clean_74.csv")
    df79 = pd.read_csv(RAW_DIR / "fptp_clean_79.csv")

    df74 = _standardize(df74, 2074)
    df79 = _standardize(df79, 2079)

    df = pd.concat([df74, df79], ignore_index=True)

    # Total votes per constituency-year
    df["total_votes_const"] = df.groupby(["district_id", "const_id", "election_year"])["vote_received"].transform("sum")
    df["vote_share"] = df["vote_received"] / df["total_votes_const"]

    # Rank (1 = winner)
    df["rank_in_const"] = df.groupby(["district_id", "const_id", "election_year"])["vote_received"] \
                            .rank(method="first", ascending=False).astype(int)

    df["won_seat"] = (df["rank_in_const"] == 1).astype(int)

    # Top1/Top2 shares & margin
    top2 = (
        df.sort_values(["district_id", "const_id", "election_year", "vote_received"], ascending=[True,True, True, False])
          .groupby(["district_id", "const_id", "election_year"])
          .head(2)
          .assign(r=lambda x: x.groupby(["district_id", "const_id", "election_year"]).cumcount() + 1)
          .pivot_table(index=["district_id", "const_id", "election_year"], columns="r", values="vote_share", aggfunc="first")
          .rename(columns={1: "top1_vote_share", 2: "top2_vote_share"})
          .reset_index()
    )
    df = df.merge(top2, on=["district_id", "const_id", "election_year"], how="left")
    df["margin_top1_top2"] = (df["top1_vote_share"] - df["top2_vote_share"]).fillna(0.0)

    # Competition metrics
    df["num_candidates"] = df.groupby(["district_id", "const_id", "election_year"])["candidate_name"].transform("count")
    # ENP = 1/sum(share^2)
    enp = (
        df.assign(share2=df["vote_share"] ** 2)
          .groupby(["district_id", "const_id", "election_year"])["share2"].sum()
          .replace(0, np.nan)
    )
    df["enp"] = df.set_index(["district_id", "const_id", "election_year"]).index.map(lambda k: 1.0 / enp.loc[k]).astype(float)

    # Sanity checks
    winners = df.groupby(["district_id", "const_id", "election_year"])["won_seat"].sum()
    if not (winners == 1).all():
        bad = winners[winners != 1]
        raise ValueError(f"Found constituency-years without exactly 1 winner:\n{bad.head(20)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
