from __future__ import annotations

from pathlib import Path
import pandas as pd


# Paths
TRAIN_IN = Path("data/interim/fptp_train.csv")
INFER_IN = Path("data/interim/fptp_infer.csv")

TRAIN_OUT = Path("data/processed/fptp_train_model_2079.csv")
INFER_OUT = Path("data/processed/fptp_infer_model_2082.csv")


# Column rename maps
TRAIN_RENAME = {
    # candidate lag
    "cand_vote_share_prev_74": "cand_vote_share_prev",
    "cand_rank_prev_74": "cand_rank_prev",
    "cand_won_prev_74": "cand_won_prev",

    # party lag
    "party_vote_share_prev_74": "party_vote_share_prev",
    "party_rank_prev_74": "party_rank_prev",
    "party_won_prev_74": "party_won_prev",
    "party_contested_prev_74": "party_contested_prev",   
    "pr_share_district_74": "pr_vote_share_prev",
    "pr_rank_district_74": "pr_rank_prev",

    # seat context lag
    "enp_prev_74": "enp_prev",
    "num_candidates_prev_74": "num_candidates_prev",
    "margin_prev_74": "margin_prev",
    "top1_share_prev_74": "top1_share_prev",
    "top2_share_prev_74": "top2_share_prev",
}

INFER_RENAME = {
    # candidate lag
    "cand_vote_share_prev_79": "cand_vote_share_prev",
    "cand_rank_prev_79": "cand_rank_prev",
    "cand_won_prev_79": "cand_won_prev",

    # party lag
    "party_vote_share_prev_79": "party_vote_share_prev",
    "party_rank_prev_79": "party_rank_prev",
    "party_won_prev_79": "party_won_prev",
    "party_contested_prev_79": "party_contested_prev",
    "pr_share_seat_79": "pr_vote_share_prev",
    "pr_rank_seat_79": "pr_rank_prev",

    # seat context lag
    "enp_prev_79": "enp_prev",
    "num_candidates_prev_79": "num_candidates_prev",
    "margin_prev_79": "margin_prev",
    "top1_share_prev_79": "top1_share_prev",
    "top2_share_prev_79": "top2_share_prev",
}


# Feature lists
TARGET_COL = ["vote_share", "rank_in_const", "won_seat"]

BASE_FEATURES = [
    # candidate attributes
    "age",
    "gender",
    "qualification",

    # demographics
    "Literacy Rate 2078",
    "edu_below_slc",
    "edu_slc_to_inter",
    "edu_grad_or_higher",

    # candidate lag
    "cand_vote_share_prev",
    "cand_rank_prev",
    "cand_won_prev",

    # party lag
    "party_vote_share_prev",
    "party_rank_prev",
    "party_won_prev",
    "party_contested_prev",
    "pr_vote_share_prev",
    "pr_rank_prev",

    # seat context lag
    "enp_prev",
    "num_candidates_prev",
    "margin_prev",
    "top1_share_prev",
    "top2_share_prev",

    # political alignment
    "is_independent",
    "is_alternative_party",
    "is_youth_favoured_party",
    "has_national_leader_presence",
    "is_old_established_party",
    "was_in_ousted_government",
]

META_COLS = [
    "state_id",
    "district_id",
    "const_id",
    "seat_id",
    "party_id",
    "candidate_name",
    "candidate_key",
    "election_year",
]



# Helpers
def select_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def main():
    train = pd.read_csv(TRAIN_IN)
    infer = pd.read_csv(INFER_IN)

    # Keep only 2079 rows for training
    train = train[train["election_year"] == 2079].copy()

    # Rename lag columns
    train = train.rename(columns=TRAIN_RENAME)
    infer = infer.rename(columns=INFER_RENAME)

    # Build final column sets
    train_cols = (
        select_existing(train, META_COLS)
        + select_existing(train, BASE_FEATURES)
        + select_existing(train, TARGET_COL)
    )

    infer_cols = (
        select_existing(infer, META_COLS)
        + select_existing(infer, BASE_FEATURES)
    )

    train_final = train[train_cols].copy()
    infer_final = infer[infer_cols].copy()

    # Ensure output dirs
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Save
    train_final.to_csv(TRAIN_OUT, index=False, encoding="utf-8")
    infer_final.to_csv(INFER_OUT, index=False, encoding="utf-8")

    print(f"[OK] Train model table written: {TRAIN_OUT}  rows={len(train_final):,}")
    print(f"[OK] Infer model table written: {INFER_OUT}  rows={len(infer_final):,}")


if __name__ == "__main__":
    main()
