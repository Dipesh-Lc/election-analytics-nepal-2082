from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# Paths (inputs)
TRAIN_IN = Path("data/interim/fptp_train_long_74_79.csv")
INFER_IN = Path("data/interim/fptp_infer_2082.csv")

PR79_IN = Path("data/raw/FPTP/pr_clean_79.csv")
PR74_IN = Path("data/raw/FPTP/pr_clean_74.csv")

DIST_DEMO_IN = Path("data/raw/district_wise_demographics_2078.csv")

DIST_REF_IN = Path("data/reference/district.csv")

# Paths (outputs)
TRAIN_OUT = Path("data/interim/fptp_train.csv")
INFER_OUT = Path("data/interim/fptp_infer.csv")

# Political alignment feature definitions
#------------------------------------------------------
OLD_ESTABLISHED_PARTIES = {"P001", "P002", "P003", "P005", "P006"} # National-level parties with sustained parliamentary presence and governing role

ALTERNATIVE_PARTIES = {"P004", "P007", "P009", "P046", "P057"} # newly established viable alternative parties 

YOUTH_FAVOURED_PARTIES = {"P004"}

NATIONAL_LEADER_PRESENCE = {"P001", "P002", "P003", "P004", "P005", "P006", "P007","P046", "P057"} # parties with prominent nationally influential leaders 

OUSTED_2079 = {"P002", "P003","P006"}
OUSTED_2082 = {"P001", "P002","P010"}
#------------------------------------------------------

# Helper functions for data cleaning and validation
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKC", str(x)).strip()
    return s


def norm_name(x) -> str:
    """Normalize candidate name for stable matching (unicode, lower, remove punctuation)."""
    s = norm_text(x).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

# Key construction
def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["state_id", "district_id", "const_id", "party_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    #const_id not unique nationally -> build seat_id
    df["seat_id"] = df["district_id"] + "_" + df["const_id"]

    df["candidate_name_norm"] = df["candidate_name"].apply(norm_name)
    df["candidate_key"] = df["candidate_name_norm"] + "_" + df["seat_id"]
    return df

# Seat-level PR priors from 2079
def build_pr79_seat_priors(pr79: pd.DataFrame) -> pd.DataFrame:
    """
    pr_clean_79.csv: district_id,const_id,party_id,vote_received
    -> seat_id, party_id, pr_share_seat_79, pr_rank_seat_79
    Adjustment:
      - For 2082 merge P008 into P003 by summing their 2079 PR votes per seat.
    """
    pr = pr79.copy()
    require_cols(pr, ["district_id", "const_id", "party_id", "vote_received"], "pr_clean_79")

    pr["district_id"] = pr["district_id"].astype(str).str.strip()
    pr["const_id"] = pr["const_id"].astype(str).str.strip()
    pr["party_id"] = pr["party_id"].astype(str).str.strip()
    pr["seat_id"] = pr["district_id"] + "_" + pr["const_id"]

    pr["vote_received"] = pd.to_numeric(pr["vote_received"], errors="coerce").fillna(0.0)

    #MERGE RULE: P008 -> P003 (sum votes within seat) 
    pr.loc[pr["party_id"] == "P008", "party_id"] = "P003"
    pr = (
        pr.groupby(["seat_id", "party_id"], as_index=False)["vote_received"]
          .sum()
    )

    # Seat totals after merge
    pr["total_pr_votes_seat_79"] = pr.groupby("seat_id")["vote_received"].transform("sum")

    pr["pr_share_seat_79"] = np.where(
        pr["total_pr_votes_seat_79"] > 0,
        pr["vote_received"] / pr["total_pr_votes_seat_79"],
        0.0
    )

    pr["pr_rank_seat_79"] = (
        pr.groupby("seat_id")["vote_received"]
          .rank(method="first", ascending=False)
          .astype(int)
    )

    return pr[["seat_id", "party_id", "pr_share_seat_79", "pr_rank_seat_79"]]


# District-level PR priors from 2074
def build_pr74_district_priors(pr74: pd.DataFrame) -> pd.DataFrame:
    """
    pr_clean_74.csv: district_id,party_id,votes_received,year
    -> district_id, party_id, pr_share_district_74, pr_rank_district_74

    This is district-level only and will be broadcast to all seats in that district.
    """
    pr = pr74.copy()
    require_cols(pr, ["district_id", "party_id", "votes_received"], "pr_clean_74")

    pr["district_id"] = pr["district_id"].astype(str).str.strip()
    pr["party_id"] = pr["party_id"].astype(str).str.strip()
    pr["votes_received"] = pd.to_numeric(pr["votes_received"], errors="coerce").fillna(0.0)

    pr["total_pr_votes_district_74"] = pr.groupby("district_id")["votes_received"].transform("sum")
    pr["pr_share_district_74"] = np.where(
        pr["total_pr_votes_district_74"] > 0,
        pr["votes_received"] / pr["total_pr_votes_district_74"],
        0.0
    )

    pr["pr_rank_district_74"] = (
        pr.groupby("district_id")["votes_received"]
          .rank(method="first", ascending=False)
          .astype(int)
    )

    return pr[["district_id", "party_id", "pr_share_district_74", "pr_rank_district_74"]]



# District demographics
def load_district_demo_with_id(demo_path: Path, district_ref_path: Path) -> pd.DataFrame:
    """
    Load district demographics and attach district_id using shared DISTRICT code.
    """
    demo = safe_read_csv(demo_path).copy()
    ref = safe_read_csv(district_ref_path).copy()

    demo.columns = [c.strip() for c in demo.columns]
    ref.columns = [c.strip() for c in ref.columns]

    if "DISTRICT" not in demo.columns:
        raise ValueError("district_wise_demographics.csv must contain 'DISTRICT' column.")
    if "DISTRICT" not in ref.columns or "district_id" not in ref.columns:
        raise ValueError("data/reference/district.csv must contain 'DISTRICT' and 'district_id' columns.")
    
    ref["district_id"] = ref["district_id"].astype(str).str.strip()
    
    out = demo.merge(ref[["DISTRICT", "district_id"]], on="DISTRICT", how="left")

    if out["district_id"].isna().any():
        missing = out.loc[out["district_id"].isna(), "DISTRICT"].drop_duplicates().tolist()
        raise ValueError(f"Unmapped DISTRICT codes in demographics: {missing[:30]}")

    return out

# Feature engineering for district demographics
def select_demo_features(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Select district-level demographic features:
    - Literacy rate
    - Education buckets (below SLC, SLC to Intermediate, Graduate or higher)
    """
    required = [
        "district_id",
        "Literacy Rate 2078",
        "Primary Education % 2078",
        "Lower Secondary % 2078",
        "Upper Secondary % 2078",
        "SLC or SEE % 2078",
        "Intermediate & equivalent % 2078",
        "Graduate & equivalent % 2078",
        "Post graduate equivalent & above % 2078",
    ]

    missing = [c for c in required if c not in demo.columns]
    if missing:
        raise ValueError(f"Missing required demographic columns: {missing}")

    out = demo[required].copy()
    out["district_id"] = out["district_id"].astype(str).str.strip()

    # Convert numeric columns
    for c in out.columns:
        if c != "district_id":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Education buckets
    out["edu_below_slc"] = (out["Primary Education % 2078"] + out["Lower Secondary % 2078"] + out["Upper Secondary % 2078"])

    out["edu_slc_to_inter"] = (out["SLC or SEE % 2078"] + out["Intermediate & equivalent % 2078"])

    out["edu_grad_or_higher"] = (out["Graduate & equivalent % 2078"] + out["Post graduate equivalent & above % 2078"])

    # Final selection
    final_cols = [
        "district_id",
        "Literacy Rate 2078",
        "edu_below_slc",
        "edu_slc_to_inter",
        "edu_grad_or_higher",
    ]

    return out[final_cols]


# Lag features for 2079 training table, built from 2074,2079 training table
def build_train79_with_lags(train_long_74_79: pd.DataFrame) -> pd.DataFrame:
    """
    Return ONLY 2079 rows, augmented with:
      - Candidate lag from 2074 (same candidate_key in same seat_id) <-- applies to ALL incl P000
      - Party lag from 2074 (seat_id + party_id)                     <-- applies only if party_id != 'P000'
      - Seat context lag from 2074 (seat-level):
          enp_prev_74, num_candidates_prev_74, margin_prev_74, top1_share_prev_74, top2_share_prev_74
    For party_id == 'P000':
      party_* lag features are forced to 0 (independent has no party history).
    """
    df = train_long_74_79.copy()
    df["election_year"] = pd.to_numeric(df["election_year"], errors="coerce").astype(int)

    df74 = df[df["election_year"] == 2074].copy()
    df79 = df[df["election_year"] == 2079].copy()


    # Candidate lag (2074 -> 2079)
    cand_prev = df74[["seat_id", "candidate_key", "vote_share", "rank_in_const", "won_seat"]].rename(
        columns={
            "vote_share": "cand_vote_share_prev_74",
            "rank_in_const": "cand_rank_prev_74",
            "won_seat": "cand_won_prev_74",
        }
    )
    df79 = df79.merge(cand_prev, on=["seat_id", "candidate_key"], how="left")

    for c in ["cand_vote_share_prev_74", "cand_rank_prev_74", "cand_won_prev_74"]:
        df79[f"missing_{c}"] = df79[c].isna().astype(int)
        df79[c] = df79[c].fillna(0.0)

    # Party lag (2074 -> 2079), excluding P000
    df74_party = df74[df74["party_id"].astype(str) != "P000"].copy()

    party_prev = (
        df74_party.sort_values(["seat_id", "party_id", "vote_share"], ascending=[True, True, False])
                  .drop_duplicates(subset=["seat_id", "party_id"])
                  [["seat_id", "party_id", "vote_share", "rank_in_const", "won_seat"]]
                  .rename(columns={
                      "vote_share": "party_vote_share_prev_74",
                      "rank_in_const": "party_rank_prev_74",
                      "won_seat": "party_won_prev_74",
                  })
    )
    party_prev["party_contested_prev_74"] = 1

    df79 = df79.merge(party_prev, on=["seat_id", "party_id"], how="left")

    df79["party_contested_prev_74"] = df79["party_contested_prev_74"].fillna(0).astype(int)

    for c in ["party_vote_share_prev_74", "party_rank_prev_74", "party_won_prev_74"]:
        df79[f"missing_{c}"] = df79[c].isna().astype(int)
        df79[c] = df79[c].fillna(0.0)

    # Force ALL party lag features to 0 for independents (P000)
    indep_mask = df79["party_id"].astype(str) == "P000"
    df79.loc[indep_mask, "party_contested_prev_74"] = 0
    for c in ["party_vote_share_prev_74", "party_rank_prev_74", "party_won_prev_74"]:
        df79.loc[indep_mask, c] = 0.0
        df79.loc[indep_mask, f"missing_{c}"] = 1


    # Seat context lag (2074 -> 2079)
    ctx74 = (
        df74[["seat_id", "enp", "num_candidates", "margin_top1_top2", "top1_vote_share", "top2_vote_share"]]
        .drop_duplicates()
        .rename(columns={
            "enp": "enp_prev_74",
            "num_candidates": "num_candidates_prev_74",
            "margin_top1_top2": "margin_prev_74",
            "top1_vote_share": "top1_share_prev_74",
            "top2_vote_share": "top2_share_prev_74",
        })
    )

    df79 = df79.merge(ctx74, on=["seat_id"], how="left")

    for c in ["enp_prev_74", "num_candidates_prev_74", "margin_prev_74", "top1_share_prev_74", "top2_share_prev_74"]:
        df79[f"missing_{c}"] = df79[c].isna().astype(int)
        df79[c] = df79[c].fillna(0.0)

    # Convenience flag
    df79["is_independent"] = (df79["party_id"].astype(str) == "P000").astype(int)

    return df79


# Lag features for 2082 inference table, built from 2079 training table
def build_infer82_with_lags(infer_82: pd.DataFrame, train_long_74_79: pd.DataFrame) -> pd.DataFrame:
    """
    For 2082 inference, bring:
      Candidate lag from 2079 (same candidate_key in same seat_id)  <-- applies to ALL incl P000
      Party lag from 2079 (seat_id + party_id)                      <-- applies only if party_id != 'P000'
      Seat context from 2079 (seat-level):
        - enp_prev_79, num_candidates_prev_79, margin_prev_79, top1_share_prev_79, top2_share_prev_79
    For party_id == 'P000':
      party_* lag features are forced to 0.
    """
    inf = infer_82.copy()
    tr = train_long_74_79.copy()

    tr["election_year"] = pd.to_numeric(tr["election_year"], errors="coerce").astype(int)
    tr79 = tr[tr["election_year"] == 2079].copy()

    # Candidate lag (2079 -> 2082) 
    cand_prev = tr79[["seat_id", "candidate_key", "vote_share", "rank_in_const", "won_seat"]].rename(
        columns={
            "vote_share": "cand_vote_share_prev_79",
            "rank_in_const": "cand_rank_prev_79",
            "won_seat": "cand_won_prev_79",
        }
    )
    inf = inf.merge(cand_prev, on=["seat_id", "candidate_key"], how="left")

    for c in ["cand_vote_share_prev_79", "cand_rank_prev_79", "cand_won_prev_79"]:
        inf[f"missing_{c}"] = inf[c].isna().astype(int)
        inf[c] = inf[c].fillna(0.0)

    # Party lag (2079 -> 2082), EXCLUDING P000 
    tr79_party = tr79[tr79["party_id"].astype(str) != "P000"].copy()

    party_prev = (
        tr79_party.sort_values(["seat_id", "party_id", "vote_share"], ascending=[True, True, False])
                  .drop_duplicates(subset=["seat_id", "party_id"])
                  [["seat_id", "party_id", "vote_share", "rank_in_const", "won_seat"]]
                  .rename(columns={
                      "vote_share": "party_vote_share_prev_79",
                      "rank_in_const": "party_rank_prev_79",
                      "won_seat": "party_won_prev_79",
                  })
    )
    party_prev["party_contested_prev_79"] = 1

    inf = inf.merge(party_prev, on=["seat_id", "party_id"], how="left")

    inf["party_contested_prev_79"] = inf["party_contested_prev_79"].fillna(0).astype(int)

    for c in ["party_vote_share_prev_79", "party_rank_prev_79", "party_won_prev_79"]:
        inf[f"missing_{c}"] = inf[c].isna().astype(int)
        inf[c] = inf[c].fillna(0.0)

    # Seat context (2079 -> 2082)
    ctx = (
        tr79[["seat_id", "enp", "num_candidates", "margin_top1_top2", "top1_vote_share", "top2_vote_share"]]
        .drop_duplicates()
        .rename(columns={
            "enp": "enp_prev_79",
            "num_candidates": "num_candidates_prev_79",
            "margin_top1_top2": "margin_prev_79",
            "top1_vote_share": "top1_share_prev_79",
            "top2_vote_share": "top2_share_prev_79",
        })
    )
    inf = inf.merge(ctx, on=["seat_id"], how="left")

    for c in ["enp_prev_79", "num_candidates_prev_79", "margin_prev_79", "top1_share_prev_79", "top2_share_prev_79"]:
        inf[f"missing_{c}"] = inf[c].isna().astype(int)
        inf[c] = inf[c].fillna(0.0)

    # Force ALL party lag features to 0 for independents (P000) 
    indep_mask = inf["party_id"].astype(str) == "P000"

    inf.loc[indep_mask, "party_contested_prev_79"] = 0
    for c in ["party_vote_share_prev_79", "party_rank_prev_79", "party_won_prev_79"]:
        inf.loc[indep_mask, c] = 0.0
        inf.loc[indep_mask, f"missing_{c}"] = 1

    inf["is_independent"] = (inf["party_id"].astype(str) == "P000").astype(int)

    return inf

def add_political_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["party_id"] = out["party_id"].astype(str).str.strip()
    out["election_year"] = pd.to_numeric(out["election_year"], errors="coerce").astype(int)

    is_ind = out["party_id"] == "P000"

    out["is_alternative_party"] = out["party_id"].isin(ALTERNATIVE_PARTIES).astype(int)
    out["is_youth_favoured_party"] = out["party_id"].isin(YOUTH_FAVOURED_PARTIES).astype(int)
    out["has_national_leader_presence"] = out["party_id"].isin(NATIONAL_LEADER_PRESENCE).astype(int)
    out["is_old_established_party"] = out["party_id"].isin(OLD_ESTABLISHED_PARTIES).astype(int)

    # independents forced false for party-level features
    for c in ["is_alternative_party","is_youth_favoured_party","has_national_leader_presence","is_old_established_party"]:
        out.loc[is_ind, c] = 0

    # year-dependent ousted flag
    out["was_in_ousted_government"] = 0
    out.loc[(out["election_year"] == 2079) & out["party_id"].isin(OUSTED_2079), "was_in_ousted_government"] = 1
    out.loc[(out["election_year"] == 2082) & out["party_id"].isin(OUSTED_2082), "was_in_ousted_government"] = 1
    out.loc[is_ind, "was_in_ousted_government"] = 0

    return out



def main():
    # Load
    train = safe_read_csv(TRAIN_IN)
    infer = safe_read_csv(INFER_IN)

    pr79 = safe_read_csv(PR79_IN)
    pr74 = safe_read_csv(PR74_IN)

    demo_raw = load_district_demo_with_id(DIST_DEMO_IN, DIST_REF_IN if DIST_REF_IN.exists() else None)
    demo = select_demo_features(demo_raw)    

    
    # Validate minimal cols
    require_cols(train, ["state_id", "district_id", "const_id", "party_id", "candidate_name", "gender", "age", "election_year"], "fptp_train_long_74_79")
    require_cols(infer, ["state_id", "district_id", "const_id", "party_id", "candidate_name", "gender", "age", "election_year"], "fptp_infer_2082")

    # Add keys
    train = add_keys(train)
    infer = add_keys(infer)

    # Ensure numeric year
    train["election_year"] = pd.to_numeric(train["election_year"], errors="coerce").astype(int)
    infer["election_year"] = pd.to_numeric(infer["election_year"], errors="coerce").astype(int)

    # PR priors
    pr79_feats = build_pr79_seat_priors(pr79)
    pr74_feats = build_pr74_district_priors(pr74)

    # Merge PR priors into train + infer
    train = train.merge(pr79_feats, on=["seat_id", "party_id"], how="left")
    infer = infer.merge(pr79_feats, on=["seat_id", "party_id"], how="left")

    train = train.merge(pr74_feats, on=["district_id", "party_id"], how="left")
    infer = infer.merge(pr74_feats, on=["district_id", "party_id"], how="left")

    for c in ["pr_share_seat_79", "pr_rank_seat_79", "pr_share_district_74", "pr_rank_district_74"]:
        for df in (train, infer):
            df[f"missing_{c}"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(0.0)

    # District demographics
    train = train.merge(demo, on="district_id", how="left")
    infer = infer.merge(demo, on="district_id", how="left")

    # Missing flags for district demographics
    demo_cols = [c for c in demo.columns if c != "district_id"]
    for c in demo_cols:
        for df in (train, infer):
            df[c] = df[c].fillna(df[c].median(skipna=True) if pd.api.types.is_numeric_dtype(df[c]) else 0.0)


    # Candidate lags 2074->2079 for training table (only adds to 2079 rows)
    train = build_train79_with_lags(train)

    # Add priors to 2082 inference from 2079 FPTP context
    infer = build_infer82_with_lags(infer, train)

    # Political alignment features
    train = add_political_alignment_features(train)
    infer = add_political_alignment_features(infer)

    # Output
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False, encoding="utf-8")
    infer.to_csv(INFER_OUT, index=False, encoding="utf-8")

    print(f"[OK] Wrote: {TRAIN_OUT}  rows={len(train):,}")
    print(f"[OK] Wrote: {INFER_OUT}  rows={len(infer):,}")


if __name__ == "__main__":
    main()