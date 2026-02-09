from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/FPTP")
OUT_PATH = Path("data/interim/fptp_infer_2082.csv")

def main():
    df82 = pd.read_csv(RAW_DIR / "fptp_clean_82.csv").copy()
    df82["election_year"] = 2082

    for c in ["state_id", "district_id", "const_id", "party_id"]:
        df82[c] = df82[c].astype(str)

    df82["candidate_name"] = df82["candidate_name"].astype(str).str.strip()
    df82["gender"] = df82["gender"].astype(str).str.strip()
    df82["age"] = pd.to_numeric(df82["age"], errors="coerce")

    if "qualification" not in df82.columns:
        df82["qualification"] = "unknown"
    df82["qualification"] = df82["qualification"].fillna("unknown").astype(str)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df82.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(df82):,} rows)")

if __name__ == "__main__":
    main()
