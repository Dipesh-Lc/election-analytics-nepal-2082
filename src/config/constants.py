from __future__ import annotations

# --- PR settings ---
PR_SEATS = 110
PR_THRESHOLD = 3.0  

# --- Model columns ---
TARGET_COL = "vote_share_change"
LAG_COL = "vote_share_lag"
GROUP_COL = "district_id"

BASE_FEATURES = [
    # Political inertia
    "vote_share_lag",
    "is_new_party",
    "IS_Major?",
    "Was_Part_Of_Ousted_Government?",
    "GenZ_and_Youth_Favored?",
    "IS_ALTERNATIVE_FORCE?",

    # Demographics (compressed)
    "under_40_ratio",
    "40-60_ratio",
    "female_ratio",

    # Education (high signal only)
    "Literacy Rate 2078",
    "SEE-Inter",
    "Grad+",

    # Migration
    "Absent rate within country 2078",
    "Absent rate abroad 2078",

    # Mobilization
    "valid_turnout",
    "valid_turnout_lag",

    # Digital
    "leader_following_log",
]

ID_COLS = ["district_id", "party_id", "election_year"]
