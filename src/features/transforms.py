from __future__ import annotations
import numpy as np
import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age buckets
    df["under_40_ratio"] = df["age_18-29_ratio"] + df["age_30-39_ratio"]
    df["40-60_ratio"] = df["age_40-49_ratio"] + df["age_50-59_ratio"]

    # Education compressions
    df["SEE-Inter"] = df["SLC or SEE % 2078"] + df["Intermediate & equivalent % 2078"]
    df["Grad+"] = df["Graduate & equivalent % 2078"] + df["Post graduate equivalent & above % 2078"]

    # Safe logs (in case you rebuild later without them)
    if "facebook_log" not in df.columns and "Facebook_Presence (In Thousands)" in df.columns:
        df["facebook_log"] = np.log1p(df["Facebook_Presence (In Thousands)"].clip(lower=0))
    if "leader_following_log" not in df.columns and "Top_leader_fb (in Thousands)" in df.columns:
        df["leader_following_log"] = np.log1p(df["Top_leader_fb (in Thousands)"].clip(lower=0))

    return df
