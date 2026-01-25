import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.pr.seats import allocate_pr_seats


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs_for_file(path_str: str) -> None:
    p = Path(path_str)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("config/pr_consolidation.json")

    # Params
    top_k = int(cfg["top_k"])
    c = float(cfg["c"])

    alpha = float(cfg["alpha"])
    alpha_shock = float(cfg.get("alpha_shock", alpha))

    w_eng = float(cfg["w_eng"])
    w_video = float(cfg["w_video"])

    beta_pos = float(cfg["beta_pos"])
    beta_neg = float(cfg["beta_neg"])

    pos_cols = cfg.get("pos_reactions", ["avg_react_Love", "avg_react_Care"])
    neg_cols = cfg.get("neg_reactions", ["avg_react_Angry", "avg_react_Sad"])

    shock_cfg = cfg.get("shock", {})
    shock_enabled = bool(shock_cfg.get("enabled", False))
    require_new_party = bool(shock_cfg.get("require_new_party", False))
    new_party_source = shock_cfg.get("new_party_source", "pred")  # "pred" or "social"
    new_party_col = shock_cfg.get("new_party_col", "is_new_party")
    z_thr = float(shock_cfg.get("z_social_threshold", 1.5))
    manual_ids = set(shock_cfg.get("manual_party_ids", []) or [])

    
    # Load inputs
    pred_path = cfg["input_predictions"]
    social_path = cfg["input_social"]

    pred = pd.read_csv(pred_path)
    soc = pd.read_csv(social_path)

    # Basic schema
    pred["party_id"] = pred["party_id"].astype(str).str.strip()
    pred["pred_vote_share_national"] = pd.to_numeric(
        pred["pred_vote_share_national"], errors="coerce"
    ).fillna(0.0)

    # Sort + viable
    pred = pred.sort_values("pred_vote_share_national", ascending=False).reset_index(drop=True)
    pred["rank"] = np.arange(1, len(pred) + 1)
    pred["is_viable"] = pred["rank"] <= top_k

    
    # Save baseline
    baseline = pred[["party_id", "pred_vote_share_national"]].copy()
    ensure_dirs_for_file(cfg["out_baseline"])
    baseline.to_csv(cfg["out_baseline"], index=False)

    
    # Consolidation mass
    S_V = pred.loc[pred["is_viable"], "pred_vote_share_national"].sum()
    S_rest = 100.0 - S_V
    M = c * S_rest

    # shrink non-viable parties
    base = pred.copy()
    base["share_after"] = base["pred_vote_share_national"]
    base.loc[~base["is_viable"], "share_after"] *= (1 - c)

    
    # Option A: structure-only redistribution
    
    struct = base.copy()
    w = struct.loc[struct["is_viable"], "pred_vote_share_national"].clip(lower=0)
    w = w / w.sum()
    struct.loc[struct["is_viable"], "share_after"] += M * w.values

    struct["share_after"] = struct["share_after"].clip(lower=0)
    struct["share_after"] = struct["share_after"] / struct["share_after"].sum() * 100.0

    struct_out = struct[["party_id", "share_after"]].rename(
        columns={"share_after": "pred_vote_share_national"}
    )
    ensure_dirs_for_file(cfg["out_struct"])
    struct_out.to_csv(cfg["out_struct"], index=False)

    
    # Build social z_social (for Option B)

    soc = soc.copy()
    soc["party_id"] = soc["party_id"].astype(str).str.strip()

    # Convert needed cols
    for col in ["num_posts", "avg_reactions_total", "avg_followers_total", "avg_video_views"]:
        if col in soc.columns:
            soc[col] = pd.to_numeric(soc[col], errors="coerce")

    # engagement rate
    soc["eng_rate"] = (
        soc["avg_reactions_total"].fillna(0.0)
        / soc["avg_followers_total"].replace(0, np.nan)
    ).fillna(0.0)

    soc["log_posts"] = np.log1p(soc["num_posts"].fillna(0.0))
    soc["log_video"] = np.log1p(soc["avg_video_views"].fillna(0.0))

    # pos/neg shares
    denom = soc["avg_reactions_total"].replace(0, np.nan)

    pos_sum = 0.0
    for col in pos_cols:
        if col in soc.columns:
            pos_sum = pos_sum + pd.to_numeric(soc[col], errors="coerce").fillna(0.0)

    neg_sum = 0.0
    for col in neg_cols:
        if col in soc.columns:
            neg_sum = neg_sum + pd.to_numeric(soc[col], errors="coerce").fillna(0.0)

    soc["pos_share"] = (pos_sum / denom).fillna(0.0)
    soc["neg_share"] = (neg_sum / denom).fillna(0.0)

    # z-scores across whatever is in social table (ideally top parties)
    soc["z_eng"] = zscore(soc["eng_rate"])
    soc["z_video"] = zscore(soc["log_video"])
    soc["z_pos"] = zscore(soc["pos_share"])
    soc["z_neg"] = zscore(soc["neg_share"])

    soc["z_social"] = (
        w_eng * soc["z_eng"]
        + w_video * soc["z_video"]
        + beta_pos * soc["z_pos"]
        - beta_neg * soc["z_neg"]
    )

    
    # Option B: structure × exp(alpha * z_social)
    
    tilted = base.merge(soc[["party_id", "z_social"]], on="party_id", how="left")
    tilted["z_social"] = tilted["z_social"].fillna(0.0)

    viable_mask = tilted["is_viable"]
    base_share = tilted.loc[viable_mask, "pred_vote_share_national"].clip(lower=0)

    tilt_mult = np.exp(alpha * tilted.loc[viable_mask, "z_social"].values)
    wB = base_share.values * tilt_mult
    wB = wB / wB.sum()

    tilted_B = tilted.copy()
    tilted_B.loc[viable_mask, "share_after"] += M * wB
    tilted_B["share_after"] = tilted_B["share_after"].clip(lower=0)
    tilted_B["share_after"] = tilted_B["share_after"] / tilted_B["share_after"].sum() * 100.0

    social_out = tilted_B[["party_id", "share_after", "z_social"]].rename(
        columns={"share_after": "pred_vote_share_national"}
    )
    ensure_dirs_for_file(cfg["out_social"])
    social_out.to_csv(cfg["out_social"], index=False)

    
    # Option B + Shock: alpha_eff is alpha_shock only for shock parties
    
    tilted_S = tilted.copy()

    # Determine is_new_party if present/required
    def get_is_new_party(pred_df: pd.DataFrame, social_df: pd.DataFrame) -> pd.Series:
        if new_party_source == "social" and new_party_col in social_df.columns:
            m = social_df[["party_id", new_party_col]].copy()
            m[new_party_col] = pd.to_numeric(m[new_party_col], errors="coerce").fillna(0).astype(int)
            return tilted_S.merge(m, on="party_id", how="left")[new_party_col].fillna(0).astype(int)

        if new_party_col in pred_df.columns:
            return pd.to_numeric(pred_df[new_party_col], errors="coerce").fillna(0).astype(int)

        return pd.Series(0, index=tilted_S.index, dtype=int)

    tilted_S["is_new_party"] = get_is_new_party(pred, soc)

    if shock_enabled:
        shock_i = tilted_S["party_id"].isin(manual_ids) | (tilted_S["z_social"] >= z_thr)
        if require_new_party:
            shock_i = shock_i & (tilted_S["is_new_party"] == 1)
        shock_i = shock_i & tilted_S["is_viable"]
    else:
        shock_i = pd.Series(False, index=tilted_S.index)

    tilted_S["shock_i"] = shock_i.astype(int)
    tilted_S["alpha_eff"] = alpha
    tilted_S.loc[tilted_S["shock_i"] == 1, "alpha_eff"] = alpha_shock

    viable_mask = tilted_S["is_viable"]
    base_share = tilted_S.loc[viable_mask, "pred_vote_share_national"].clip(lower=0)

    tilt_mult = np.exp(
        tilted_S.loc[viable_mask, "alpha_eff"].values
        * tilted_S.loc[viable_mask, "z_social"].values
    )
    wS = base_share.values * tilt_mult
    wS = wS / wS.sum()

    tilted_S.loc[viable_mask, "share_after"] += M * wS
    tilted_S["share_after"] = tilted_S["share_after"].clip(lower=0)
    tilted_S["share_after"] = tilted_S["share_after"] / tilted_S["share_after"].sum() * 100.0

    social_shock_out = tilted_S[["party_id", "share_after", "z_social", "shock_i"]].rename(
        columns={"share_after": "pred_vote_share_national"}
    )
    ensure_dirs_for_file(cfg["out_social_shock"])
    social_shock_out.to_csv(cfg["out_social_shock"], index=False)

    
    # Comparison table
    
    compare = baseline.rename(columns={"pred_vote_share_national": "share_baseline"}).copy()

    compare = compare.merge(
        struct_out.rename(columns={"pred_vote_share_national": "share_A"}),
        on="party_id", how="left"
    )
    compare = compare.merge(
        social_out[["party_id", "pred_vote_share_national"]].rename(columns={"pred_vote_share_national": "share_B"}),
        on="party_id", how="left"
    )
    compare = compare.merge(
        social_shock_out[["party_id", "pred_vote_share_national", "shock_i", "z_social"]].rename(
            columns={"pred_vote_share_national": "share_Bshock"}
        ),
        on="party_id", how="left"
    )

    compare["delta_A"] = compare["share_A"] - compare["share_baseline"]
    compare["delta_B"] = compare["share_B"] - compare["share_baseline"]
    compare["delta_Bshock"] = compare["share_Bshock"] - compare["share_baseline"]

    compare = compare.sort_values("share_baseline", ascending=False).reset_index(drop=True)
    ensure_dirs_for_file(cfg["out_compare"])
    compare.to_csv(cfg["out_compare"], index=False)

    
    # Seats allocation for each scenario
    
    seats_baseline = allocate_pr_seats(
        baseline, seats=110, threshold_pct=3.0, share_col="pred_vote_share_national"
    )
    seats_struct = allocate_pr_seats(
        struct_out, seats=110, threshold_pct=3.0, share_col="pred_vote_share_national"
    )
    seats_social = allocate_pr_seats(
        social_out[["party_id", "pred_vote_share_national"]],
        seats=110, threshold_pct=3.0, share_col="pred_vote_share_national"
    )
    seats_social_shock = allocate_pr_seats(
        social_shock_out[["party_id", "pred_vote_share_national"]],
        seats=110, threshold_pct=3.0, share_col="pred_vote_share_national"
    )

    ensure_dirs_for_file(cfg["out_seats_baseline"])
    ensure_dirs_for_file(cfg["out_seats_struct"])
    ensure_dirs_for_file(cfg["out_seats_social"])
    ensure_dirs_for_file(cfg["out_seats_social_shock"])

    seats_baseline.to_csv(cfg["out_seats_baseline"], index=False)
    seats_struct.to_csv(cfg["out_seats_struct"], index=False)
    seats_social.to_csv(cfg["out_seats_social"], index=False)
    seats_social_shock.to_csv(cfg["out_seats_social_shock"], index=False)

    # Diagnostics

    def topk_sum(df_shares: pd.DataFrame, col: str, k: int) -> float:
        return df_shares.sort_values(col, ascending=False).head(k)[col].sum()

    top_base = topk_sum(baseline, "pred_vote_share_national", top_k)
    top_A = topk_sum(struct_out, "pred_vote_share_national", top_k)
    top_B = topk_sum(social_out, "pred_vote_share_national", top_k)
    top_Bs = topk_sum(social_shock_out, "pred_vote_share_national", top_k)

    print(f"Top-{top_k} baseline sum: {top_base:.2f}%")
    print(f"Top-{top_k} Option A sum:  {top_A:.2f}%  (c={c})")
    print(f"Top-{top_k} Option B sum:  {top_B:.2f}%  (c={c}, alpha={alpha})")
    print(f"Top-{top_k} B+Shock sum:   {top_Bs:.2f}%  (c={c}, alpha={alpha}, alpha_shock={alpha_shock}, z_thr={z_thr})")

    shock_parties = social_shock_out.loc[social_shock_out["shock_i"] == 1, ["party_id", "z_social"]]
    if len(shock_parties) > 0:
        print("\nShock parties:")
        print(shock_parties.sort_values("z_social", ascending=False).to_string(index=False))
    else:
        print("\nShock parties: none (criteria not met)")

    print("\nWrote outputs:")
    print(" -", cfg["out_baseline"])
    print(" -", cfg["out_struct"])
    print(" -", cfg["out_social"])
    print(" -", cfg["out_social_shock"])
    print(" -", cfg["out_compare"])
    print(" -", cfg["out_seats_baseline"])
    print(" -", cfg["out_seats_struct"])
    print(" -", cfg["out_seats_social"])
    print(" -", cfg["out_seats_social_shock"])


if __name__ == "__main__":
    main()
