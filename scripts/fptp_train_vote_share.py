from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.fptp.train_vote_share import (
    load_vote_share_winners,
    load_train_cfg_from_backtest,
    train_and_save_vote_share_winners,
)

# Paths
TRAIN_CSV = Path("data/processed/fptp_train_model_2079.csv")
BACKTEST_METRICS_JSON = Path("data/outputs/fptp_vote_share_backtest_metrics.json")
ARTIFACTS_DIR = Path("models/artifacts")


def main() -> None:
    FEATURES = [
        "age",
        "Literacy Rate 2078", "edu_below_slc", "edu_slc_to_inter", "edu_grad_or_higher",
        "cand_vote_share_prev", "cand_rank_prev", "cand_won_prev", "missing_cand_vote_share_prev",
        "party_vote_share_prev", "party_rank_prev", "party_won_prev", "party_contested_prev",
        "pr_vote_share_prev", "pr_rank_prev", "missing_pr_vote_share_prev",
        "enp_prev", "num_candidates_prev", "margin_prev", "top1_share_prev", "top2_share_prev",
        "is_independent", "is_alternative_party", "is_youth_favoured_party",
        "has_national_leader_presence", "is_old_established_party", "was_in_ousted_government",
    ]

    df_train = pd.read_csv(TRAIN_CSV)

    winners = load_vote_share_winners(BACKTEST_METRICS_JSON)
    if not winners:
        raise ValueError("No winners found in backtest metrics JSON.")

    cfg = load_train_cfg_from_backtest(BACKTEST_METRICS_JSON)

    result = train_and_save_vote_share_winners(
        df_train=df_train,
        features=FEATURES,
        cfg=cfg,
        winners=winners,
        artifacts_dir=ARTIFACTS_DIR,
        artifact_prefix="fptp_vote_share",
    )

    print("[OK] Vote-share winners trained on full data.")
    print("Winners:", result["winners"])
    print("Model paths:")
    for k, v in result["model_paths"].items():
        print(f"  - {k}: {v}")
    print("Manifest:", result["manifest_path"])
    print("Best params:", result["best_params_path"])


if __name__ == "__main__":
    main()
