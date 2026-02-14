from pathlib import Path
import pandas as pd

from src.models.fptp.train_winners_full import (
    load_train_cfg_from_metrics,
    load_winners_from_metrics,
    train_and_save_winners,
)

def main() -> None:
    # Paths
    TRAIN_CSV = Path("data/processed/fptp_train_model_2079.csv")
    METRICS_JSON = Path("data/outputs/fptp_cv_metrics_seat_id.json")
    ARTIFACTS_DIR = Path("models/artifacts")

    # Features 
    categorical = ["gender", "qualification"]
    numeric = [
        # candidate
        "age",

        # district demo
        "Literacy Rate 2078", "edu_below_slc", "edu_slc_to_inter", "edu_grad_or_higher",

        # candidate lag 
        "cand_vote_share_prev", "cand_rank_prev", "cand_won_prev","missing_cand_vote_share_prev",

        # party lag
        "party_vote_share_prev", "party_rank_prev", "party_won_prev", "party_contested_prev",
        "pr_vote_share_prev", "pr_rank_prev","missing_pr_vote_share_prev",

        # seat context lag
        "enp_prev", "num_candidates_prev", "margin_prev", "top1_share_prev", "top2_share_prev",

        # political alignment
        "is_independent", "is_alternative_party", "is_youth_favoured_party",
        "has_national_leader_presence", "is_old_established_party", "was_in_ousted_government",
    ]

    df_train = pd.read_csv(TRAIN_CSV)

    winners = load_winners_from_metrics(METRICS_JSON)
    train_cfg = load_train_cfg_from_metrics(METRICS_JSON)

    result = train_and_save_winners(
        df_train=df_train,
        numeric_features=numeric,
        categorical_features=categorical,
        winners=winners,
        cfg=train_cfg,
        artifacts_dir=ARTIFACTS_DIR,
        artifact_prefix="fptp",
    )

    print("[OK] Trained winners:", result["winners"])
    print("Saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
