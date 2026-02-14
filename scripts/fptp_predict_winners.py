from pathlib import Path
import pandas as pd

from src.models.fptp.predict_winners import (
    PredictConfig,
    load_models_from_manifest,
    predict_candidates_and_seat_winners,
)

#Paths
PREDICT_CSV = Path("data/processed/fptp_infer_model_2082.csv")
ARTIFACTS_DIR = Path("models/artifacts")  
OUT_DIR = Path("data/outputs/fptp_predictions_2082")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    #Features
    categorical = ["gender", "qualification"]
    numeric = [
        # candidate
        "age",

        # district demo
        "Literacy Rate 2078", "edu_below_slc", "edu_slc_to_inter", "edu_grad_or_higher",

        # candidate lag
        "cand_vote_share_prev", "cand_rank_prev", "cand_won_prev", "missing_cand_vote_share_prev",

        # party lag
        "party_vote_share_prev", "party_rank_prev", "party_won_prev", "party_contested_prev",
        "pr_vote_share_prev", "pr_rank_prev", "missing_pr_vote_share_prev",

        # seat context lag
        "enp_prev", "num_candidates_prev", "margin_prev", "top1_share_prev", "top2_share_prev",

        # political alignment
        "is_independent", "is_alternative_party", "is_youth_favoured_party",
        "has_national_leader_presence", "is_old_established_party", "was_in_ousted_government",
    ]

    # Load data
    df_pred = pd.read_csv(PREDICT_CSV)

    # Load trained winner models
    cfg = PredictConfig(
        artifacts_dir=ARTIFACTS_DIR,
        artifact_prefix="fptp",
    )
    models = load_models_from_manifest(cfg)


    # Predict
    cand_df, seat_df = predict_candidates_and_seat_winners(
        df_pred=df_pred,
        numeric_features=numeric,
        categorical_features=categorical,
        models=models,
    )

    # Save outputs
    cand_path = OUT_DIR / "fptp_predictions_candidates.csv"
    seat_path = OUT_DIR / "fptp_predictions_seat_winners.csv"

    cand_df.to_csv(cand_path, index=False, encoding="utf-8")
    seat_df.to_csv(seat_path, index=False, encoding="utf-8")

    print("[OK] Prediction complete.")
    print("Saved:", cand_path)
    print("Saved:", seat_path)
    print("Models used:", list(models.keys()))


if __name__ == "__main__":
    main()
