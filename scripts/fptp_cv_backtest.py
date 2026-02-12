from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.models.fptp.cv_backtest import run_groupkfold_cv_backtest, CVConfig

TRAIN_PATH = Path("data/processed/fptp_train_model_2079.csv")
OUT_DIR = Path("data/outputs")

def main():
    df = pd.read_csv(TRAIN_PATH)

    categorical = ["gender", "qualification"]
    numeric = [
        # candidate
        "age",

        # district demo
        "Literacy Rate 2078", "edu_below_slc", "edu_slc_to_inter", "edu_grad_or_higher",

        # candidate lag (year-agnostic if you used the renaming script)
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

    cfg = CVConfig(
        target="won_seat",
        outer_group="seat_id", 
        outer_splits=5,
        inner_splits=4,
        random_state=42,
    )

    out = run_groupkfold_cv_backtest(
        df_train=df,
        numeric_features=numeric,
        categorical_features=categorical,
        cfg=cfg,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / f"fptp_cv_metrics_{cfg.outer_group}.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8"
    )

    # Save OOF predictions as CSV for each winner (1 or 2 files)
    for winner_name, records in out["oof_predictions_by_winner"].items():
        oof_df = pd.DataFrame(records)
        oof_df.to_csv(
            OUT_DIR / f"fptp_cv_oof_{cfg.outer_group}_{winner_name}.csv",
            index=False,
            encoding="utf-8"
        )

    print("[OK] CV backtest complete.")
    print(f"Winner (mean logloss): {out['winner_by_mean_logloss']}")
    print(f"Winner (mean seat_acc): {out['winner_by_mean_seat_acc']}")
    print("Overall OOF seat_acc by winner:")
    for k, v in out["overall_seat_acc_by_winner"].items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
