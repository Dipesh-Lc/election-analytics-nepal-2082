from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.fptp.backtest_vote_share import run_vote_share_seat_holdout_backtest, VoteShareBacktestConfig


TRAIN_PATH = Path("data/processed/fptp_train_model_2079.csv")
OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = OUT_DIR / "fptp_vote_share_backtest_metrics.json"
OOF_OUT = OUT_DIR / "fptp_vote_share_backtest_oof_predictions.csv"


def main():
    df = pd.read_csv(TRAIN_PATH)
    FEATURES = [
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

    cfg = VoteShareBacktestConfig(
        target_col="vote_share",
        group_col="seat_id",
        outer_splits=5,
        inner_splits=4,
        random_state=42,
        use_xgboost=True,
    )

    result = run_vote_share_seat_holdout_backtest(df, FEATURES, cfg)

    # Save metrics
    metrics_payload = {
        "config": result["config"],
        "features": result["features"],
        "per_fold_results": result["per_fold_results"],
        "summary_by_model": result["summary_by_model"],
        "winners": result["winners"],
        "winner_by_error": result["winner_by_error"],
        "winner_by_seat_acc": result["winner_by_seat_acc"],
    }

    METRICS_OUT.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Save OOF preds
    result["oof_predictions"].to_csv(OOF_OUT, index=False, encoding="utf-8")

    print(f"[OK] wrote: {METRICS_OUT}")
    print(f"[OK] wrote: {OOF_OUT}")
    print("[WINNERS]", result["winners"])



if __name__ == "__main__":
    main()
