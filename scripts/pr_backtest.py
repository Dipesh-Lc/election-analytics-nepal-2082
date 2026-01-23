from __future__ import annotations
import json
import joblib

from src.config.paths import PATHS
from src.utils.io import read_csv, write_csv
from src.models.pr.backtest import run_district_holdout_backtest

TRAIN_PATH = PATHS.processed / "pr_train_2079.csv"

def main():
    df_train = read_csv(TRAIN_PATH)

    best_model, winner_name, metrics, heldout = run_district_holdout_backtest(
        df_train_raw=df_train,
        test_size=0.2,
        inner_splits=5,
        random_state=42
    )

    # Save heldout predictions
    write_csv(heldout, PATHS.outputs / "pr_backtest_heldout_predictions.csv")

    # Save metrics
    (PATHS.outputs / "pr_backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    # Save the trained-on-training-split model (useful for inspection)
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, PATHS.artifacts / f"pr_winner_model_{winner_name}_trained_on_split.joblib")

    print(f"Winner: {winner_name}")
    print(json.dumps(metrics["winner"], indent=2))

if __name__ == "__main__":
    main()
