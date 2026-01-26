from __future__ import annotations
import json
import joblib

from src.config.paths import PATHS
from src.utils.io import read_csv
from src.models.pr.train_final import train_final_model

TRAIN_PATH = PATHS.processed / "pr_train_2079.csv"
METRICS_PATH = PATHS.outputs / "pr_backtest_metrics.json"

def main():
    df_train = read_csv(TRAIN_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    winner = metrics["winner"]
    best_params = metrics[winner]["best_params"] if metrics.get(winner) else {}

    model = train_final_model(df_train, winner, best_params)

    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATHS.artifacts / "pr_final_model.joblib")

    # Save feature list 
    from src.config.constants import BASE_FEATURES
    (PATHS.artifacts / "pr_feature_list.json").write_text(
        json.dumps(BASE_FEATURES, indent=2),
        encoding="utf-8"
    )

    print(f"Saved final model: {PATHS.artifacts / 'pr_final_model.joblib'}")

if __name__ == "__main__":
    main()
