from __future__ import annotations
import joblib

from src.config.paths import PATHS
from src.utils.io import read_csv, write_csv
from src.models.pr.predict import predict_pr_2082
from src.pr.aggregate import add_predicted_votes, national_vote_share

INFER_PATH = PATHS.processed / "pr_infer_2082.csv"
MODEL_PATH = PATHS.artifacts / "pr_final_model.joblib"

def main():
    df_infer = read_csv(INFER_PATH)
    model = joblib.load(MODEL_PATH)

    df_pred = predict_pr_2082(df_infer, model)
    df_pred = add_predicted_votes(df_pred)

    nat = national_vote_share(df_pred)

    write_csv(df_pred, PATHS.outputs / "pr_predictions_district_2082.csv")
    write_csv(nat, PATHS.outputs / "pr_predictions_national_2082.csv")

    print(nat.head(15))

if __name__ == "__main__":
    main()
