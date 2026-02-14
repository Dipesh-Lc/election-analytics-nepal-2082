from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.fptp.predict_vote_share import (
    PredictVoteShareConfig,
    load_manifest,
    load_models,
    predict_vote_share_and_append_columns,
)

# Paths
INFER_CSV = Path("data/processed/fptp_infer_model_2082.csv")  
ARTIFACTS_DIR = Path("models/artifacts")
OUT_DIR = Path("data/outputs/fptp_vote_share_predictions")
OUT_PATH = OUT_DIR / "fptp_vote_share_predictions_2082.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_infer = pd.read_csv(INFER_CSV)

    cfg = PredictVoteShareConfig(artifacts_dir=ARTIFACTS_DIR, artifact_prefix="fptp_vote_share")
    manifest = load_manifest(cfg)
    models = load_models(cfg)

    features = manifest["features"]
    seat_col = manifest["group_col"]  # seat_id

    out = predict_vote_share_and_append_columns(
        df_infer=df_infer,
        features=features,
        seat_col=seat_col,
        models=models,
    )

    out.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("[OK] Vote-share prediction complete.")
    print("Saved:", OUT_PATH)
    print("Models used:", list(models.keys()))


if __name__ == "__main__":
    main()
