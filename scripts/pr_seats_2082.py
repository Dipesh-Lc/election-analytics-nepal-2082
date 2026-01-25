from __future__ import annotations
from src.config.paths import PATHS
from src.utils.io import read_csv, write_csv
from src.pr.seats import allocate_pr_seats
def main():
    nat = read_csv(PATHS.outputs / "pr_predictions_national_2082.csv")
    out = allocate_pr_seats(
        nat,
        seats=110,
        threshold_pct=3.0,
        share_col="pred_vote_share_national"
    )
    write_csv(out, PATHS.outputs / "pr_seats_2082.csv")
    print(out.head(15))

if __name__ == "__main__":
    main()
