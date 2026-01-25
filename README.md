# election-analytics-nepal-2082 (Proportional Representation Forecast)

A data science project forecasting Nepal’s **2082 snap election** under the **Proportional Representation (PR)** system using historical election results, census demographics, voter roll signals, and social media engagement as an optional “nowcasting” overlay.

This election follows a period of political instability and mass youth protests, making it a rare case to study non-routine electoral behavior, generational shifts in voter preferences, and the emergence of new political forces without long electoral histories.

---

## Project Scope (Current)

This repository focuses **exclusively on Proportional Representation (PR)** outcomes.

The PR pipeline is fully implemented end-to-end:
- District-level vote share modeling
- National vote share aggregation
- Electoral rule enforcement:
  - **3% national threshold**
  - **Sainte-Laguë seat allocation (110 seats)**
- Scenario analysis using social media engagement and backlash signals

> **Note:** Direct constituency (FPTP) modeling is planned for a future iteration once candidate-level datasets are available.

---

## What This Project Does

### 1. District-Level Vote Share Modeling
- Builds a district × party training table using:
  - PR election results (2074, 2079)
  - Census demographics (2078)
  - Voter roll and turnout indicators (2082)
- Models **vote share change** rather than raw vote share to improve stability
- Uses district-holdout backtesting for validation

### 2. National PR Aggregation
- Aggregates district predictions using turnout-weighted sums
- Produces national vote share estimates for 2082

### 3. Electoral Rule Simulation
- Applies Nepal’s PR rules:
  - Parties below **3% national vote share** are excluded
  - Remaining votes are renormalized
  - **Sainte-Laguë** method allocates 110 PR seats

### 4. Scenario Layer (Optional)
To account for non-routine political shocks not captured in historical data, the project includes:
- **Micro-party vote consolidation scenarios**
- **Social media tilt and backlash penalties**
- **Shock-gated amplification** for high-momentum new parties

These scenarios are configurable and intentionally bounded.

---

## Repository Structure

election-analytics-nepal-2082/
├── config/                     # Scenario and consolidation parameters
│
├── data/
│   ├── raw/                    # Source datasets
│   ├── processed/              # Cleaned modeling tables
│   ├── outputs/                # Predictions, seats, metrics, figures
│
├── src/
│   ├── models/
│   │   └── pr/                 # PR model training, backtesting, prediction
│   ├── pr/                     # Aggregation and PR seat allocation logic
│   ├── features/               # Feature engineering & transforms
│   └── utils/                  # IO helpers and validation utilities
│
├── scripts/                    # Reproducible execution scripts
├── notebooks/                  # Analysis & visualization notebooks
│
├── README.md
└── PROJECT_CHARTER.md

---

## How to Run (End-to-End)

Run all commands from the **project root directory**.

### 1) Train the final PR model
Trains the district-level vote share change model using historical PR results and demographic features.

```bash
python scripts/pr_train_final.py
```

### 2) Backtest (district holdout)
Evaluates model performance using district-level holdout validation.

```bash
python scripts/pr_backtest.py
```

### 3) Predict 2082 PR vote shares
Generates district-level and national PR vote share predictions for the 2082 election.

```bash
python scripts/pr_predict_2082.py
```

### 4) Run consolidation, sentiment scenarios, and seat allocation
Applies micro-party consolidation, social media tilt and backlash penalties, optional shock gating, and PR seat allocation using the Sainte-Laguë method.

```bash
python scripts/pr_consolidation_with_sentiment.py
```

---

## Key Outputs

All outputs are written to `data/outputs/`.

**Predictions**
- `pr_predictions_district_2082.csv`
- `pr_predictions_national_2082.csv`

**Scenario Results**
- `pr_national_consolidated_structure.csv`
- `pr_national_consolidated_social_tilt.csv`
- `pr_national_consolidated_social_tilt_shock.csv`

**PR Seat Allocation**
- `pr_seats_*.csv`

**Evaluation**
- `pr_backtest_metrics.json`

---

## Assumptions & Limitations

- The valid turnout for 2082 is assumed to be unchanged from 2079.
- Social media engagement is **not representative** of the electorate and is used only as a bounded scenario overlay.
- Models rely on historical relationships and cannot fully capture real-time campaign dynamics.
- Predictions are **probabilistic and exploratory**, not guarantees of electoral outcomes.

---

## Ethical Note

This project is intended strictly for **analytical and educational purposes**.

- No individual-level data is used
- No targeted political messaging or persuasion is performed
- All social media data is analyzed at an **aggregate level only**

---

## Future Work

- Candidate-level and constituency-level (FPTP) modeling

