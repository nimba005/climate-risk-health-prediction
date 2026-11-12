# Climate Risk and Health Prediction

A machine learning pipeline for the Zindi **Climate Risk and Health Prediction Challenge**: predict whether a recorded death falls into a climate-sensitive category, using demographic, geographic, health and climate data.

**Leaderboard metric:** `0.6 × F1 + 0.4 × ROC-AUC`
- **F1** is scored on `TargetF1`, a 0/1 prediction at the fixed **0.5** threshold. The rules forbid tuning this threshold.
- **ROC-AUC** is scored on `TargetRAUC`, the raw probability.

## What the pipeline does

1. **Loads the data** (`src/data.py`). It finds the train, test and sample-submission CSVs, infers the ID column (the sample's first column) and the target (the only train column missing from test), and converts text labels such as "Yes"/"No" to 1/0.
2. **Builds features** (`src/features.py`):
   - converts numbers stored as text into real numbers
   - splits dates into year, month, day-of-year, weekday and a cyclical month (for seasonality)
   - encodes categories with rare-level grouping and frequency features
   - drops constant columns, near-unique text columns and duplicate columns
3. **Checks for leakage** (`src/evaluate.py`). It warns if any single column predicts the target almost perfectly on its own.
4. **Chooses class weighting.** With a fixed 0.5 threshold, class weighting changes how many positives the model predicts. The pipeline tries both settings with cross-validation (CV) and keeps the better one.
5. **Trains four models** with 5-fold stratified cross-validation: LightGBM, CatBoost, XGBoost and Logistic Regression (a baseline).
6. **Blends the models.** It tests every combination of the models' out-of-fold predictions and keeps the combination with the best competition score.
7. **Writes the submission** in the exact row order of `SampleSubmission.csv`, after checking that every ID has a prediction.

## Setup (Windows / VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run it

1. **Download the data.** On Zindi, open the challenge, then the **Data** tab. Download the train, test and sample-submission CSVs into `data/raw/`.
2. **Optionally explore it.** Open `notebooks/01_exploratory_analysis.ipynb` and run all cells.
3. **Train and create the submission:**
   ```powershell
   python -m src.train            # full run
   python -m src.train --quick    # about 1 minute, to check everything works
   ```
4. **Upload** `submissions/submission_latest.csv` on the challenge's **Submissions** tab.

If the target or ID column cannot be detected automatically, the error message says so. Name them yourself:
```powershell
python -m src.train --target <TargetColumn> --id-col <IdColumn>
```

### Other options
| Command | Purpose |
|---|---|
| `python -m src.train --models lgbm,catboost` | Train only some models |
| `python -m src.train --balance none` | Force class weighting on or off (`none` / `balanced`) instead of `auto` |
| `python -m src.predict` | Rebuild the submission from saved models without retraining |
| `python -m src.predict --models xgb` | Submit a different model or blend from the last training run |
| `python -m src.make_synthetic` then `python -m src.train --data-dir data/synthetic --quick` | Test the pipeline on fake data |

Files built from anything other than `data/raw/` are named `NOT_FOR_ZINDI_*.csv`, and they never replace `submission_latest.csv`.

## Outputs

| Path | Contents |
|---|---|
| `submissions/submission_latest.csv` | The file to upload |
| `reports/cv_report.json` | CV scores per model, chosen blend, dropped columns and leakage check |
| `reports/feature_importance_lgbm.csv` | Which features the model relies on |
| `reports/oof_predictions.csv` | Out-of-fold predictions, for error analysis |
| `models/artifacts.joblib` | All trained fold models |

## Improving your score

- **Trust local CV.** Compare the CV score in `cv_report.json` with the leaderboard score. If they match, improve against CV, not the public leaderboard.
- **Read the leakage warnings.** A column that alone reaches AUC ≈ 1.0 (for example a cause-of-death code) may define the label. Check whether it is legitimately in the test data.
- **Add domain features** in `src/features.py`. Examples: age groups (under 5, over 65), temperature × age, rainfall anomalies against a region's average, heat-index style combinations.
- **External data.** The rules allow only publicly available climate datasets (for example ERA5 or CHIRPS), and nothing that is not open source.
- **Rules.** No AutoML, and no threshold tuning on `TargetF1`.

## Project structure

```
data/raw/            Zindi files (not committed: competition data may not be shared)
notebooks/           Exploratory analysis
src/
  config.py          Paths and constants
  data.py            Loading and column detection
  features.py        Feature engineering
  models.py          LightGBM / CatBoost / XGBoost / Logistic Regression wrappers
  evaluate.py        Competition metric and leakage check
  train.py           Cross-validation, blending, saving
  predict.py         Submission writing and validation
  make_synthetic.py  Fake data for testing
```
