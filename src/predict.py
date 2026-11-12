"""Write a Zindi submission, either from a training run or from saved models.

Usage (after training):
    python -m src.predict                     # uses models/artifacts.joblib
    python -m src.predict --models lgbm       # override the ensemble with a single model
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, RAW_DIR, SUBMISSIONS_DIR, THRESHOLD
from .data import load_dataset
from .features import build_features


def make_submission(sample: pd.DataFrame, id_col: str, test_ids: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    """Fill the sample submission's columns, in the sample's row order.

    Columns with "F1" in the name get the 0/1 prediction at the fixed 0.5 threshold; every
    other target column gets the raw probability.
    """
    proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    by_id = pd.Series(proba, index=test_ids.values)
    if by_id.index.duplicated().any():
        raise ValueError("Duplicate IDs in the test set.")

    sub = sample[[id_col]].copy()
    p = by_id.reindex(sub[id_col]).to_numpy()
    if np.isnan(p).any():
        raise ValueError(f"{np.isnan(p).sum()} submission IDs have no prediction.")

    for col in sample.columns:
        if col == id_col:
            continue
        if "f1" in col.lower():
            sub[col] = (p >= THRESHOLD).astype(int)
        else:
            sub[col] = p
    if len(sample.columns) == 2:
        print(f"[submit] NOTE: the sample has a single target column '{sample.columns[1]}'; "
              f"it was filled with probabilities. Check the competition's instructions.")
    return sub


def save_submission(sub: pd.DataFrame, tag: str, data_dir: Path) -> Path:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    is_real = Path(data_dir).resolve() == RAW_DIR.resolve()
    prefix = "submission" if is_real else f"NOT_FOR_ZINDI_{Path(data_dir).name}"
    path = SUBMISSIONS_DIR / f"{prefix}_{tag}_{stamp}.csv"
    sub.to_csv(path, index=False)
    if is_real:
        latest = SUBMISSIONS_DIR / "submission_latest.csv"
        sub.to_csv(latest, index=False)
        print(f"[submit] wrote {path} ({len(sub)} rows) and {latest.name}")
    else:
        print(f"[submit] wrote {path} ({len(sub)} rows). Not from data/raw, so do not upload it.")
    print(sub.head().to_string(index=False))
    return path


def predict_from_artifacts(artifacts_path: Path, models: list[str] | None = None) -> Path:
    art = joblib.load(artifacts_path)
    ds = load_dataset(Path(art["data_dir"]), art["train_path"], art["test_path"], art["sample_path"],
                      art["target"], art["id_col"])
    fs = build_features(ds.train, ds.test, ds.id_col, ds.target)
    if list(fs.X_test.columns) != art["feature_columns"]:
        raise ValueError("Features differ from training time. Retrain with `python -m src.train`.")

    names = models or art["ensemble"]
    preds = []
    for name in names:
        fold_models = art["fold_models"][name]
        preds.append(np.mean([m.predict_proba(fs.X_test) for m in fold_models], axis=0))
    proba = np.mean(preds, axis=0)
    sub = make_submission(ds.sample, ds.id_col, ds.test[ds.id_col], proba)
    return save_submission(sub, "+".join(names), art["data_dir"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=MODELS_DIR / "artifacts.joblib")
    ap.add_argument("--models", help="comma-separated model names to blend instead of the saved ensemble")
    args = ap.parse_args()
    predict_from_artifacts(args.artifacts, args.models.split(",") if args.models else None)


if __name__ == "__main__":
    main()
