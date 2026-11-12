"""Train, cross-validate and blend models, then write a submission.

Usage:
    python -m src.train                                  # full run on data/raw
    python -m src.train --quick                          # fast check that everything works
    python -m src.train --models lgbm,catboost           # choose models
    python -m src.train --target Climate_Sensitive       # if the target can't be inferred
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ALL_MODELS, MODELS_DIR, N_FOLDS, RAW_DIR, REPORTS_DIR, SEED
from .data import load_dataset
from .evaluate import competition_score, format_scores, leakage_check
from .features import build_features
from .models import blend, make_model
from .predict import make_submission, save_submission


def run_cv(name, X, y, X_test, cat_cols, balanced, n_folds, quick):
    """K-fold CV: out-of-fold predictions for scoring, fold-averaged predictions for test."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_models = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = make_model(name, cat_cols, balanced, quick)
        model.fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(X_test) / n_folds
        fold_models.append(model)
    return oof, test_pred, fold_models


def choose_balance(X, y, cat_cols, n_folds):
    """Class weighting shifts probabilities around the fixed 0.5 cut-off, which changes F1.

    The threshold itself is never tuned (competition rules); this only decides how the model
    is trained, using cross-validated score.
    """
    results = {}
    for balanced in (False, True):
        oof, _, _ = run_cv("lgbm", X, y, X.iloc[:1], cat_cols, balanced, n_folds, quick=True)
        results[balanced] = competition_score(y, oof)
        print(f"[balance] class_weight={'balanced' if balanced else 'none':8s} {format_scores(results[balanced])}")
    choice = results[True]["score"] > results[False]["score"]
    print(f"[balance] using {'balanced' if choice else 'unweighted'} training")
    return choice


def best_blend(oofs: dict[str, np.ndarray], y) -> tuple[list[str], dict]:
    best_names, best = None, None
    for k in range(1, len(oofs) + 1):
        for names in itertools.combinations(oofs, k):
            s = competition_score(y, blend(oofs, names))
            if best is None or s["score"] > best["score"] + 1e-6:
                best_names, best = list(names), s
    return best_names, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--target")
    ap.add_argument("--id-col")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--balance", choices=["auto", "none", "balanced"], default="auto")
    ap.add_argument("--quick", action="store_true", help="fewer trees and higher learning rate, for smoke tests")
    ap.add_argument("--skip-leakage-check", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset(args.data_dir, args.train, args.test, args.sample, args.target, args.id_col)
    fs = build_features(ds.train, ds.test, ds.id_col, ds.target)
    X, X_test, y = fs.X_train, fs.X_test, ds.y

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    single_feature_auc = [] if args.skip_leakage_check else leakage_check(X, y, fs.cat_cols)

    balanced = {"none": False, "balanced": True}.get(args.balance)
    if balanced is None:
        balanced = choose_balance(X, y, fs.cat_cols, args.folds)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(model_names) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown models {unknown}; choose from {ALL_MODELS}")

    oofs, test_preds, fold_models, scores = {}, {}, {}, {}
    for name in model_names:
        t = time.time()
        oofs[name], test_preds[name], fold_models[name] = run_cv(
            name, X, y, X_test, fs.cat_cols, balanced, args.folds, args.quick)
        scores[name] = competition_score(y, oofs[name])
        print(f"[cv] {name:9s} {format_scores(scores[name])}  ({time.time() - t:.0f}s)")

    ensemble, ens_score = best_blend(oofs, y)
    print(f"[cv] best blend {ensemble}: {format_scores(ens_score)}")

    # Feature importance from LightGBM if it was trained (useful for EDA and the write-up).
    if "lgbm" in fold_models:
        imp = pd.DataFrame({
            "feature": X.columns,
            "gain": np.mean([m.model.booster_.feature_importance("gain") for m in fold_models["lgbm"]], axis=0),
        }).sort_values("gain", ascending=False)
        imp.to_csv(REPORTS_DIR / "feature_importance_lgbm.csv", index=False)
        print("[cv] top features: " + ", ".join(imp.feature.head(10)))

    oof_df = pd.DataFrame({ds.id_col: ds.train[ds.id_col], "y": y, **{f"oof_{k}": v for k, v in oofs.items()}})
    oof_df.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)

    report = {
        "id_col": ds.id_col,
        "target": ds.target,
        "label_mapping": {str(k): v for k, v in ds.label_mapping.items()},
        "n_train": len(X), "n_test": len(X_test), "positive_rate": float(y.mean()),
        "n_features": X.shape[1], "categorical": fs.cat_cols, "dropped_columns": fs.dropped,
        "balanced_training": balanced, "quick": args.quick, "folds": args.folds,
        "models": scores, "ensemble": ensemble, "ensemble_scores": ens_score,
        "single_feature_auc_top10": single_feature_auc[:10],
    }
    (REPORTS_DIR / "cv_report.json").write_text(json.dumps(report, indent=2, default=float))

    joblib.dump({
        "fold_models": fold_models, "ensemble": ensemble,
        "feature_columns": list(X.columns), "cat_cols": fs.cat_cols,
        "id_col": ds.id_col, "target": ds.target, "balanced": balanced,
        "data_dir": str(args.data_dir),
        "train_path": str(args.train) if args.train else None,
        "test_path": str(args.test) if args.test else None,
        "sample_path": str(args.sample) if args.sample else None,
    }, MODELS_DIR / "artifacts.joblib")
    print(f"[save] models -> {MODELS_DIR / 'artifacts.joblib'}, report -> {REPORTS_DIR / 'cv_report.json'}")

    sub = make_submission(ds.sample, ds.id_col, ds.test[ds.id_col], blend(test_preds, ensemble))
    save_submission(sub, "+".join(ensemble), args.data_dir)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
