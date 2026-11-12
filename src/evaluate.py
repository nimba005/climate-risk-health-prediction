"""The competition metric plus a few diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from .config import AUC_WEIGHT, F1_WEIGHT, LEAKAGE_AUC_WARNING, SEED, THRESHOLD


def competition_score(y_true, proba) -> dict:
    """Weighted score used on the leaderboard: 0.6 * F1 (at 0.5) + 0.4 * ROC-AUC."""
    proba = np.asarray(proba)
    pred = (proba >= THRESHOLD).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    auc = roc_auc_score(y_true, proba)
    return {
        "score": F1_WEIGHT * f1 + AUC_WEIGHT * auc,
        "f1": f1,
        "auc": auc,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "pred_positive_rate": float(pred.mean()),
    }


def format_scores(s: dict) -> str:
    return (f"score={s['score']:.5f}  F1={s['f1']:.5f}  AUC={s['auc']:.5f}  "
            f"precision={s['precision']:.3f}  recall={s['recall']:.3f}  "
            f"predicted-positive={s['pred_positive_rate']:.3f}")


def leakage_check(X: pd.DataFrame, y: pd.Series, cat_cols: list[str]) -> list[tuple[str, float]]:
    """Cross-validated AUC of each feature on its own, using a shallow tree.

    A single column that almost perfectly predicts the target is usually a leak (for example
    a cause-of-death code the label was derived from). It will score well in CV but is worth
    understanding before trusting it.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = []
    for col in X.columns:
        s = X[col]
        x = (s.cat.codes if col in cat_cols else s.fillna(-999)).to_numpy().reshape(-1, 1)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=SEED)
        p = cross_val_predict(tree, x, y, cv=cv, method="predict_proba")[:, 1]
        results.append((col, roc_auc_score(y, p)))
    results.sort(key=lambda t: -t[1])

    flagged = [(c, a) for c, a in results if a >= LEAKAGE_AUC_WARNING]
    for col, auc in flagged:
        print(f"[leakage] WARNING: '{col}' alone reaches AUC {auc:.4f}. Check it is not derived from the target.")
    print("[leakage] strongest single features: "
          + ", ".join(f"{c}={a:.3f}" for c, a in results[:8]))
    return results
