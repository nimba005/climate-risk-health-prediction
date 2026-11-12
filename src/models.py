"""Model definitions. Each wrapper takes the shared feature frame and handles its own encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import SEED

# Tree counts are fixed rather than early-stopped on the validation fold: early stopping on the
# fold you score inflates CV. The settings are deliberately conservative (small trees, strong
# regularisation) because the training set is small (~3k rows) and noisy; repeated 5-fold CV
# showed these beat deeper trees.


class LGBMModel:
    def __init__(self, cat_cols, balanced, quick=False):
        self.cat_cols = cat_cols
        self.model = LGBMClassifier(
            n_estimators=150 if quick else 600,
            learning_rate=0.04 if quick else 0.01,
            num_leaves=7,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=5.0,
            class_weight="balanced" if balanced else None,
            random_state=SEED,
            verbose=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class CatBoostModel:
    def __init__(self, cat_cols, balanced, quick=False):
        self.cat_cols = cat_cols
        self.model = CatBoostClassifier(
            iterations=150 if quick else 500,
            learning_rate=0.1 if quick else 0.03,
            depth=4,
            l2_leaf_reg=5,
            # One-hot small categoricals; target statistics (CTRs) are much slower and only
            # pay off for high-cardinality columns, which still get them.
            one_hot_max_size=10,
            auto_class_weights="Balanced" if balanced else None,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=0,
        )

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(self._prep(X), y, cat_features=self.cat_cols)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))[:, 1]


class XGBModel:
    def __init__(self, cat_cols, balanced, quick=False):
        self.cat_cols = cat_cols
        self.balanced = balanced
        self.params = dict(
            n_estimators=150 if quick else 400,
            learning_rate=0.05 if quick else 0.02,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=5.0,
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=8,
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if self.balanced else 1.0
        self.model = XGBClassifier(scale_pos_weight=spw, **self.params)
        self.model.fit(X, y, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LogRegModel:
    """Simple linear baseline: a sanity check that the tree models are actually adding value."""

    def __init__(self, cat_cols, balanced, quick=False):
        self.cat_cols = cat_cols
        self.balanced = balanced

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        num_cols = [c for c in X.columns if c not in self.cat_cols]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), self.cat_cols),
        ])
        self.model = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(C=0.5, max_iter=3000,
                                       class_weight="balanced" if self.balanced else None)),
        ])
        self.model.fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))[:, 1]


MODEL_CLASSES = {
    "lgbm": LGBMModel,
    "catboost": CatBoostModel,
    "xgb": XGBModel,
    "logreg": LogRegModel,
}


def make_model(name: str, cat_cols: list[str], balanced: bool, quick: bool = False):
    return MODEL_CLASSES[name](cat_cols, balanced, quick)


def blend(preds: dict[str, np.ndarray], names) -> np.ndarray:
    return np.mean([preds[n] for n in names], axis=0)
