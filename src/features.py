"""Turn the raw train/test tables into model-ready feature matrices.

Every step here is unsupervised (it never looks at the target), so it is safe to fit it on
train and test together. That keeps category sets and encodings identical across both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RARE_CATEGORY_MIN_COUNT

DATE_NAME = re.compile(r"date|time|dob|day_of|timestamp", re.I)
DATE_VALUE = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")
MISSING = "__missing__"
RARE = "__rare__"


@dataclass
class FeatureSet:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    cat_cols: list[str]
    num_cols: list[str]
    dropped: dict[str, str]


def _looks_like_date(name: str, s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    matches = sample.str.match(DATE_VALUE).mean()
    return matches > 0.9 or (DATE_NAME.search(name) is not None and matches > 0.5)


def _date_parts(name: str, s: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    return pd.DataFrame({
        f"{name}_year": dt.dt.year,
        f"{name}_month": dt.dt.month,
        f"{name}_dayofyear": dt.dt.dayofyear,
        f"{name}_dayofweek": dt.dt.dayofweek,
        # Cyclical month encoding lets models see December and January as neighbours (seasonality).
        f"{name}_month_sin": np.sin(2 * np.pi * dt.dt.month / 12),
        f"{name}_month_cos": np.cos(2 * np.pi * dt.dt.month / 12),
    }, index=s.index)


def build_features(train: pd.DataFrame, test: pd.DataFrame, id_col: str, target: str) -> FeatureSet:
    n_train = len(train)
    full = pd.concat([train.drop(columns=[target]), test], axis=0, ignore_index=True)
    full = full.drop(columns=[id_col])
    dropped: dict[str, str] = {}
    out: dict[str, pd.Series] = {}
    cat_cols: list[str] = []

    for col in full.columns:
        s = full[col]
        nunique = s.nunique(dropna=True)

        if nunique <= 1:
            dropped[col] = "constant"
            continue

        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)

        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
            s = s.astype("object")
            # Numbers stored as text (e.g. "23.5") become real numbers.
            as_num = pd.to_numeric(s, errors="coerce")
            if as_num.notna().sum() >= 0.95 * s.notna().sum():
                out[col] = as_num
                continue
            if _looks_like_date(col, s):
                for k, v in _date_parts(col, s).items():
                    out[k] = v
                continue
            # Near-unique text (another identifier or free text) cannot generalise.
            if nunique > 0.9 * s.notna().sum():
                dropped[col] = "near-unique text"
                continue

            text = s.astype(str).str.strip().where(s.notna(), MISSING)
            counts = text.map(text.value_counts())
            text = text.where(counts >= RARE_CATEGORY_MIN_COUNT, RARE)
            out[col] = text.astype("category")
            out[f"{col}_freq"] = counts.astype(float) / len(text)
            cat_cols.append(col)
        else:
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)

    X = pd.DataFrame(out)
    X["n_missing"] = full.isna().sum(axis=1).astype(float)

    # Exact duplicate columns add nothing but noise and training time.
    num_part = X.drop(columns=cat_cols)
    dup_mask = num_part.T.duplicated()
    for col in num_part.columns[dup_mask.values]:
        dropped[col] = "duplicate of another column"
    X = X.drop(columns=list(num_part.columns[dup_mask.values]))

    num_cols = [c for c in X.columns if c not in cat_cols]
    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)

    print(f"[features] {X.shape[1]} features ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    if dropped:
        print(f"[features] dropped: {dropped}")
    return FeatureSet(X_train, X_test, cat_cols, num_cols, dropped)
