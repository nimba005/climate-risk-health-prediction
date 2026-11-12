"""Locate and load the Zindi files, and work out the ID and target columns."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

POSITIVE_LABELS = {"1", "yes", "y", "true", "t", "positive", "climate-sensitive", "climate sensitive", "sensitive"}
NEGATIVE_LABELS = {"0", "no", "n", "false", "f", "negative", "non-climate-sensitive", "not climate-sensitive",
                   "not sensitive", "non-sensitive"}


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    id_col: str
    target: str
    y: pd.Series
    label_mapping: dict = field(default_factory=dict)


def _find_csv(data_dir: Path, kind: str) -> Path:
    csvs = sorted(data_dir.glob("*.csv"))
    names = {p: p.stem.lower() for p in csvs}
    if kind == "sample":
        hits = [p for p, n in names.items() if "sample" in n or "submission" in n]
    else:
        hits = [p for p, n in names.items() if kind in n and "sample" not in n and "submission" not in n]
    if len(hits) != 1:
        found = ", ".join(p.name for p in csvs) or "no CSV files"
        raise FileNotFoundError(
            f"Expected exactly one '{kind}' CSV in {data_dir}, found {len(hits)} ({found}). "
            f"Pass the file explicitly with --{kind}."
        )
    return hits[0]


def _encode_target(raw: pd.Series) -> tuple[pd.Series, dict]:
    values = raw.dropna().unique()
    if len(values) != 2:
        raise ValueError(f"Target '{raw.name}' must have exactly 2 classes, found {len(values)}: {values[:10]}")

    if pd.api.types.is_numeric_dtype(raw) and set(values) <= {0, 1}:
        return raw.astype(int), {0: 0, 1: 1}

    as_text = {v: str(v).strip().lower() for v in values}
    positives = [v for v, t in as_text.items() if t in POSITIVE_LABELS]
    negatives = [v for v, t in as_text.items() if t in NEGATIVE_LABELS]
    if len(positives) == 1 and len(negatives) == 1:
        mapping = {negatives[0]: 0, positives[0]: 1}
    else:
        # Fall back to sorted order and say so, so the user can check it.
        low, high = sorted(values, key=str)
        mapping = {low: 0, high: 1}
        print(f"[data] WARNING: could not tell which target value is positive; using {mapping}. "
              f"Check this is right.")
    return raw.map(mapping).astype(int), mapping


def _merge_extra_features(data_dir: Path, used: set[Path], train: pd.DataFrame, test: pd.DataFrame,
                          id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join any other per-ID CSV in the data folder (e.g. climate_features.csv) onto train and test.

    A file qualifies if it has the ID column, one row per ID, and covers every train and test ID.
    Data dictionaries and other files are ignored. Columns already in train are not duplicated.
    """
    for path in sorted(data_dir.glob("*.csv")):
        if path.resolve() in used:
            continue
        try:
            extra = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if id_col not in extra.columns or not extra[id_col].is_unique:
            continue
        ids = set(extra[id_col])
        if not (set(train[id_col]) <= ids and set(test[id_col]) <= ids):
            print(f"[data] skipping {path.name}: does not cover every train and test ID")
            continue
        new_cols = [c for c in extra.columns if c != id_col and c not in train.columns]
        if not new_cols:
            continue
        extra = extra[[id_col] + new_cols]
        train = train.merge(extra, on=id_col, how="left")
        test = test.merge(extra, on=id_col, how="left")
        print(f"[data] merged {len(new_cols)} columns from {path.name}")
    return train, test


def load_dataset(data_dir: Path, train_path: Path | None = None, test_path: Path | None = None,
                 sample_path: Path | None = None, target: str | None = None,
                 id_col: str | None = None) -> Dataset:
    data_dir = Path(data_dir)
    train_path = Path(train_path) if train_path else _find_csv(data_dir, "train")
    test_path = Path(test_path) if test_path else _find_csv(data_dir, "test")
    sample_path = Path(sample_path) if sample_path else _find_csv(data_dir, "sample")

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    sample = pd.read_csv(sample_path, low_memory=False)
    print(f"[data] train {train.shape} <- {train_path.name}")
    print(f"[data] test  {test.shape} <- {test_path.name}")
    print(f"[data] sample submission {sample.shape} <- {sample_path.name}, columns {list(sample.columns)}")

    if id_col is None:
        id_col = sample.columns[0]
    for name, frame in (("train", train), ("test", test)):
        if id_col not in frame.columns:
            raise KeyError(f"ID column '{id_col}' not found in {name}. Pass it with --id-col.")

    if target is None:
        candidates = [c for c in train.columns if c not in test.columns and c != id_col]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not infer the target column: train-only columns are {candidates}. "
                f"Pass it with --target."
            )
        target = candidates[0]
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train.")

    n_missing = train[target].isna().sum()
    if n_missing:
        print(f"[data] dropping {n_missing} train rows with a missing target")
        train = train[train[target].notna()].reset_index(drop=True)

    y, mapping = _encode_target(train[target])
    print(f"[data] id='{id_col}', target='{target}', positive rate={y.mean():.3f} "
          f"({y.sum()} of {len(y)})")

    # Any other train-only columns would leak information that the test set does not have.
    extra = [c for c in train.columns if c not in test.columns and c not in (target, id_col)]
    if extra:
        print(f"[data] dropping train-only columns not present in test: {extra}")
        train = train.drop(columns=extra)

    used = {Path(p).resolve() for p in (train_path, test_path, sample_path)}
    train, test = _merge_extra_features(data_dir, used, train, test, id_col)

    missing_ids = set(sample[id_col]) - set(test[id_col])
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} sample-submission IDs are not in the test file.")

    return Dataset(train=train, test=test, sample=sample, id_col=id_col, target=target, y=y,
                   label_mapping=mapping)
