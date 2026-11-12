"""Generate a fake dataset shaped like the competition's, for testing the pipeline end to end.

This is NOT competition data. It lives in data/synthetic/ so it never mixes with data/raw/.
    python -m src.make_synthetic
    python -m src.train --data-dir data/synthetic --quick
"""
import numpy as np
import pandas as pd

from .config import SEED, SYNTHETIC_DIR


def generate(n: int = 6000, test_frac: float = 0.3, seed: int = SEED):
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West", "Central", "Coast", "Lake", "Rift"]
    region = rng.choice(regions, n)
    region_heat = pd.Series(region).map(dict(zip(regions, rng.normal(0, 3, len(regions))))).to_numpy()
    month = rng.integers(1, 13, n)
    year = rng.integers(2015, 2024, n)
    day = rng.integers(1, 29, n)
    age = np.clip(rng.gamma(2.2, 22, n), 0, 105).round()
    temp = 26 + 6 * np.sin(2 * np.pi * (month - 3) / 12) + region_heat + rng.normal(0, 2, n)
    rain = np.clip(rng.gamma(1.5, 40, n) * (1 + np.cos(2 * np.pi * (month - 4) / 12)), 0, None)
    humidity = np.clip(50 + 0.15 * rain + rng.normal(0, 10, n), 5, 100)
    sex = rng.choice(["Male", "Female"], n)
    place = rng.choice(["Hospital", "Home", "Health centre", "Other"], n, p=[0.5, 0.35, 0.1, 0.05])
    education = rng.choice(["None", "Primary", "Secondary", "Tertiary", None], n)

    logit = (-1.4 + 0.05 * (temp - 26) ** 2 / 4 + 0.012 * (rain - 60) / 2
             + 0.9 * ((age < 5) | (age > 70)) + 0.4 * (place == "Home") - 0.3 * (education == "Tertiary")
             + rng.normal(0, 0.8, n))
    target = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "ID": [f"ID_{i:06d}" for i in rng.permutation(n)],
        "Age": age,
        "Sex": sex,
        "Region": region,
        "Date_of_death": [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(year, month, day)],
        "Place_of_death": place,
        "Education": education,
        "Max_temperature_C": temp.round(1).astype(str),   # number stored as text
        "Rainfall_mm": rain.round(1),
        "Humidity": humidity.round(1),
        "Country": "Kenya",                               # constant column
        "Record_note": [f"note-{i}" for i in range(n)],   # near-unique text
        "Climate_Sensitive": target,
    })
    df.loc[rng.random(n) < 0.05, "Rainfall_mm"] = np.nan
    df.loc[rng.random(n) < 0.03, "Age"] = np.nan

    is_test = rng.random(n) < test_frac
    train = df[~is_test].reset_index(drop=True)
    test = df[is_test].drop(columns=["Climate_Sensitive"]).reset_index(drop=True)
    sample = pd.DataFrame({"ID": test["ID"].sample(frac=1, random_state=seed).values,
                           "TargetF1": 0, "TargetRAUC": 0.0})
    return train, test, sample


def main():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample = generate()
    train.to_csv(SYNTHETIC_DIR / "Train.csv", index=False)
    test.to_csv(SYNTHETIC_DIR / "Test.csv", index=False)
    sample.to_csv(SYNTHETIC_DIR / "SampleSubmission.csv", index=False)
    print(f"wrote synthetic data to {SYNTHETIC_DIR}: train {train.shape}, test {test.shape}")


if __name__ == "__main__":
    main()
