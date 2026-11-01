"""Train a station/year water pollutant regression baseline from a local CSV."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

POLLUTANTS = ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]

def train(csv_path, output_path):
    data = pd.read_csv(csv_path, sep=";", dtype={"id": str})
    missing = set(["id", "date"] + POLLUTANTS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data["year"] = pd.to_datetime(data["date"], dayfirst=True, errors="raise").dt.year
    for column in POLLUTANTS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["id"] + POLLUTANTS).sort_values("year")
    years = sorted(data["year"].unique())
    if len(years) < 3:
        raise ValueError("At least three distinct years are required for temporal validation")
    cutoff = years[max(1, int(len(years) * 0.8))]
    training = data[data.year < cutoff]
    testing = data[data.year >= cutoff]
    preprocess = ColumnTransformer([
        ("year", "passthrough", ["year"]),
        ("station", OneHotEncoder(handle_unknown="ignore"), ["id"]),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    model.fit(training[["year", "id"]], training[POLLUTANTS])
    prediction = model.predict(testing[["year", "id"]])
    baseline = DummyRegressor(strategy="mean").fit(training[["year", "id"]], training[POLLUTANTS])
    baseline_prediction = baseline.predict(testing[["year", "id"]])
    metrics = {"temporal_test_start": int(cutoff), "training_rows": len(training), "test_rows": len(testing),
               "pollutants": {name: {"mse": mean_squared_error(testing[name], prediction[:, i]),
                                      "r2": r2_score(testing[name], prediction[:, i]),
                                      "mean_baseline_mse": mean_squared_error(testing[name], baseline_prediction[:, i])}
                              for i, name in enumerate(POLLUTANTS)}}
    artifact = {"model": model, "pollutants": POLLUTANTS, "stations": sorted(training.id.unique()),
                "training_years": [int(training.year.min()), int(training.year.max())]}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))
    return artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Local semicolon-delimited station measurements")
    parser.add_argument("--output", default="data/water-model.joblib")
    args = parser.parse_args()
    train(args.csv, args.output)
