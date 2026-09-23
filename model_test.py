from pathlib import Path
import pickle
import pandas as pd

MODEL_PATH = Path("model/model.pkl")

with MODEL_PATH.open("rb") as file_handle:
    payload = pickle.load(file_handle)

model = payload["model"]
feature_names = payload["features"]

# Ersätt värden nedan med resultatet från din SQL-fråga mot feat__daily
neon_row = {
    "tomorrow_temp_mean": 8.775,
    "tomorrow_temp_min": 7.1,
    "tomorrow_temp_max": 12.2,
    "tomorrow_wind_mean": 2.1666666666666665,
    "price_today": 0.09852135416666667,
    "price_yesterday": 0.04344833333333333,
    "price_7d_ago": 0.7495031249999999,
    "price_7d_mean": 0.6041904761904763,
    "peak_today": 0.46878,
    "dayofweek": 6,
    "month": 9,
    "is_weekend": True,
}

missing = [name for name in feature_names if neon_row.get(name) is None]
if missing:
    raise ValueError(f"Feature-värden saknas: {missing}")

feature_row = pd.DataFrame(
    [{name: neon_row[name] for name in feature_names}], 
    columns=feature_names
)

prediction = float(model.predict(feature_row)[0])
print(f"Prediction: {prediction}")
print(f"Features: {feature_names}")
print(f"Metrics: {payload.get('metrics')}")