import pickle
from pathlib import Path

import pandas as pd

from model.predict import predict_with_intervals

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

with MODEL_PATH.open("rb") as file_handle:
    payload = pickle.load(file_handle)

base_features = payload.get("base_features", payload["features"])

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

missing = [name for name in base_features if neon_row.get(name) is None]
if missing:
    raise ValueError(f"Feature-värden saknas: {missing}")

raw_df = pd.DataFrame([neon_row])
res = predict_with_intervals(payload, raw_df)

print(f"Prediction (Mean):   {res['prediction']:.4f} SEK/kWh")
print(f"Prediction (Median): {res['p50']:.4f} SEK/kWh")
print(f"80% Confidence Band: [{res['p10']:.4f} .. {res['p90']:.4f}] SEK/kWh (P10–P90)")
print(f"Features ({len(payload['features'])}): {payload['features']}")
print(f"Metrics: {payload.get('metrics')}")