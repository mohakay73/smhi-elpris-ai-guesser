"""
Inference pipeline: load model.pkl, query latest data from Neon,
and predict tomorrow's electricity price for SE2.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

# Import feature engineering functions from train.py
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from train import build_features, load_weather_from_db, daily_weather_table

load_dotenv()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")
PRICE_AREA = os.getenv("PRICE_AREA", "SE2").upper()


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found at {MODEL_PATH}. Run train.py first.")

    with MODEL_PATH.open("rb") as fh:
        payload = pickle.load(fh)

    model = payload["model"]
    features = payload["features"]

    print(f"Loaded model trained at {payload['trained_at']} (MAE: {payload['metrics']['mae']:.4f})")

    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set.")

    con = duckdb.connect()
    con.sql("INSTALL postgres; LOAD postgres;")
    con.sql(f"ATTACH '{DATABASE_URL}' AS pg (TYPE postgres);")

    # Pull latest prices to build price summary table
    prices_query = f"""
        SELECT starts_at, sek_per_kwh, peak_price AS peak
        FROM (
            SELECT starts_at, sek_per_kwh, MAX(sek_per_kwh) OVER(PARTITION BY starts_at::date) as peak_price
            FROM pg.stg__elpris
            WHERE price_area = '{PRICE_AREA}'
        )
        ORDER BY starts_at ASC;
    """
    # Alternatively use standard load from db logic:
    prices_query = f"""
        SELECT starts_at, sek_per_kwh
        FROM pg.stg__elpris
        WHERE price_area = '{PRICE_AREA}'
        ORDER BY starts_at ASC;
    """
    prices = con.sql(prices_query).df()
    if prices.empty:
        raise SystemExit(f"No price records found for {PRICE_AREA}.")

    prices["ts_utc"] = pd.to_datetime(prices["starts_at"], utc=True)
    prices["date_local"] = pd.to_datetime(prices["ts_utc"].dt.tz_convert("Europe/Stockholm").dt.date)

    grouped = prices.groupby("date_local")["sek_per_kwh"]
    daily_price = pd.DataFrame(
        {
            "mean_price": grouped.mean(),
            "peak_price": grouped.max(),
            "min_price": grouped.min(),
            "n_intervals": grouped.size(),
        }
    ).sort_index()

    # Load weather and build daily weather table
    weather = load_weather_from_db(con, PRICE_AREA)
    daily_weather = daily_weather_table(weather)

    # Build feature DataFrame
    df = build_features(daily_price, daily_weather)
    
    # Take the very last row (today) to predict tomorrow
    latest_row = df.iloc[[-1]][features].copy()

    # Impute missing tomorrow weather features using today's actual weather as proxy
    for col in ("tomorrow_temp_mean", "tomorrow_temp_min", "tomorrow_temp_max", "tomorrow_wind_mean"):
        base_col = col.replace("tomorrow_", "")
        if latest_row[col].isna().any() and base_col in df.columns:
            fallback_val = df.iloc[-1].get(base_col)
            if pd.notna(fallback_val):
                latest_row[col] = fallback_val

    print("\nLatest Feature Vector for Inference (Imputed):")
    print(latest_row.T)

    prediction = model.predict(latest_row)[0]
    target_name = payload["target"]
    print(f"\n🔮 Predicted tomorrow's {target_name} price for {PRICE_AREA}: {prediction:.4f} SEK/kWh")


if __name__ == "__main__":
    main()