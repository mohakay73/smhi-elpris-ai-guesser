"""
Starter for Tuesday's feature-pipeline studio.

Standard project only. Copy this file to your project root as
feature_pipeline.py and work in a pull request.
"""

from __future__ import annotations

import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from model.train import (
    AREA_STATIONS,
    STOCKHOLM,
    build_features,
    daily_price_table,
    daily_weather_table,
)

load_dotenv()

PRICE_AREA = os.environ["PRICE_AREA"]
PARAM_TEMP = "1"
PARAM_WIND = "4"

def load_prices_db(conn, area: str) -> pd.DataFrame:
    """Read staged price intervals in the shape daily_price_table() expects."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT starts_at, sek_per_kwh
            FROM stg__elpris
            WHERE price_area = %s
            ORDER BY starts_at
            """,
            (area,),
        )
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No price rows in stg__elpris for {area}.")

    df = pd.DataFrame(rows, columns=["starts_at", "sek_per_kwh"])
    df["ts_utc"] = pd.to_datetime(df["starts_at"], utc=True)
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    # Ensure this is wrapped in pd.to_datetime so it becomes a timestamp index later
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)
    return df.sort_values("ts_utc").reset_index(drop=True)


def load_weather_db(conn, station: int | str) -> pd.DataFrame:
    """Read and reshape staged weather data into wide format."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT parameter, observed_at, value
            FROM stg__weather
            WHERE station::text = %s
              AND parameter IN ('1', '4', 'temperature', 'wind')
            ORDER BY observed_at
            """,
            (str(station),),
        )
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No weather rows in stg__weather for station {station}.")

    long_df = pd.DataFrame(rows, columns=["parameter", "observed_at", "value"])
    
    long_df["parameter"] = long_df["parameter"].replace({
        "temperature": "temp_c",
        "1": "temp_c",
        "wind": "wind_ms",
        "4": "wind_ms",
    })

    wide = long_df.pivot(index="observed_at", columns="parameter", values="value").reset_index()

    wide["ts_utc"] = pd.to_datetime(wide["observed_at"], utc=True)
    wide["ts_local"] = wide["ts_utc"].dt.tz_convert(STOCKHOLM)
    # Ensure this is wrapped in pd.to_datetime as well
    wide["date_local"] = pd.to_datetime(wide["ts_local"].dt.date)
    return wide.sort_values("ts_utc").reset_index(drop=True)

def write_features(conn, area: str, features: pd.DataFrame) -> int:
    """Write feature rows to feat__daily with an idempotent upsert."""
    if features.empty:
        return 0

    features = features.copy()
    
    # Bring the index (dates) back into a column if it's not already a column
    if "date_local" not in features.columns and "target_date" not in features.columns:
        features = features.reset_index()
        if "index" in features.columns and "date_local" not in features.columns:
            features = features.rename(columns={"index": "date_local"})

    if "price_area" not in features.columns:
        features["price_area"] = area

    date_col = "date" if "date" in features.columns else "date_local"
    if "target_date" not in features.columns and date_col in features.columns:
        features["target_date"] = features[date_col]

    # Ensure target_date is formatted cleanly as a date object
    if "target_date" in features.columns:
        features["target_date"] = pd.to_datetime(features["target_date"]).dt.date

    cols = list(features.columns)
    
    with conn.cursor() as cur:
        tuples = [tuple(x) for x in features.to_numpy()]
        cols_sql = ", ".join([f'"{col}"' for col in cols])
        
        update_cols = [c for c in cols if c not in ("price_area", "target_date")]
        update_sql = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])

        query = f"""
            INSERT INTO feat__daily ({cols_sql})
            VALUES %s
            ON CONFLICT (price_area, target_date)
            DO UPDATE SET {update_sql}
        """
        execute_values(cur, query, tuples)

    return len(features)

def refresh_daily_features(conn, area: str) -> int:
    """Build the team's feature table once from the staging views."""
    station_info = AREA_STATIONS[area]
    station = station_info[0] if isinstance(station_info, (list, tuple)) else station_info
    
    prices = load_prices_db(conn, area)
    weather = load_weather_db(conn, station)

    daily_prices = daily_price_table(prices)
    daily_weather = daily_weather_table(weather)
    
    # Ensure daily tables have a proper DatetimeIndex based on their date column
    if "date_local" in daily_prices.columns:
        daily_prices.index = pd.to_datetime(daily_prices["date_local"])
    elif "date" in daily_prices.columns:
        daily_prices.index = pd.to_datetime(daily_prices["date"])
        
    if "date_local" in daily_weather.columns:
        daily_weather.index = pd.to_datetime(daily_weather["date_local"])
    elif "date" in daily_weather.columns:
        daily_weather.index = pd.to_datetime(daily_weather["date"])

    features = build_features(daily_prices, daily_weather)

    return write_features(conn, area, features)


def main() -> None:
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
        n = refresh_daily_features(conn, PRICE_AREA)
        conn.commit()
    print(f"wrote {n} rows to feat__daily for {PRICE_AREA}")


if __name__ == "__main__":
    main()