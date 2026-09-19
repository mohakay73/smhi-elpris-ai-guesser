from __future__ import annotations

import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from model.train import (
    AREA_STATIONS,
    STOCKHOLM,
    daily_price_table,
    daily_weather_table,
)

load_dotenv()

PRICE_AREA = os.environ["PRICE_AREA"]
TARGET = os.getenv("TARGET", "mean").lower()
PARAM_TEMP = "1"
PARAM_WIND = "4"


def load_prices_db(conn, area: str) -> pd.DataFrame:
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
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)
    return df.sort_values("ts_utc").reset_index(drop=True)


def load_weather_db(conn, station: int | str) -> pd.DataFrame:
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
    wide["date_local"] = pd.to_datetime(wide["ts_local"].dt.date)
    return wide.sort_values("ts_utc").reset_index(drop=True)


def load_forecast_db(conn, area: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT observed_at, temp_c, wind_ms
            FROM stg__forecast
            WHERE station = %s
            ORDER BY observed_at
            """,
            (f"forecast:{area}",),
        )
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["observed_at", "temp_c", "wind_ms"])
    df["ts_utc"] = pd.to_datetime(df["observed_at"], utc=True)
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)
    return df.sort_values("ts_utc").reset_index(drop=True)


def daily_forecast_table(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame(columns=["temp_mean", "temp_min", "temp_max", "wind_mean"])
    grouped = forecast.groupby("date_local")
    daily = pd.DataFrame(
        {
            "temp_mean": grouped["temp_c"].mean(),
            "temp_min": grouped["temp_c"].min(),
            "temp_max": grouped["temp_c"].max(),
            "wind_mean": grouped["wind_ms"].mean(),
        }
    )
    return daily.sort_index()


def build_features(
    daily_price: pd.DataFrame,
    daily_weather: pd.DataFrame,
    daily_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = daily_price.join(daily_weather, how="left")

    if daily_forecast is not None and not daily_forecast.empty:
        forecast_shifted = daily_forecast.copy()
        forecast_shifted.index = forecast_shifted.index - pd.Timedelta(days=1)
        for col in ("temp_mean", "temp_min", "temp_max", "wind_mean"):
            if col in forecast_shifted.columns:
                df[f"tomorrow_{col}"] = df.index.map(forecast_shifted[col])
            else:
                df[f"tomorrow_{col}"] = None
    else:
        for col in ("temp_mean", "temp_min", "temp_max", "wind_mean"):
            if col in df.columns:
                df[f"tomorrow_{col}"] = df[col].shift(-1)
            else:
                df[f"tomorrow_{col}"] = None

    df["price_today"] = df["mean_price"]
    df["price_yesterday"] = df["mean_price"].shift(1)
    df["price_7d_ago"] = df["mean_price"].shift(7)
    df["price_7d_mean"] = df["mean_price"].rolling(window=7, min_periods=1).mean()
    df["peak_today"] = df["peak_price"]

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    label_col = "mean_price" if TARGET == "mean" else "peak_price"
    df["y"] = df[label_col].shift(-1)
    return df


def write_features(conn, area: str, features: pd.DataFrame) -> int:
    """Write feature rows to feat__daily with an idempotent upsert."""
    if features.empty:
        return 0

    features = features.copy()
    
    if "date_local" not in features.columns and "target_date" not in features.columns:
        features = features.reset_index()
        if "index" in features.columns and "date_local" not in features.columns:
            features = features.rename(columns={"index": "date_local"})

    if "price_area" not in features.columns:
        features["price_area"] = area

    date_col = "date" if "date" in features.columns else "date_local"
    if "target_date" not in features.columns and date_col in features.columns:
        features["target_date"] = features[date_col]

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
    station_info = AREA_STATIONS[area] if 'AREA_STATIONS' in globals() or 'AREA_STATIONS' in locals() else "134110"
    if isinstance(station_info, (list, tuple)):
        station = station_info[0]
    else:
        station = station_info
    
    prices = load_prices_db(conn, area)
    weather = load_weather_db(conn, station)
    forecast_raw = load_forecast_db(conn, area)

    daily_prices = daily_price_table(prices)
    daily_weather = daily_weather_table(weather)
    daily_forecast = daily_forecast_table(forecast_raw)
    
    if "date_local" in daily_prices.columns:
        daily_prices.index = pd.to_datetime(daily_prices["date_local"])
    elif "date" in daily_prices.columns:
        daily_prices.index = pd.to_datetime(daily_prices["date"])
        
    if "date_local" in daily_weather.columns:
        daily_weather.index = pd.to_datetime(daily_weather["date_local"])
    elif "date" in daily_weather.columns:
        daily_weather.index = pd.to_datetime(daily_weather["date"])

    features = build_features(daily_prices, daily_weather, daily_forecast)

    return write_features(conn, area, features)


def main() -> None:
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
        n = refresh_daily_features(conn, PRICE_AREA)
        conn.commit()
    print(f"wrote {n} rows to feat__daily for {PRICE_AREA}")


if __name__ == "__main__":
    main()