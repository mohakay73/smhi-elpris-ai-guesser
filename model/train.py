from __future__ import annotations

import json
import os
import pickle
import zoneinfo
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
PRICE_AREA = os.getenv("PRICE_AREA", "SE2").upper()
TARGET = os.getenv("TARGET", "mean").lower()
DATABASE_URL = os.getenv("DATABASE_URL")
STOCKHOLM = zoneinfo.ZoneInfo("Europe/Stockholm")

AREA_STATIONS = {
    "SE1": "162860",
    "SE2": "134110",
    "SE3": "97400",
    "SE4": "53430",
}


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set in environment or .env file.")
    con = duckdb.connect()
    con.sql("INSTALL postgres; LOAD postgres;")
    con.sql(f"ATTACH '{DATABASE_URL}' AS pg (TYPE postgres);")
    return con


def load_prices_from_db(con: duckdb.DuckDBPyConnection, area: str) -> pd.DataFrame:
    query = f"""
        SELECT starts_at, sek_per_kwh
        FROM pg.stg__elpris
        WHERE price_area = '{area}'
        ORDER BY starts_at ASC;
    """
    df = con.sql(query).df()
    if df.empty:
        raise SystemExit(f"No price records found in stg__elpris for area {area}.")

    df["ts_utc"] = pd.to_datetime(df["starts_at"], utc=True)
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)
    return df.sort_values("ts_utc").reset_index(drop=True)


def load_weather_from_db(con: duckdb.DuckDBPyConnection, area: str) -> pd.DataFrame:
    station = AREA_STATIONS.get(area, "134110")
    query = f"""
        SELECT station, parameter, observed_at, value, quality
        FROM pg.stg__weather
        WHERE station = '{station}'
        ORDER BY observed_at ASC;
    """
    df = con.sql(query).df()
    if df.empty:
        raise SystemExit(f"No weather records found for station {station} ({area}).")

    df["ts_utc"] = pd.to_datetime(df["observed_at"], utc=True)
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)

    pivoted = df.pivot_table(
        index=["date_local", "ts_local", "ts_utc"],
        columns="parameter",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted = pivoted.rename(columns={"temperature": "temp_c", "wind": "wind_ms"})
    return pivoted.sort_values("ts_utc").reset_index(drop=True)


def daily_price_table(prices: pd.DataFrame) -> pd.DataFrame:
    raw_grouped = prices.groupby("date_local")["sek_per_kwh"]

    if "ts_local" in prices.columns:
        # Resample to hourly mean first so peak_price is consistent across the
        # 2025-10-01 switch from 24 hourly intervals to 96 15-minute intervals.
        hourly = (
            prices.set_index("ts_local")
            .groupby("date_local")["sek_per_kwh"]
            .resample("1h")
            .mean()
            .reset_index()
        )
        hourly_grouped = hourly.groupby("date_local")["sek_per_kwh"]
    else:
        hourly_grouped = raw_grouped

    daily = pd.DataFrame(
        {
            "mean_price": raw_grouped.mean(),
            "peak_price": hourly_grouped.max(),
            "min_price": hourly_grouped.min(),
            "n_intervals": raw_grouped.size(),
        }
    )
    return daily.sort_index()


def daily_weather_table(weather: pd.DataFrame) -> pd.DataFrame:
    grouped = weather.groupby("date_local")
    daily = pd.DataFrame(
        {
            "temp_mean": grouped["temp_c"].mean() if "temp_c" in weather else None,
            "temp_min": grouped["temp_c"].min() if "temp_c" in weather else None,
            "temp_max": grouped["temp_c"].max() if "temp_c" in weather else None,
            "wind_mean": grouped["wind_ms"].mean() if "wind_ms" in weather else None,
            "n_obs": grouped.size(),
        }
    )
    return daily.sort_index()


def enrich_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical & market domain features from the 12 base features."""
    out = df.copy()
    temp_mean = pd.to_numeric(out["tomorrow_temp_mean"], errors="coerce")
    temp_min = pd.to_numeric(out["tomorrow_temp_min"], errors="coerce")
    temp_max = pd.to_numeric(out["tomorrow_temp_max"], errors="coerce")
    wind_mean = pd.to_numeric(out["tomorrow_wind_mean"], errors="coerce")
    price_today = pd.to_numeric(out["price_today"], errors="coerce")
    price_7d_mean = pd.to_numeric(out["price_7d_mean"], errors="coerce")
    peak_today = pd.to_numeric(out["peak_today"], errors="coerce")

    # 1. Heating Degree Days (electric heating & heat pumps ramp up below 17°C)
    out["heating_degree_days"] = (17.0 - temp_mean).clip(lower=0.0)

    # 2. Wind Power Proxy (turbine output scales with wind_speed^3 up to rated ~15 m/s)
    out["wind_power_proxy"] = wind_mean.clip(lower=0.0, upper=15.0) ** 3

    # 3. Diurnal Temperature Spread (clear-sky radiation cooling vs overcast low-pressure)
    out["tomorrow_temp_spread"] = (temp_max - temp_min).clip(lower=0.0)

    # 4. Price Momentum vs 7-day Regime
    out["price_momentum"] = price_today - price_7d_mean

    # 5. Intraday Spikiness Ratio (grid tightness indicator)
    out["peak_to_mean_ratio"] = peak_today / price_today.clip(lower=0.05)

    return out


def build_features(daily_price: pd.DataFrame, daily_weather: pd.DataFrame) -> pd.DataFrame:
    df = daily_price.join(daily_weather, how="left")

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

    df = enrich_domain_features(df)

    label_col = "mean_price" if TARGET == "mean" else "peak_price"
    df["y"] = df[label_col].shift(-1)
    return df


BASE_FEATURES = [
    "tomorrow_temp_mean",
    "tomorrow_temp_min",
    "tomorrow_temp_max",
    "tomorrow_wind_mean",
    "price_today",
    "price_yesterday",
    "price_7d_ago",
    "price_7d_mean",
    "peak_today",
    "dayofweek",
    "month",
    "is_weekend",
]

DOMAIN_FEATURES = [
    "heating_degree_days",
    "wind_power_proxy",
    "tomorrow_temp_spread",
    "price_momentum",
    "peak_to_mean_ratio",
]

FEATURES = BASE_FEATURES + DOMAIN_FEATURES


def _make_hgb_params() -> dict:
    return {
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_depth": 6,
        "min_samples_leaf": 15,
        "l2_regularization": 1.0,
        "random_state": 42,
    }


def main() -> None:
    if TARGET not in ("mean", "peak"):
        raise SystemExit(f"TARGET must be 'mean' or 'peak', got {TARGET!r}")

    print(f"Connecting to Neon DB for area={PRICE_AREA}, target={TARGET}...")
    con = get_duckdb_connection()

    prices = load_prices_from_db(con, PRICE_AREA)
    weather = load_weather_from_db(con, PRICE_AREA)

    daily_price = daily_price_table(prices)
    daily_weather = daily_weather_table(weather)

    p_min, p_max = daily_price.index.min().date(), daily_price.index.max().date()
    missing_w = len(daily_price.index.difference(daily_weather.index))
    print(f"price days   {len(daily_price)} ({p_min} .. {p_max})")
    print(f"weather days {len(daily_weather)} (missing on {missing_w} price days)")

    df = build_features(daily_price, daily_weather)
    df = df.dropna(subset=["y"])

    usable = df.dropna(subset=["price_yesterday", "price_7d_ago"])
    print(f"usable rows  {len(usable)} of {len(df)}")

    cutoff = int(len(usable) * 0.8)
    train, test = usable.iloc[:cutoff], usable.iloc[cutoff:]
    print(f"train {len(train)} ({train.index.min().date()} .. {train.index.max().date()})")
    print(f"test  {len(test)} ({test.index.min().date()} .. {test.index.max().date()})")

    # 1. Main point-prediction model (squared_error)
    model = HistGradientBoostingRegressor(loss="squared_error", **_make_hgb_params())
    model.fit(train[FEATURES], train["y"])

    # 2. Quantile regression models for P10, P50 (median), P90 uncertainty bands
    quantile_models: dict[str, HistGradientBoostingRegressor] = {}
    for q_name, q_val in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        q_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q_val,
            **_make_hgb_params(),
        )
        q_model.fit(train[FEATURES], train["y"])
        quantile_models[q_name] = q_model

    pred = model.predict(test[FEATURES])
    pred_p10 = quantile_models["p10"].predict(test[FEATURES])
    pred_p50 = quantile_models["p50"].predict(test[FEATURES])
    pred_p90 = quantile_models["p90"].predict(test[FEATURES])

    mae = mean_absolute_error(test["y"], pred)
    median_mae = mean_absolute_error(test["y"], pred_p50)
    r2 = r2_score(test["y"], pred)

    in_band = (test["y"] >= pred_p10) & (test["y"] <= pred_p90)
    picp_80 = float(in_band.mean())
    mean_band_width = float((pred_p90 - pred_p10).mean())

    naive = test["price_today"] if TARGET == "mean" else test["peak_today"]
    naive_mae = mean_absolute_error(test["y"], naive)

    print()
    print(f"  MAE (Mean)   {mae:.4f} SEK/kWh")
    print(f"  MAE (P50)    {median_mae:.4f} SEK/kWh")
    print(f"  R2           {r2:.3f}")
    print(f"  naive MAE    {naive_mae:.4f} SEK/kWh  (tomorrow = today)")
    print(f"  P10-P90 Cov  {picp_80 * 100:.1f}% (avg width {mean_band_width:.4f} SEK/kWh)")
    verdict = "beats" if mae < naive_mae else "LOSES TO"
    print(f"  model {verdict} the naive baseline")

    payload = {
        "model": model,
        "quantile_models": quantile_models,
        "features": FEATURES,
        "base_features": BASE_FEATURES,
        "domain_features": DOMAIN_FEATURES,
        "price_area": PRICE_AREA,
        "target": TARGET,
        "trained_at": datetime.now(tz=STOCKHOLM).isoformat(),
        "n_train": len(train),
        "metrics": {
            "mae": float(mae),
            "p50_mae": float(median_mae),
            "r2": float(r2),
            "naive_mae": float(naive_mae),
            "picp_80": picp_80,
            "mean_band_width": mean_band_width,
        },
        "sklearn_version": __import__("sklearn").__version__,
    }
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(payload, fh)

    print(f"\nwrote {MODEL_PATH}")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()