"""
Baseline training pipeline: predict tomorrow's electricity price.

This is the script your team's training pipeline (project component B) grows
out of. It works as-is. It is not good. Read the LIMITATIONS section at the
bottom before you decide what to change first -- several of the shortcuts
here are the exact ones the project asks you to fix.

Run it:

    cp .env.example .env          # then set PRICE_AREA and TARGET
    uv run python model/train.py

It reads data/historik_elpris.csv.gz and data/historik_vader.csv.gz, builds
one feature row per day, fits a model, prints an evaluation, and writes
model/model.pkl.

Configuration comes from the environment (see .env.example):

    PRICE_AREA   SE1 | SE2 | SE3 | SE4      your team's assigned area
    TARGET       mean | peak                your team's assigned target
                 mean = tomorrow's average price across the day
                 peak = tomorrow's single most expensive hour
"""

from __future__ import annotations

import json
import os
import pickle
import zoneinfo
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PRICE_CSV = ROOT / "data" / "historik_elpris.csv.gz"
WEATHER_CSV = ROOT / "data" / "historik_vader.csv.gz"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

PRICE_AREA = os.getenv("PRICE_AREA", "SE2").upper()
TARGET = os.getenv("TARGET", "mean").lower()

STOCKHOLM = zoneinfo.ZoneInfo("Europe/Stockholm")

# Which SMHI station supplies the weather for each price area.
# Keep this in sync with the same table in data/fetch_historik.py.
AREA_STATIONS = {
    "SE1": (162860, "Lulea-Kallax Flygplats"),
    "SE2": (134110, "Ostersunds Flygplats"),
    "SE3": (98230, "Stockholm-Observatoriekullen A"),
    "SE4": (53430, "Malmo A"),
}


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def load_prices(area: str) -> pd.DataFrame:
    """Load prices for one area, on a proper timezone-aware index.

    Two things happen here that are easy to get wrong.

    First, `time_start` carries a UTC offset that changes with daylight
    saving (+01:00 winter, +02:00 summer). Parsing with utc=True gives a
    single unambiguous instant per row; converting to Europe/Stockholm
    afterwards gives the local wall-clock the price actually applied to.
    Skip the utc=True and pandas hands you an object-dtype column of mixed
    offsets that sorts wrongly across the DST boundary.

    Second, the source contains exact duplicate rows from re-ingestion.
    They must go before any aggregation, or a duplicated hour is silently
    weighted twice in the daily mean.
    """
    df = pd.read_csv(PRICE_CSV)
    df = df[df["price_area"] == area].copy()
    if df.empty:
        raise SystemExit(f"No price rows for {area}. Areas present: "
                         f"{sorted(pd.read_csv(PRICE_CSV)['price_area'].unique())}")

    df["ts_utc"] = pd.to_datetime(df["time_start"], utc=True, format="ISO8601")
    df = df.drop_duplicates(subset=["ts_utc"], keep="first")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    df["date_local"] = df["ts_local"].dt.date

    return df.sort_values("ts_utc").reset_index(drop=True)


def load_weather(area: str) -> pd.DataFrame:
    """Load the weather series for this area's station.

    SMHI timestamps are UTC and carry no offset marker, so they must be
    localised explicitly. Getting this wrong shifts every weather feature by
    one or two hours relative to the prices -- a bug that will not crash
    anything and will quietly cost you accuracy.

    Quality codes: G is checked and approved, Y is suspect or aggregated.
    This baseline keeps both. Whether that is the right call for your area
    is a question worth actually answering rather than inheriting.
    """
    station, _name = AREA_STATIONS[area]

    df = pd.read_csv(WEATHER_CSV, sep=";")
    df = df[df["Stationsnummer"] == station].copy()
    if df.empty:
        raise SystemExit(f"No weather rows for station {station} ({area}).")

    stamp = pd.to_datetime(
        df["Datum"].astype(str) + " " + df["Tid (UTC)"].astype(str),
        errors="coerce",
    )
    df["ts_utc"] = stamp.dt.tz_localize("UTC")
    df = df.dropna(subset=["ts_utc"])
    df["ts_local"] = df["ts_utc"].dt.tz_convert(STOCKHOLM)
    df["date_local"] = df["ts_local"].dt.date

    df = df.rename(columns={"Lufttemperatur": "temp_c", "Vindhastighet": "wind_ms"})
    return df.sort_values("ts_utc").reset_index(drop=True)


# --------------------------------------------------------------------------
# aggregate to one row per day
# --------------------------------------------------------------------------
def daily_price_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Collapse the price series to one row per local calendar day.

    Note what is deliberately NOT assumed: that a day has 24 rows. It does
    not. Sweden moved to 15-minute prices on 2025-10-01, so days before that
    have 24 points and days after have 96. Daylight saving adds two more
    exceptions a year -- 23 hours in March, 25 in October.

    Aggregating with mean() and max() over whatever rows exist is correct
    across all four cases. Any formula with a hard-coded 24 in it is not.

    `peak_price` is the single most expensive interval. Be aware that after
    the resolution change this means the most expensive *quarter hour*, not
    the most expensive hour -- a 15-minute peak is systematically higher than
    an hourly one. That discontinuity sits right in the middle of the
    training data and is not handled here. See LIMITATIONS.
    """
    grouped = prices.groupby("date_local")["SEK_per_kWh"]
    daily = pd.DataFrame(
        {
            "mean_price": grouped.mean(),
            "peak_price": grouped.max(),
            "min_price": grouped.min(),
            "n_intervals": grouped.size(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def daily_weather_table(weather: pd.DataFrame) -> pd.DataFrame:
    """One row per day. Days with no observations at all simply do not appear.

    They are left absent rather than filled, so the gap stays visible to the
    join below instead of being disguised as a real measurement.
    """
    grouped = weather.groupby("date_local")
    daily = pd.DataFrame(
        {
            "temp_mean": grouped["temp_c"].mean(),
            "temp_min": grouped["temp_c"].min(),
            "temp_max": grouped["temp_c"].max(),
            "wind_mean": grouped["wind_ms"].mean(),
            "n_obs": grouped["temp_c"].size(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def build_features(daily_price: pd.DataFrame, daily_weather: pd.DataFrame) -> pd.DataFrame:
    """One row per prediction day D, predicting the price on day D+1.

    The information boundary matters more than the feature list. At the
    moment you predict -- some time on day D -- you know:

      * prices actually observed on D and earlier
      * weather actually observed on D and earlier
      * a *forecast* of tomorrow's weather

    You do not know tomorrow's prices. That is the thing being predicted, and
    the source does not publish it until roughly 13:00 on day D anyway.

    This baseline uses tomorrow's OBSERVED temperature as a stand-in for
    tomorrow's forecast temperature. That is a defensible modelling choice
    at training time and a lie at inference time, because in production you
    will have a forecast with error in it, not the truth. Your model will
    therefore look better here than it performs in the wild. Quantifying that
    gap is a genuinely good use of a sprint.
    """
    df = daily_price.join(daily_weather, how="left")

    # Tomorrow's weather, as a forecast would supply it.
    for col in ("temp_mean", "temp_min", "temp_max", "wind_mean"):
        df[f"tomorrow_{col}"] = df[col].shift(-1)

    # Price history available at prediction time.
    df["price_today"] = df["mean_price"]
    df["price_yesterday"] = df["mean_price"].shift(1)
    df["price_7d_ago"] = df["mean_price"].shift(7)
    df["price_7d_mean"] = df["mean_price"].rolling(7, min_periods=3).mean()
    df["peak_today"] = df["peak_price"]

    # Calendar.
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # The label.
    label_col = "mean_price" if TARGET == "mean" else "peak_price"
    df["y"] = df[label_col].shift(-1)

    return df


FEATURES = [
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


# --------------------------------------------------------------------------
def main() -> None:
    if TARGET not in ("mean", "peak"):
        raise SystemExit(f"TARGET must be 'mean' or 'peak', got {TARGET!r}")

    print(f"area={PRICE_AREA}  target={TARGET}")

    prices = load_prices(PRICE_AREA)
    weather = load_weather(PRICE_AREA)

    daily_price = daily_price_table(prices)
    daily_weather = daily_weather_table(weather)

    print(f"price days   {len(daily_price)}  "
          f"({daily_price.index.min().date()} .. {daily_price.index.max().date()})")
    print(f"weather days {len(daily_weather)}  "
          f"missing on {len(daily_price.index.difference(daily_weather.index))} price days")

    # Visible evidence of the resolution change, so nobody has to take it on
    # faith. If your own fetched data does not show this, check your dates.
    counts = daily_price["n_intervals"].value_counts().sort_index()
    print(f"intervals per day: {counts.to_dict()}")

    df = build_features(daily_price, daily_weather)
    df = df.dropna(subset=["y"])

    usable = df.dropna(subset=["price_yesterday", "price_7d_ago"])
    print(f"usable rows  {len(usable)} of {len(df)}")

    # Chronological split. NOT random: this is a time series, and a random
    # split lets the model learn from days that come after the ones it is
    # tested on. That leak flatters the score and teaches you nothing.
    cutoff = int(len(usable) * 0.8)
    train, test = usable.iloc[:cutoff], usable.iloc[cutoff:]
    print(f"train {len(train)} ({train.index.min().date()} .. {train.index.max().date()})")
    print(f"test  {len(test)} ({test.index.min().date()} .. {test.index.max().date()})")

    # HistGradientBoostingRegressor handles NaN natively, which is why the
    # missing weather days above did not need imputing to get this far. That
    # is convenient, not free -- the model is quietly learning "weather
    # missing" as a signal. Whether that is acceptable is your call.
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[FEATURES], train["y"])

    pred = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["y"], pred)
    r2 = r2_score(test["y"], pred)

    # Always compare against the stupidest thing that could work. A model
    # that cannot beat "tomorrow will be like today" is not a model.
    naive = test["price_today"] if TARGET == "mean" else test["peak_today"]
    naive_mae = mean_absolute_error(test["y"], naive)

    print()
    print(f"  MAE          {mae:.4f} SEK/kWh")
    print(f"  R2           {r2:.3f}")
    print(f"  naive MAE    {naive_mae:.4f} SEK/kWh  (tomorrow = today)")
    verdict = "beats" if mae < naive_mae else "LOSES TO"
    print(f"  model {verdict} the naive baseline")

    payload = {
        "model": model,
        "features": FEATURES,
        "price_area": PRICE_AREA,
        "target": TARGET,
        "trained_at": datetime.now(tz=STOCKHOLM).isoformat(),
        "n_train": len(train),
        "metrics": {"mae": float(mae), "r2": float(r2), "naive_mae": float(naive_mae)},
        "sklearn_version": __import__("sklearn").__version__,
    }
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(payload, fh)

    print(f"\nwrote {MODEL_PATH}")
    print(json.dumps(payload["metrics"], indent=2))


# --------------------------------------------------------------------------
# LIMITATIONS -- read before extending
# --------------------------------------------------------------------------
# Known problems with this baseline, roughly in the order they will hurt you.
# You are not asked to train a better model -- the model itself is given (see
# the project brief, section 9). But most of what's below is exactly the kind
# of real-data problem that only shows up once you wrap a live, scheduled
# pipeline around this model instead of running it once against a static CSV
# -- which weather endpoint to call, how to handle a live feed's gaps and
# quality flags, a mid-series definitional break. Handling that IS the
# project: it is component A/B's job, not out-of-scope model research. #5 and
# #6 below are the two closer to genuine modeling choices, and even there the
# ask is to notice the tradeoff and document it, not to out-engineer this
# baseline.
#
# 1. Tomorrow's weather is the observed truth, not a forecast. In production
#    you must call SMHI's forecast API instead, and your accuracy will drop.
#    Measure by how much before you promise anyone anything.
#
# 2. The peak target changes meaning on 2025-10-01. Before that date the peak
#    is the most expensive hour; after it, the most expensive quarter-hour,
#    which is systematically higher. If TARGET=peak you are training across a
#    definitional break. Resampling everything to hourly first is one fix;
#    training only on post-change data is another; ignoring it is not.
#
# 3. Missing weather is passed to the model as NaN and learned as a category.
#    Short gaps could be interpolated. Long outages probably should not be.
#    Decide, write down why, and make the code say so.
#
# 4. Quality code Y is treated identically to G. Roughly a fifth of the
#    archive is Y.
#
# 5. Nothing is done about price spikes. A handful of extreme days dominate
#    MAE. Whether you care depends on what the prediction is for -- and that
#    is a question about the user, not about the data.
#
# 6. There is no cross-validation, one fixed split, and no confidence
#    interval on any number printed above. With ~700 rows, that MAE has more
#    uncertainty than its four decimal places suggest.
#
# 7. The model is retrained from scratch on a CSV. Your project must retrain
#    from your Postgres feature table instead, and register each result as a
#    new version in Hugging Face Hub. That is component B, and this script is
#    only its starting point.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()
