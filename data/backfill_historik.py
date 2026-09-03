"""
Load the historical CSVs into your own bronze tables.

    uv run python data/backfill_historik.py --area SE3

Run this once, after you have fetched real data with fetch_historik.py and
applied db/schema.sql to your own database.

The point of this script is that it writes bronze in **the same shape your
daily job writes**. If the backfill wrote flat rows and the daily job writes
API payloads, every query downstream needs two code paths, and the pipeline you
tested against history is not the pipeline that runs tomorrow morning.

So the two `build_*_payload` functions below are the parts to check against
your own ingestion code. Everything else is plumbing.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

DATA_DIR = Path(__file__).parent
PRICES_CSV = DATA_DIR / "historik_elpris.csv.gz"
WEATHER_CSV = DATA_DIR / "historik_vader.csv.gz"

# The station is read from your own weather CSV, not from a hard-coded table,
# so whichever station you actually fetched is the one that gets loaded.


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def build_price_payload(rows):
    """One day of prices, in the shape elprisetjustnu.se returns."""
    return [
        {
            "SEK_per_kWh": row.SEK_per_kWh,
            "EUR_per_kWh": row.EUR_per_kWh,
            "EXR": row.EXR,
            "time_start": row.time_start,
            "time_end": row.time_end,
        }
        for row in rows
    ]


def load_prices(conn, area):
    df = pd.read_csv(PRICES_CSV)
    df = df[df.price_area == area]
    if df.empty:
        raise SystemExit(f"No rows for {area}. Areas in the file: "
                         f"{sorted(pd.read_csv(PRICES_CSV).price_area.unique())}")

    # The API is queried one local date at a time, so bronze is one row per day.
    df["price_date"] = df.time_start.str[:10]

    records = [
        (area, day, json.dumps(build_price_payload(list(group.itertuples(index=False)))))
        for day, group in df.groupby("price_date", sort=True)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO raw__elpris (price_area, price_date, data) VALUES %s",
            records,
        )
    conn.commit()
    return len(records), len(df)


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

PARAMETERS = {
    "temperature": "Lufttemperatur",
    "wind": "Vindhastighet",
}


def build_weather_payload(value, quality, observed_at):
    """One observation, in the shape SMHI returns."""
    # A gap in the CSV is NaN, and json.dumps writes that as bare NaN, which
    # Postgres rejects as invalid JSON — failing the whole batch with an error
    # that never mentions NaN. Real SMHI data has gaps; the synthetic file does not.
    return {
        "value": None if pd.isna(value) else value,
        "quality": quality,
        "date": observed_at.isoformat(),
    }


def load_weather(conn, station=None):
    # The header has a space and brackets in "Tid (UTC)", so rename on the way in.
    df = pd.read_csv(WEATHER_CSV, sep=";")
    df = df.rename(columns={"Tid (UTC)": "tid", "Datum": "datum"})

    present = sorted(df.Stationsnummer.unique())
    if station is None:
        if len(present) != 1:
            raise SystemExit(f"The file has several stations: {present}. "
                             f"Pass --station to say which one is yours.")
        station = present[0]
    df = df[df.Stationsnummer == station]
    if df.empty:
        raise SystemExit(f"No rows for station {station}. In the file: {present}")

    for param, column in PARAMETERS.items():
        if df[column].isna().all():
            raise SystemExit(
                f"{column} is empty for station {station}. Run "
                f"check_weather_coverage.py — you probably need another station.")

    # SMHI gives UTC without saying so. Say so, or Postgres assumes server time
    # and the weather lands 1-2 hours away from the prices it belongs to.
    df["observed_at"] = pd.to_datetime(df.datum + " " + df.tid).dt.tz_localize("UTC")

    records = []
    for param, column in PARAMETERS.items():
        for row in df.itertuples(index=False):
            records.append((
                str(station),
                param,
                getattr(row, "observed_at"),
                json.dumps(build_weather_payload(
                    getattr(row, column), row.Kvalitet, row.observed_at)),
            ))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO raw__weather (station, parameter, observed_at, data)
               VALUES %s""",
            records,
        )
    conn.commit()
    return len(records), station


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", required=True, help="your team's price area, e.g. SE3")
    ap.add_argument("--station", type=int, default=None,
                    help="SMHI station id (default: the one in your weather CSV)")
    args = ap.parse_args()

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env.")

    conn = psycopg2.connect(url)

    days, hours = load_prices(conn, args.area.upper())
    print(f"prices  : {days:,} days ({hours:,} intervals) -> raw__elpris")

    obs, station = load_weather(conn, args.station)
    print(f"weather : {obs:,} observations -> raw__weather  (station {station})")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*), min(price_date), max(price_date) FROM raw__elpris")
        print("\nraw__elpris :", cur.fetchone())
        cur.execute("SELECT count(*), min(observed_at), max(observed_at) FROM raw__weather")
        print("raw__weather:", cur.fetchone())

    conn.close()


if __name__ == "__main__":
    main()