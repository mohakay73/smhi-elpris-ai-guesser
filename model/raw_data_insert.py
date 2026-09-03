import os
import psycopg2
from psycopg2.extras import Json

conn = psycopg2.connect(os.environ["DATABASE_URL"])

try:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO raw__elpris (price_area, price_date, data) VALUES (%s, %s, %s)",
            (area, day, Json(payload)),
        )
    conn.commit()
except Exception:
    conn.rollback()
    raise
finally:
    conn.close()

import os
from datetime import datetime, timezone

import psycopg2
import requests
from psycopg2.extras import Json

SMHI_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
    "/parameter/{param}/station/{station}/period/latest-day/data.json"
)

# One call per parameter. This is the shape of the API, not a design choice.
PARAMETERS = {1: "temperature", 4: "wind"}


def fetch_weather_rows(station):
    """Both parameters for one station, as rows ready for raw__weather."""
    rows = []
    for param_id, parameter_name in PARAMETERS.items():
        response = requests.get(
            SMHI_URL.format(param=param_id, station=station), timeout=15
        )
        response.raise_for_status()

        for entry in response.json().get("value") or []:
            # SMHI gives epoch MILLISECONDS in UTC, and says so nowhere.
            observed_at = datetime.fromtimestamp(entry["date"] / 1000, tz=timezone.utc)
            rows.append((station, parameter_name, observed_at, Json(entry)))
    return rows


def insert_weather_rows(rows):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO raw__weather (station, parameter, observed_at, data) "
                "VALUES (%s, %s, %s, %s)",
                rows,
            )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()