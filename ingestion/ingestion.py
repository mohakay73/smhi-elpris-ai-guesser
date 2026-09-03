"""Fetch a day of hourly weather for one location and land it in the raw layer."""

import os
from datetime import datetime, timezone
import psycopg2
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from psycopg2.extras import Json

load_dotenv()  # only matters when running outside Docker; compose passes .env in

app = FastAPI()

SMHI_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
    "/parameter/{param}/station/{station}/period/latest-day/data.json"
)

# 1 = air temperature, 4 = wind speed
PARAMETERS = {1: "temperature", 4: "wind"}

def get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL not set in environment or .env file",
        )
    return url

def fetch_weather_rows(station: int | str):
    """Fetch both temperature and wind speed for a single station."""
    rows = []
    for param_id, parameter_name in PARAMETERS.items():
        response = requests.get(
            SMHI_URL.format(param=param_id, station=station), timeout=15
        )
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"SMHI API returned {response.status_code} for param {param_id}",
            )

        for entry in response.json().get("value") or []:
            # SMHI provides epoch milliseconds in UTC -> convert to aware UTC datetime
            observed_at = datetime.fromtimestamp(entry["date"] / 1000, tz=timezone.utc)
            rows.append((str(station), parameter_name, observed_at, Json(entry)))
    return rows


def insert_weather_rows(rows):
    """Insert raw weather records in bulk."""
    if not rows:
        return 0
    conn = psycopg2.connect(get_db_url())
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


@app.get("/")
def read_root():
    return {
        "status": "ready",
        "endpoints": {
            "electricity": "/ingestion/electricity?area=SE3&date=2026-08-25",
            "weather": "/ingestion/weather?station=87440",
        },
    }


@app.get("/ingestion/electricity")
def ingest_electricity(area: str, date: str):
    """
    Fetch raw electricity prices and insert one row into raw__elpris.
    date format: YYYY-MM-DD (e.g. 2026-08-25)
    area: SE1, SE2, SE3, or SE4
    """
    try:
        year, month, day = date.split("-")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    url = f"https://www.elprisetjustnu.se/api/v1/prices/{year}/{month}-{day}_{area.upper()}.json"
    response = requests.get(url, timeout=10)

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Elpris API returned {response.status_code}: {response.text[:200]}",
        )

    payload = response.json()
    conn = psycopg2.connect(get_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO raw__elpris (price_area, price_date, data) VALUES (%s, %s, %s)",
                (area.upper(), date, Json(payload)),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database insert failed: {exc}")
    finally:
        conn.close()

    return {
        "status": "ok",
        "price_area": area.upper(),
        "price_date": date,
        "records_ingested": len(payload) if isinstance(payload, list) else 1,
    }


@app.get("/ingestion/weather")
def ingest_weather(station: str):
    """Fetch and insert latest-day temperature & wind speed for a station."""
    try:
        rows = fetch_weather_rows(station)
        inserted_count = insert_weather_rows(rows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weather ingestion failed: {exc}")

    return {
        "status": "ok",
        "station": station,
        "rows_inserted": inserted_count,
    }