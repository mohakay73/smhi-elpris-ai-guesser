from datetime import datetime, timezone
import os

import psycopg2
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from psycopg2.extras import Json

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI()

PARAMETERS = {1: "temperature", 4: "wind"}

SMHI_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
    "/parameter/{param}/station/{station}/period/latest-day/data.json"
)


@app.get("/")
def read_root():
    return {
        "electricity": "/ingestion/electricity?area=SE2&date=2026-01-01",
        "weather": "/ingestion/weather?station=134110",
        "both": "/ingestion?area=SE2&date=2026-01-01&station=134110",
    }


@app.get("/ingestion/electricity")
def ingestion_electricity(area: str, date: str):
    area = area.upper()

    response = requests.get(
        f"https://www.elprisetjustnu.se/api/v1/prices/{date[:4]}/{date[5:7]}-{date[8:10]}_{area}.json",
        timeout=15,
    )

    if not response.ok:
        raise HTTPException(
            502,
            f"Electricity API returned {response.status_code}: {response.text[:200]}",
        )

    payload = response.json()
    insert_elpris(area, date, payload)

    return {
        "status": "ok",
        "table": "raw__elpris",
        "price_area": area,
        "price_date": date,
        "rows_in_payload": len(payload),
    }


@app.get("/ingestion/weather")
def ingestion_weather(station: int = 134110):
    rows = fetch_weather_rows(station)
    rows_inserted = insert_weather_rows(rows)

    return {
        "status": "ok",
        "table": "raw__weather",
        "station": station,
        "rows_inserted": rows_inserted,
    }


@app.get("/ingestion")
def ingestion_both(area: str, date: str, station: int = 134110):
    area = area.upper()

    price_response = requests.get(
        f"https://www.elprisetjustnu.se/api/v1/prices/{date[:4]}/{date[5:7]}-{date[8:10]}_{area}.json",
        timeout=15,
    )

    if not price_response.ok:
        raise HTTPException(
            502,
            f"Electricity API returned {price_response.status_code}: {price_response.text[:200]}",
        )

    price_payload = price_response.json()
    insert_elpris(area, date, price_payload)

    weather_rows = fetch_weather_rows(station)
    weather_count = insert_weather_rows(weather_rows)

    return {
        "status": "ok",
        "electricity": {
            "table": "raw__elpris",
            "price_area": area,
            "price_date": date,
            "rows_in_payload": len(price_payload),
        },
        "weather": {
            "table": "raw__weather",
            "station": station,
            "rows_inserted": weather_count,
        },
    }


def insert_elpris(area, date, payload):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO raw__elpris (price_area, price_date, data) VALUES (%s, %s, %s)",
                (area, date, Json(payload)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_weather_rows(station):
    rows = []

    for param_id, parameter_name in PARAMETERS.items():
        response = requests.get(
            SMHI_URL.format(param=param_id, station=station),
            timeout=15,
        )
        response.raise_for_status()

        for entry in response.json().get("value") or []:
            observed_at = datetime.fromtimestamp(entry["date"] / 1000, tz=timezone.utc)
            rows.append((str(station), parameter_name, observed_at, Json(entry)))

    return rows


def insert_weather_rows(rows):
    conn = psycopg2.connect(DATABASE_URL)
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