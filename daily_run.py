import os
import sys
from datetime import datetime, timezone
import requests
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    sys.exit("DATABASE_URL is not set.")

# Team configuration
AREA = "SE2"
STATION = "134110"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# 1. Ingest Electricity
year, month, day = TODAY.split("-")
price_url = f"https://www.elprisetjustnu.se/api/v1/prices/{year}/{month}-{day}_{AREA}.json"
res = requests.get(price_url, timeout=15)
res.raise_for_status()
price_payload = res.json()

conn = psycopg2.connect(DATABASE_URL)
with conn.cursor() as cur:
    cur.execute(
        "INSERT INTO raw__elpris (price_area, price_date, data) VALUES (%s, %s, %s)",
        (AREA, TODAY, Json(price_payload)),
    )
conn.commit()
print(f"Ingested {len(price_payload)} price intervals for {AREA} ({TODAY}).")

# 2. Ingest Weather (Temperature & Wind)
params = {1: "temperature", 4: "wind"}
weather_rows = []

for param_id, param_name in params.items():
    smhi_url = (
        f"https://opendata-download-metobs.smhi.se/api/version/1.0"
        f"/parameter/{param_id}/station/{STATION}/period/latest-day/data.json"
    )
    w_res = requests.get(smhi_url, timeout=15)
    w_res.raise_for_status()
    for entry in w_res.json().get("value") or []:
        obs_at = datetime.fromtimestamp(entry["date"] / 1000, tz=timezone.utc)
        weather_rows.append((str(STATION), param_name, obs_at, Json(entry)))

with conn.cursor() as cur:
    cur.executemany(
        "INSERT INTO raw__weather (station, parameter, observed_at, data) VALUES (%s, %s, %s, %s)",
        weather_rows,
    )
conn.commit()
conn.close()
print(f"Ingested {len(weather_rows)} weather observations for station {STATION}.")