"""
Download the REAL history from elprisetjustnu.se and SMHI.

The CSVs shipped in this folder are synthetic placeholders so that the repo
runs on a fresh clone (see generate_synthetic_historik.py). This script
replaces them with actual measurements. Run it once, early.

    uv run python data/fetch_historik.py --from 2024-08-01 --to 2026-08-20

It is deliberately slow and polite: one HTTP call per day per price area,
with a pause between calls. Two years across four areas is roughly 2900
requests and takes something like 25-40 minutes. Fetch only the areas you
need if you are in a hurry:

    uv run python data/fetch_historik.py --areas SE3 --from 2025-01-01

Station lookup
--------------
Only station 98230 (Stockholm-Observatoriekullen A) has been verified. To
find the right key for another town before you fetch:

    uv run python data/fetch_historik.py --find-station "Lulea"

Neither API needs a key, and neither has a published rate limit. Do not
remove the sleep anyway -- these are free public services run for everyone.

Sources
-------
Prices  : https://www.elprisetjustnu.se/elpris-api
          GET /api/v1/prices/YYYY/MM-DD_SEn.json
          Available from 2022-11-01. Tomorrow's prices publish around 13:00
          local time the day before, not at midnight.
          Prices exclude VAT, grid fees and taxes.
Weather : https://opendata.smhi.se/apidocs/metobs/
          GET /api/version/1.0/parameter/{p}/station/{id}/period/{period}/data.csv
          parameter 1 = air temperature (celsius, hourly instantaneous)
          parameter 4 = wind speed (m/s, 10-min mean, hourly)
          period corrected-archive = quality-controlled history, excluding
          the last three months. Use latest-months to cover the recent tail.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zoneinfo
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PRICE_URL = "https://www.elprisetjustnu.se/api/v1/prices/{year}/{month:02d}-{day:02d}_{area}.json"
SMHI_BASE = "https://opendata-download-metobs.smhi.se/api/version/1.0"
SMHI_STATIONS = SMHI_BASE + "/parameter/1/station.json"
SMHI_DATA = SMHI_BASE + "/parameter/{param}/station/{station}/period/{period}/data.csv"

STOCKHOLM = zoneinfo.ZoneInfo("Europe/Stockholm")

# Keep this table in sync with generate_synthetic_historik.py and with the
# team assignment sheet. Verify every id before you rely on it.
AREA_STATIONS = {
    "SE1": (162860, "Lulea-Kallax Flygplats"),
    "SE2": (134110, "Ostersunds Flygplats"),
    "SE2": (134110, "Ostersunds Flygplats"),
    "SE3": (98230, "Stockholm-Observatoriekullen A"),
    "SE4": (53430, "Malmo A"),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "YH-02032-course-project/1.0 (teaching material)"})


# --------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------
def fetch_prices(areas: list[str], start: date, end: date, pause: float) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(areas) * ((end - start).days + 1)
    done = 0

    for area in areas:
        day = start
        while day <= end:
            url = PRICE_URL.format(year=day.year, month=day.month, day=day.day, area=area)
            try:
                resp = SESSION.get(url, timeout=30)
            except requests.RequestException as exc:
                print(f"  ! {area} {day}: {exc}", file=sys.stderr)
                day += timedelta(days=1)
                done += 1
                continue

            if resp.status_code == 200:
                for entry in resp.json():
                    # Note what is NOT done here: no renaming, no rounding, no
                    # timezone normalisation, no dropping of the fields you
                    # think you will not need. Land the source shape; clean it
                    # downstream. That separation is the whole point of a
                    # bronze layer.
                    entry["price_area"] = area
                    rows.append(entry)
            elif resp.status_code == 404:
                # Normal: the date has no published prices (too far in the
                # future, or before the source's history begins).
                pass
            else:
                print(f"  ! {area} {day}: HTTP {resp.status_code}", file=sys.stderr)

            done += 1
            if done % 50 == 0:
                print(f"  prices {done}/{total}", file=sys.stderr)
            time.sleep(pause)
            day += timedelta(days=1)

    if not rows:
        return pd.DataFrame(
            columns=["price_area", "SEK_per_kWh", "EUR_per_kWh", "EXR", "time_start", "time_end"]
        )

    df = pd.DataFrame(rows)
    cols = ["price_area", "SEK_per_kWh", "EUR_per_kWh", "EXR", "time_start", "time_end"]
    return df[cols].sort_values(["price_area", "time_start"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def find_station(query: str) -> None:
    """Print stations whose name contains `query` (case-insensitive)."""
    resp = SESSION.get(SMHI_STATIONS, timeout=60)
    resp.raise_for_status()
    hits = [
        s
        for s in resp.json().get("station", [])
        if query.casefold() in s.get("name", "").casefold()
    ]
    if not hits:
        print(f"No station matching {query!r}.")
        return
    for s in sorted(hits, key=lambda s: not s.get("active", False)):
        flag = "active" if s.get("active") else "  ----"
        print(f"{flag}  {s['key']:>8}  {s['name']}")


def _parse_smhi_csv(text: str, value_name: str) -> pd.DataFrame:
    """Pull the data block out of an SMHI CSV.

    The file is not a plain CSV. It opens with several metadata blocks
    separated by blank lines, and only then comes the real header row, which
    begins with 'Datum'. Some rows carry trailing explanation columns that do
    not belong to the data at all. Find the header, read from there, and keep
    only the columns you asked for.
    """
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("Datum;")),
        None,
    )
    if header_idx is None:
        raise ValueError("no 'Datum;' header row found -- did the CSV format change?")

    block = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(block),
        sep=";",
        usecols=[0, 1, 2, 3],
        names=["Datum", "Tid (UTC)", value_name, "Kvalitet"],
        header=0,
        dtype=str,
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["Datum", "Tid (UTC)"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


def fetch_station_series(station: int, param: int, value_name: str) -> pd.DataFrame:
    """Fetch one parameter for one station, stitching archive + recent months.

    corrected-archive stops about three months short of today. latest-months
    covers the tail. You need both, and they overlap -- de-duplicate on
    timestamp, preferring the corrected values.
    """
    frames = []
    for period in ("corrected-archive", "latest-months"):
        url = SMHI_DATA.format(param=param, station=station, period=period)
        resp = SESSION.get(url, timeout=120)
        if resp.status_code != 200:
            print(f"  ! station {station} param {param} {period}: HTTP {resp.status_code}",
                  file=sys.stderr)
            continue
        resp.encoding = "utf-8"
        try:
            frame = _parse_smhi_csv(resp.text, value_name)
        except ValueError as exc:
            print(f"  ! station {station} param {param} {period}: {exc}", file=sys.stderr)
            continue
        frame["_period"] = period
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["Datum", "Tid (UTC)", value_name, "Kvalitet"])

    out = pd.concat(frames, ignore_index=True)
    # corrected-archive sorts before latest-months, so keep='first' prefers it.
    out = out.sort_values("_period", kind="stable")
    out = out.drop_duplicates(subset=["Datum", "Tid (UTC)"], keep="first")
    return out.drop(columns=["_period"])


def fetch_weather(areas: list[str], start: date, end: date) -> pd.DataFrame:
    parts = []
    for area in areas:
        station, name = AREA_STATIONS[area]
        print(f"  weather {area}: station {station} ({name})", file=sys.stderr)

        temp = fetch_station_series(station, 1, "Lufttemperatur")
        wind = fetch_station_series(station, 4, "Vindhastighet")

        merged = temp.merge(
            wind[["Datum", "Tid (UTC)", "Vindhastighet"]],
            on=["Datum", "Tid (UTC)"],
            how="outer",
        )
        merged["Stationsnummer"] = station
        merged["Stationsnamn"] = name
        parts.append(merged)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    stamp = pd.to_datetime(out["Datum"] + " " + out["Tid (UTC)"], errors="coerce")
    out = out[(stamp >= pd.Timestamp(start)) & (stamp <= pd.Timestamp(end) + pd.Timedelta(days=1))]

    cols = [
        "Stationsnummer",
        "Stationsnamn",
        "Datum",
        "Tid (UTC)",
        "Lufttemperatur",
        "Vindhastighet",
        "Kvalitet",
    ]
    return out.reindex(columns=cols).sort_values(
        ["Stationsnummer", "Datum", "Tid (UTC)"], kind="stable"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="2024-08-01")
    ap.add_argument("--to", dest="end", default=None, help="default: yesterday")
    ap.add_argument("--areas", nargs="+", default=list(AREA_STATIONS),
                    choices=list(AREA_STATIONS))
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between price calls")
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    ap.add_argument("--find-station", metavar="NAME",
                    help="look up SMHI station keys by name and exit")
    args = ap.parse_args()

    if args.find_station:
        find_station(args.find_station)
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    outdir = Path(args.outdir)

    print(f"Fetching {', '.join(args.areas)} from {start} to {end}", file=sys.stderr)

    print("prices...", file=sys.stderr)
    prices = fetch_prices(args.areas, start, end, args.pause)

    print("weather...", file=sys.stderr)
    weather = fetch_weather(args.areas, start, end)

    price_path = outdir / "historik_elpris.csv.gz"
    weather_path = outdir / "historik_vader.csv.gz"
    prices.to_csv(price_path, index=False, compression="gzip")
    weather.to_csv(weather_path, sep=";", index=False, compression="gzip", encoding="utf-8")

    print(f"\nprices  {len(prices):>8,} rows -> {price_path}")
    print(f"weather {len(weather):>8,} rows -> {weather_path}")
    print("\nThese are now real measurements. Re-run model/train.py.")


if __name__ == "__main__":
    main()
