#!/usr/bin/env python3
"""Download hourly weather for the ARC from Open-Meteo (free, no key; CC BY 4.0).

Two archives, both "what the forecast said at the time", so no future leakage:
  hist_forecast  historical-forecast-api: stitched from the first hours of each model run
                 (~ what you'd see if you looked at a weather app right now).
                 Used for the 1h / 3h horizons.
  prev_day1      previous-runs-api, *_previous_day1: the forecast issued ~24h earlier.
                 Used for the 24h horizon, so the model only sees a day-ahead forecast.

Data by Open-Meteo.com (https://open-meteo.com/), licensed CC BY 4.0.

Usage:
  python weather.py --start 2026-10-25 --end 2026-12-13 --out external/
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 40.1013, -88.2405           # ARC
TZ = "America/Chicago"
VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
        "wind_speed_10m", "cloud_cover", "weather_code"]
ENDPOINTS = {
    "hist_forecast": ("https://historical-forecast-api.open-meteo.com/v1/forecast", ""),
    "prev_day1": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1"),
}


def fetch(url, suffix, start, end):
    params = {"latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
              "timezone": TZ, "hourly": ",".join(v + suffix for v in VARS)}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h).rename(columns={v + suffix: v for v in VARS})
    df["time"] = pd.to_datetime(df["time"])             # naive local (America/Chicago)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-25")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--out", default=str(Path(__file__).parent / "external"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, suffix) in ENDPOINTS.items():
        df = fetch(url, suffix, a.start, a.end)
        df.to_csv(out / f"weather_{name}.csv", index=False)
        print(f"{name}: {len(df)} hours, {df.time.min()} -> {df.time.max()}, "
              f"missing temp={df.temperature_2m.isna().sum()}")


if __name__ == "__main__":
    main()
