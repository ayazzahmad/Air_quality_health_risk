"""Download daily historical weather for every city in data/city_day.csv.

Source: Open-Meteo historical weather API (ERA5 reanalysis), free and keyless.
https://open-meteo.com/en/docs/historical-weather-api

Run once; the result is committed so training never needs network access:

    python scripts/fetch_weather.py

Writes
    data/weather_daily.csv       one row per (City, Date)
    data/weather_locations.csv   requested vs. resolved grid point per city

All values are for the local (Asia/Kolkata) calendar day, matching the
pollution data. Weather for day t is only known at the end of day t, which is
the same availability rule the pollution features follow.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import config as cfg  # noqa: E402

API_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DAILY = PROJECT_ROOT / "data" / "weather_daily.csv"
OUT_LOCATIONS = PROJECT_ROOT / "data" / "weather_locations.csv"

# City-centre coordinates, checked against the Open-Meteo geocoder (all within
# ~5 km; an ERA5 grid cell is ~25 km wide). Jorapokhar is the CPCB station at
# "Jorapokhar, Jharia" in Dhanbad district, not the West Singhbhum village of
# the same name that geocoders return first.
CITIES = {
    "Ahmedabad":          (23.02, 72.57),
    "Aizawl":             (23.73, 92.72),
    "Amaravati":          (16.51, 80.52),
    "Amritsar":           (31.63, 74.87),
    "Bengaluru":          (12.97, 77.59),
    "Bhopal":             (23.26, 77.41),
    "Brajrajnagar":       (21.82, 83.92),
    "Chandigarh":         (30.73, 76.78),
    "Chennai":            (13.08, 80.27),
    "Coimbatore":         (11.02, 76.96),
    "Delhi":              (28.61, 77.21),
    "Ernakulam":          (9.98, 76.28),
    "Gurugram":           (28.46, 77.03),
    "Guwahati":           (26.14, 91.74),
    "Hyderabad":          (17.39, 78.49),
    "Jaipur":             (26.91, 75.79),
    "Jorapokhar":         (23.70, 86.41),
    "Kochi":              (9.93, 76.27),
    "Kolkata":            (22.57, 88.36),
    "Lucknow":            (26.85, 80.95),
    "Mumbai":             (19.08, 72.88),
    "Patna":              (25.59, 85.14),
    "Shillong":           (25.58, 91.89),
    "Talcher":            (20.95, 85.23),
    "Thiruvananthapuram": (8.52, 76.94),
    "Visakhapatnam":      (17.69, 83.22),
}

# Daily aggregates provided by the API, renamed to short column names
DAILY_VARS = {
    "temperature_2m_mean":         "temp_mean",
    "temperature_2m_max":          "temp_max",
    "temperature_2m_min":          "temp_min",
    "precipitation_sum":           "precip_sum",
    "precipitation_hours":         "precip_hours",
    "wind_speed_10m_mean":         "wind_mean",
    "wind_speed_10m_max":          "wind_max",
    "wind_direction_10m_dominant": "wind_dir",
    "relative_humidity_2m_mean":   "humidity_mean",
    "surface_pressure_mean":       "pressure_mean",
}
# Boundary-layer (mixing) height is only available hourly; a shallow layer
# traps pollution near the ground, so it is aggregated to a daily mean and max.
HOURLY_VARS = ["boundary_layer_height"]


def _get(params: dict, retries: int = 5) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry in {wait}s ({exc})")
            time.sleep(wait)


def fetch_city(city: str, lat: float, lon: float, start: str, end: str):
    base = {"latitude": lat, "longitude": lon, "start_date": start,
            "end_date": end, "timezone": "Asia/Kolkata"}

    daily_json = _get({**base, "daily": ",".join(DAILY_VARS)})
    daily = (pd.DataFrame(daily_json["daily"])
             .rename(columns={"time": "Date", **DAILY_VARS}))

    hourly_json = _get({**base, "hourly": ",".join(HOURLY_VARS)})
    hourly = pd.DataFrame(hourly_json["hourly"])
    hourly["Date"] = hourly["time"].str[:10]
    blh = (hourly.groupby("Date")["boundary_layer_height"]
           .agg(blh_mean="mean", blh_max="max").reset_index())

    df = daily.merge(blh, on="Date", how="left")
    df.insert(0, "City", city)

    location = {
        "City": city, "requested_lat": lat, "requested_lon": lon,
        "grid_lat": round(daily_json["latitude"], 4),
        "grid_lon": round(daily_json["longitude"], 4),
        "grid_elevation_m": daily_json.get("elevation"),
    }
    return df, location


def main() -> int:
    pollution = pd.read_csv(cfg.DATA_PATH, usecols=["City", "Date"], parse_dates=["Date"])
    start = pollution["Date"].min().date().isoformat()
    end = pollution["Date"].max().date().isoformat()

    missing = sorted(set(pollution["City"]) - set(CITIES))
    if missing:
        print(f"No coordinates for: {missing}")
        return 1

    print(f"Fetching {len(CITIES)} cities, {start} -> {end}")
    frames, locations = [], []
    for city, (lat, lon) in CITIES.items():
        print(f"  {city}")
        df, loc = fetch_city(city, lat, lon, start, end)
        frames.append(df)
        locations.append(loc)
        time.sleep(1)  # stay well inside the free API's rate limits

    weather = pd.concat(frames, ignore_index=True)
    weather.to_csv(OUT_DAILY, index=False)
    pd.DataFrame(locations).to_csv(OUT_LOCATIONS, index=False)
    print(f"Saved {len(weather):,} rows -> {OUT_DAILY.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
