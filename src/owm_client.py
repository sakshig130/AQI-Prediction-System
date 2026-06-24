"""
owm_client.py
OpenWeatherMap API Client for Dhanbad AQI System
Free tier: 1,000 calls/day | 60 calls/min

Fetches:
  - Current weather
  - Air pollution (PM2.5, PM10, CO, NO2, O3, SO2, NH3 directly!)
  - 5-day / 3-hour forecast (free tier)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Dhanbad, Jharkhand coordinates
DHANBAD_LAT = 23.7957
DHANBAD_LON = 86.4304

# Get from environment variable (never hardcode secrets)
OWM_API_KEY = os.environ.get("OWM_API_KEY")

BASE_URL = "https://api.openweathermap.org"

HEADERS = {"Accept": "application/json"}


def _get(endpoint: str, params: dict) -> dict:
    """Generic GET with error handling and basic retry."""
    if not OWM_API_KEY:
        raise RuntimeError("OWM_API_KEY is not set. Add it to your environment or .env file.")
    params["appid"] = OWM_API_KEY
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_URL}{endpoint}",
                params=params,
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:  # rate limit
                print(f"  Rate limited. Waiting 60s... (attempt {attempt+1})")
                time.sleep(61)
            else:
                raise e
        except requests.exceptions.ConnectionError:
            print(f"  Connection error. Retrying in 5s... (attempt {attempt+1})")
            time.sleep(5)
    raise RuntimeError("OWM API request failed after 3 attempts")


# ──────────────────────────────────────────────
# Current Conditions
# ──────────────────────────────────────────────

def get_current_weather(lat: float = DHANBAD_LAT, lon: float = DHANBAD_LON) -> dict:
    """
    Fetch current weather. Returns raw OWM response dict.
    Keys: main.temp, main.humidity, wind.speed, wind.deg, weather[0].description
    Temperatures are in Kelvin by default; use units=metric for Celsius.
    """
    data = _get("/data/2.5/weather", {
        "lat":   lat,
        "lon":   lon,
        "units": "metric",   # Celsius
    })
    return data


def get_current_air_pollution(lat: float = DHANBAD_LAT, lon: float = DHANBAD_LON) -> dict:
    """
    Fetch current air pollution from OWM Air Pollution API (FREE).
    Returns PM2.5, PM10, CO, NO, NO2, O3, SO2, NH3 directly.
    OWM AQI: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
    """
    data = _get("/data/2.5/air_pollution", {
        "lat": lat,
        "lon": lon,
    })
    return data


# ──────────────────────────────────────────────
# 5-Day Forecast (free tier: 3-hour intervals)
# ──────────────────────────────────────────────

def get_forecast_weather(lat: float = DHANBAD_LAT, lon: float = DHANBAD_LON) -> pd.DataFrame:
    """
    Fetch 5-day / 3-hour weather forecast.
    Returns a DataFrame with columns matching our feature schema.
    """
    data = _get("/data/2.5/forecast", {
        "lat":   lat,
        "lon":   lon,
        "units": "metric",
    })
    rows = []
    for item in data.get("list", []):
        rows.append({
            "Timestamp":   pd.to_datetime(item["dt"], unit="s"),
            "Temperature": item["main"]["temp"],
            "Humidity":    item["main"]["humidity"],
            "WindSpeed":   item["wind"]["speed"],
            "WindDeg":     item["wind"].get("deg", 0),
            "Pressure":    item["main"]["pressure"],
            "Clouds":      item.get("clouds", {}).get("all", 0),
            "Rain3h":      item.get("rain", {}).get("3h", 0),
        })
    return pd.DataFrame(rows)


def get_forecast_air_pollution(lat: float = DHANBAD_LAT, lon: float = DHANBAD_LON) -> pd.DataFrame:
    """
    Fetch 5-day air pollution forecast (FREE on OWM).
    Returns DataFrame with PM2.5, PM10, NO2, O3, CO, SO2, NH3.
    """
    data = _get("/data/2.5/air_pollution/forecast", {
        "lat": lat,
        "lon": lon,
    })
    rows = []
    for item in data.get("list", []):
        comp = item.get("components", {})
        rows.append({
            "Timestamp": pd.to_datetime(item["dt"], unit="s"),
            "CO":        comp.get("co",   None),
            "NO":        comp.get("no",   None),
            "NO2":       comp.get("no2",  None),
            "O3":        comp.get("o3",   None),
            "SO2":       comp.get("so2",  None),
            "NH3":       comp.get("nh3",  None),
            "PM2_5":     comp.get("pm2_5", None),
            "PM10":      comp.get("pm10",  None),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Merged Snapshot for Inference
# ──────────────────────────────────────────────

def get_current_snapshot() -> dict:
    """
    Fetch and merge current weather + air quality into a single dict
    for immediate model inference.
    """
    weather = get_current_weather()
    air     = get_current_air_pollution()

    comp = air.get("list", [{}])[0].get("components", {})

    snapshot = {
        "timestamp":   datetime.utcnow().isoformat(),
        "temperature": weather["main"]["temp"],
        "humidity":    weather["main"]["humidity"],
        "wind_speed":  weather["wind"]["speed"],
        "wind_deg":    weather["wind"].get("deg", 0),
        "pressure":    weather["main"]["pressure"],
        "weather_desc": weather["weather"][0]["description"],
        # Pollutants (µg/m³ except CO which is µg/m³ too in OWM)
        "co":    comp.get("co"),
        "no":    comp.get("no"),
        "no2":   comp.get("no2"),
        "o3":    comp.get("o3"),
        "so2":   comp.get("so2"),
        "nh3":   comp.get("nh3"),
        "pm2_5": comp.get("pm2_5"),
        "pm10":  comp.get("pm10"),
        "owm_aqi": air.get("list", [{}])[0].get("main", {}).get("aqi"),
    }
    return snapshot


def get_merged_forecast() -> pd.DataFrame:
    """
    Merge weather + air pollution forecast into one DataFrame.
    Used for generating 5-day AQI predictions.
    """
    weather_df = get_forecast_weather()
    air_df     = get_forecast_air_pollution()

    merged = pd.merge(weather_df, air_df, on="Timestamp", how="inner")
    return merged.sort_values("Timestamp").reset_index(drop=True)


# ──────────────────────────────────────────────
# Utility: map OWM AQI (1-5) to CPCB category
# ──────────────────────────────────────────────

OWM_TO_LABEL = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

CPCB_BREAKPOINTS = [50, 100, 200, 300, 400, 500]
CPCB_LABELS      = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_to_category(aqi: float) -> str:
    for bp, label in zip(CPCB_BREAKPOINTS, CPCB_LABELS):
        if aqi <= bp:
            return label
    return "Severe"


def aqi_health_message(aqi: float) -> str:
    cat = aqi_to_category(aqi)
    messages = {
        "Good":         "Air quality is good. Enjoy outdoor activities.",
        "Satisfactory": "Air is acceptable. Sensitive groups should limit prolonged outdoor exertion.",
        "Moderate":     "Sensitive individuals may experience symptoms. Reduce outdoor time.",
        "Poor":         "Everyone may experience health effects. Avoid outdoor activities.",
        "Very Poor":    "Health alert — serious effects for all. Stay indoors.",
        "Severe":       "HAZARDOUS. Do not go outside. Emergency conditions.",
    }
    return f"[{cat}] {messages.get(cat, '')}"


if __name__ == "__main__":
    # Quick test — replace with your API key in env
    print("Testing OWM client for Dhanbad...")
    snap = get_current_snapshot()
    print("Current snapshot:")
    for k, v in snap.items():
        print(f"  {k}: {v}")
