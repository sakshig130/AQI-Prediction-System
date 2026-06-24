"""
src/data_collector.py  —  FIXED VERSION
Fixes all 5 merge bugs. Replace your existing data_collector.py with this file.
"""

import os, time, shutil, logging, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

OWM_KEY    = os.environ.get("OWM_API_KEY")
LAT, LON   = 23.7957, 86.4304
BASE_URL   = "https://api.openweathermap.org"
IST_OFFSET = timedelta(hours=5, minutes=30)   # BUG 1 FIX

DATA_DIR   = Path(__file__).parent.parent / "data"
MASTER_CSV = DATA_DIR / "master_data.csv"
BACKUP_DIR = DATA_DIR / "backups"

# Exact column order matching original dataset — DO NOT CHANGE ORDER
COLUMN_ORDER = ["Timestamp","AQI","CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3","Temperature","Humidity","WindSpeed"]

# Dhanbad monthly climate normals from your training dataset (fallback weather)
_TEMP = {1:16,2:19,3:25,4:31,5:34,6:32,7:29,8:28,9:28,10:26,11:21,12:17}
_HUM  = {1:70,2:62,3:48,4:38,5:42,6:74,7:86,8:85,9:78,10:65,11:63,12:67}
_WIND = {1:1.2,2:1.5,3:2.0,4:2.3,5:2.1,6:2.8,7:2.6,8:2.4,9:1.8,10:1.4,11:1.2,12:1.1}


def _get(endpoint, params):
    if not OWM_KEY:
        raise RuntimeError("OWM_API_KEY is not set. Add it to your environment or .env file.")
    params["appid"] = OWM_KEY
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if r.status_code == 429:
                log.warning("Rate limited — waiting 65s"); time.sleep(65)
            else:
                raise
        except requests.ConnectionError:
            log.warning(f"Connection error, retry {attempt+1}/3"); time.sleep(5)
    raise RuntimeError("OWM API failed after 3 attempts")


def _unix_to_ist(unix_ts):
    """BUG 1 FIX: OWM UTC → IST naive (no tzinfo), matching original dataset format."""
    utc = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    ist = utc + IST_OFFSET
    return pd.Timestamp(ist.replace(tzinfo=None))


def fetch_air_pollution_history(start_unix, end_unix):
    """BUG 2 + 5 FIX: Uppercase column names, AQI from OWM main.aqi."""
    data = _get("/data/2.5/air_pollution/history", {"lat":LAT,"lon":LON,"start":start_unix,"end":end_unix})
    rows = []
    for item in data.get("list", []):
        c = item.get("components", {})
        rows.append({
            "Timestamp": _unix_to_ist(item["dt"]),
            "AQI":   float(item.get("main", {}).get("aqi") or np.nan),
            # BUG 2: ALL uppercase — co→CO, no→NO, no2→NO2, o3→O3, so2→SO2, pm2_5→PM2_5, pm10→PM10, nh3→NH3
            "CO":    c.get("co"),    "NO":    c.get("no"),
            "NO2":   c.get("no2"),   "O3":    c.get("o3"),
            "SO2":   c.get("so2"),   "NH3":   c.get("nh3"),
            "PM2_5": c.get("pm2_5"), "PM10":  c.get("pm10"),
        })
    log.info(f"  Air pollution rows fetched: {len(rows)}")
    return pd.DataFrame(rows)


def fetch_weather_history(start_unix, end_unix):
    """Try OWM One Call history; fall back to climate normals. BUG 3 FIX: Humidity=int."""
    rows = []
    current = start_unix
    while current < end_unix:
        try:
            data = _get("/data/3.0/onecall/timemachine", {"lat":LAT,"lon":LON,"dt":current,"units":"metric"})
            for item in data.get("hourly", []):
                if start_unix <= item["dt"] <= end_unix:
                    rows.append({
                        "Timestamp":   _unix_to_ist(item["dt"]),
                        "Temperature": item.get("temp"),
                        "Humidity":    int(round(item["humidity"])) if item.get("humidity") is not None else None,  # BUG 3
                        "WindSpeed":   item.get("wind_speed"),
                    })
        except Exception:
            pass  # silently fall through to climate normals
        current += 86400
        time.sleep(0.3)

    if rows:
        log.info(f"  Weather rows from OWM history: {len(rows)}")
        return pd.DataFrame(rows)

    # Climate normals fallback
    log.warning("  Using Dhanbad climate normals as weather fallback")
    start_ist = _unix_to_ist(start_unix)
    end_ist   = _unix_to_ist(end_unix)
    ts_range  = pd.date_range(start=start_ist, end=end_ist, freq="h")
    fallback  = [{"Timestamp":ts,"Temperature":float(_TEMP[ts.month]),
                  "Humidity":int(_HUM[ts.month]),"WindSpeed":float(_WIND[ts.month])} for ts in ts_range]
    log.info(f"  Climate normal rows generated: {len(fallback)}")
    return pd.DataFrame(fallback)


def collect_weekly_data(days_back=7, master_csv=MASTER_CSV):
    """
    Main pipeline function. Fetches past `days_back` days from OWM,
    merges with master CSV using IST timestamps, deduplicates, appends.
    All 5 bugs fixed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    now_utc    = datetime.now(timezone.utc)
    end_unix   = int(now_utc.timestamp())
    start_unix = end_unix - (days_back * 24 * 3600)

    log.info(f"Collecting {days_back} days for Dhanbad (IST window)...")
    log.info(f"  From IST: {_unix_to_ist(start_unix).date()}  To IST: {_unix_to_ist(end_unix).date()}")

    # ── Step 1: Fetch ──────────────────────────────────────────────────────────
    air_df     = fetch_air_pollution_history(start_unix, end_unix)
    weather_df = fetch_weather_history(start_unix, end_unix)

    if air_df.empty:
        log.error("No air pollution data returned — aborting")
        return 0, master_csv

    # ── Step 2: Align timestamps to hour, merge ────────────────────────────────
    air_df["Timestamp"]     = air_df["Timestamp"].dt.floor("h")
    weather_df["Timestamp"] = weather_df["Timestamp"].dt.floor("h")
    merged = pd.merge(air_df, weather_df, on="Timestamp", how="left")
    log.info(f"  Merged rows: {len(merged)}")

    # ── Step 3: BUG 4 FIX — Replace -9999 sentinel values with NaN ────────────
    sentinel_count = 0
    for col in ["CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3"]:
        if col in merged.columns:
            n = (merged[col] == -9999).sum()
            if n > 0:
                merged[col] = merged[col].replace(-9999, np.nan)
                sentinel_count += n
    if sentinel_count:
        log.warning(f"  Replaced {sentinel_count} sentinel -9999 values with NaN")
    # Also drop physically impossible PM values
    merged = merged[merged["PM10"].fillna(1) > -100].copy()

    # ── Step 4: BUG 3 FIX — Ensure Humidity is integer ────────────────────────
    merged["Humidity"] = pd.to_numeric(merged["Humidity"], errors="coerce").round().astype("Int64")

    # ── Step 5: Add any missing columns and enforce column order ───────────────
    for col in COLUMN_ORDER:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[COLUMN_ORDER].copy()

    # ── Step 6: Load or create master CSV ─────────────────────────────────────
    if master_csv.exists():
        existing = pd.read_csv(master_csv, dtype={"Humidity": "Int64"})
        existing["Timestamp"] = pd.to_datetime(existing["Timestamp"])
        log.info(f"  Existing master CSV: {len(existing):,} rows  "
                 f"({existing['Timestamp'].min().date()} → {existing['Timestamp'].max().date()})")
    else:
        # First run — seed from original training dataset
        original = DATA_DIR / "final_clean_merged_dataset.csv"
        if original.exists():
            existing = pd.read_csv(original, dtype={"Humidity": "Int64"})
            existing["Timestamp"] = pd.to_datetime(existing["Timestamp"])
            # BUG 4: Clean -9999 from original dataset too
            for col in ["CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3"]:
                if col in existing.columns:
                    existing[col] = existing[col].replace(-9999, np.nan)
            existing.to_csv(master_csv, index=False)
            log.info(f"  Seeded master CSV from original dataset: {len(existing):,} rows")
        else:
            existing = pd.DataFrame(columns=COLUMN_ORDER)
            log.warning("  Original dataset not found — starting fresh master CSV")

    # ── Step 7: BUG 1 FIX — Deduplicate using IST timestamps (both sides) ─────
    # Both existing and merged now use IST naive timestamps — comparison is correct
    existing_ts = set(pd.to_datetime(existing["Timestamp"]).dt.floor("h").astype(str))
    merged["_ts_key"] = merged["Timestamp"].dt.floor("h").astype(str)
    new_rows = merged[~merged["_ts_key"].isin(existing_ts)].drop(columns=["_ts_key"]).copy()
    log.info(f"  Duplicate check: {len(merged)} fetched, {len(new_rows)} are new")

    if new_rows.empty:
        log.info("  Nothing new to add — master CSV is already current")
        return 0, master_csv

    # ── Step 8: Backup and save ────────────────────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"master_backup_{ts_str}.csv"
    if master_csv.exists():
        shutil.copy(master_csv, backup)
        log.info(f"  Backed up → {backup.name}")

    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated["Timestamp"] = pd.to_datetime(updated["Timestamp"])
    updated = updated.sort_values("Timestamp").reset_index(drop=True)
    updated.to_csv(master_csv, index=False)

    log.info(f"  DONE: {len(existing):,} → {len(updated):,} rows (+{len(new_rows)})")
    log.info(f"  Date range: {updated['Timestamp'].min().date()} → {updated['Timestamp'].max().date()}")

    # Trim old backups (keep last 4 weeks)
    for old in sorted(BACKUP_DIR.glob("master_backup_*.csv"))[:-4]:
        old.unlink()

    return len(new_rows), master_csv


def validate_master_csv(master_csv=MASTER_CSV):
    """
    Run this after collect_weekly_data() to confirm the merge worked.
    Checks for: duplicate timestamps, -9999 values, dtype mismatches.
    """
    if not Path(master_csv).exists():
        print("ERROR: master CSV not found"); return

    df = pd.read_csv(master_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    dups      = df.duplicated(subset=["Timestamp"]).sum()
    sentinels = sum((df[c]==-9999).sum() for c in df.select_dtypes("number").columns)
    nulls     = df[["PM2_5","PM10","CO","NO2"]].isna().sum()

    print("\n" + "="*50)
    print("MASTER CSV VALIDATION REPORT")
    print("="*50)
    print(f"Total rows   : {len(df):,}")
    print(f"Date range   : {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
    print(f"Duplicates   : {dups}  {'✓ OK' if dups==0 else '✗ BUG 1 NOT FIXED'}")
    print(f"-9999 values : {sentinels}  {'✓ OK' if sentinels==0 else '✗ BUG 4 NOT FIXED'}")
    print(f"PM2.5 mean   : {df['PM2_5'].mean():.1f} µg/m³")
    print(f"PM10  mean   : {df['PM10'].mean():.1f} µg/m³")
    print(f"Humidity type: {df['Humidity'].dtype}  {'✓ OK' if 'int' in str(df['Humidity'].dtype).lower() else '⚠ float (minor)'}")
    print()
    print("Null counts (acceptable if OWM returned None):")
    for col, n in nulls.items():
        print(f"  {col}: {n}")
    print("="*50)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n, path = collect_weekly_data(days_back=7)
    print(f"\nAdded {n} new rows.")
    validate_master_csv(path)
