"""
api/main.py  —  v3 with hot-swap reload support
The ModelStore now checks for a .reload_needed flag file on every request.
When the scheduler deploys new models, it writes that flag.
The next API request triggers an in-memory reload — zero downtime, zero restart.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import (
    build_inference_row, get_feature_columns, add_computed_aqi,
    add_time_features, encode_wind_direction, add_lag_features,
    BREAKPOINTS, _sub_index,
)
from owm_client import (
    get_current_weather, get_current_air_pollution,
    get_merged_forecast, aqi_to_category, aqi_health_message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Dhanbad AQI API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODELS_DIR  = Path(__file__).parent.parent / "models"
DATA_DIR    = Path(__file__).parent.parent / "data"
RELOAD_FLAG = MODELS_DIR / ".reload_needed"

DHANBAD_MONTHLY_PM25 = {1:130,2:110,3:85,4:70,5:65,6:45,7:35,8:30,9:40,10:75,11:120,12:145}
DHANBAD_MONTHLY_PM10 = {1:155,2:130,3:105,4:90,5:85,6:60,7:50,8:45,9:55,10:95,11:145,12:170}


# ── Model Store with hot-swap ─────────────────────────────────────────────────

class ModelStore:
    def __init__(self):
        self.dual_pm25  = None
        self.dual_pm10  = None
        self.dual_meta  = None
        self._reload_ts = None
        self.load()

    def load(self):
        try:
            self.dual_pm25 = joblib.load(MODELS_DIR / "model_dual_pm2_5.joblib")
            self.dual_pm10 = joblib.load(MODELS_DIR / "model_dual_pm10.joblib")
            with open(MODELS_DIR / "model_dual_meta.json") as f:
                self.dual_meta = json.load(f)
            self._reload_ts = datetime.now().isoformat()
            log.info(f"Models loaded at {self._reload_ts}")
            trained_at = self.dual_meta.get("trained_at", "unknown")
            rows_used  = self.dual_meta.get("rows_used", "?")
            log.info(f"  Trained at: {trained_at} on {rows_used} rows")
        except FileNotFoundError as e:
            log.error(f"Model missing: {e} — run: python train.py --mode dual")

    def check_and_reload(self):
        """
        Called on every API request. If the scheduler wrote a reload flag,
        reload models in-place. Takes ~2 seconds, zero downtime.
        """
        if not RELOAD_FLAG.exists():
            return
        flag_ts = RELOAD_FLAG.read_text().strip()
        if flag_ts == self._reload_ts:
            return   # Already reloaded for this flag

        log.info(f"Hot-swap triggered by scheduler (flag: {flag_ts}) — reloading models...")
        self.load()
        RELOAD_FLAG.unlink(missing_ok=True)
        log.info("Hot-swap complete — new models now active")


store = ModelStore()


# ── Pydantic models ───────────────────────────────────────────────────────────

class AQIPrediction(BaseModel):
    timestamp:       str
    aqi_predicted:   float
    aqi_category:    str
    health_message:  str
    pm25_predicted:  float | None = None
    pm10_predicted:  float | None = None
    pm25_subindex:   float | None = None
    pm10_subindex:   float | None = None
    model_used:      str
    data_rows_used:  int | None = None
    model_trained_at: str | None = None
    confidence_note: str

class ForecastItem(BaseModel):
    timestamp:     str
    aqi_predicted: float
    aqi_category:  str
    pm25:          float | None = None
    pm10:          float | None = None

class ForecastResponse(BaseModel):
    location:     str
    generated_at: str
    forecasts:    list[ForecastItem]

class DailySummary(BaseModel):
    date:         str
    aqi_mean:     float
    aqi_max:      float
    aqi_min:      float
    dominant_cat: str

class PipelineStatus(BaseModel):
    model_trained_at:   str | None
    data_rows_used:     int | None
    data_date_range:    dict | None
    last_reload:        str | None
    training_runs:      list | None
    master_csv_rows:    int | None
    master_csv_updated: str | None


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_live_history(pm25, pm10, now, weather=None):
    rng  = np.random.default_rng(seed=42)
    rows = []
    for h in range(48, 0, -1):
        noise = 1.0 + rng.normal(0, 0.05)
        rows.append({
            "Timestamp":   now - pd.Timedelta(hours=h),
            "PM2_5":       max(0.1, pm25 * noise),
            "PM10":        max(0.1, pm10 * noise),
            "Temperature": weather["main"]["temp"] if weather else 28.0,
            "Humidity":    weather["main"]["humidity"] if weather else 55,
            "WindSpeed":   weather["wind"]["speed"] if weather else 1.5,
            "CO":0,"NO":0,"NO2":0,"O3":0,"SO2":0,"NH3":0,
        })
    df = pd.DataFrame(rows)
    df = add_computed_aqi(df)
    return df


def get_safe_pm(comp, now):
    m    = now.month
    pm25 = comp.get("pm2_5") or 0.0
    pm10 = comp.get("pm10")  or 0.0
    if pm25 < 1.0:
        pm25 = DHANBAD_MONTHLY_PM25.get(m, 70.0)
        log.warning(f"OWM PM2.5=0, using seasonal mean {pm25}")
    if pm10 < 1.0:
        pm10 = DHANBAD_MONTHLY_PM10.get(m, 90.0)
        log.warning(f"OWM PM10=0, using seasonal mean {pm10}")
    return float(pm25), float(pm10)


def predict_dual(row_df):
    if not store.dual_pm25:
        raise HTTPException(503, "Models not loaded")
    feat_cols = store.dual_meta["feature_cols"]
    X    = row_df.reindex(columns=feat_cols, fill_value=0.0)
    pm25 = float(np.expm1(store.dual_pm25.predict(X)[0]))
    pm10 = float(np.expm1(store.dual_pm10.predict(X)[0]))
    return max(0.1, min(1000, pm25)), max(0.1, min(1200, pm10))


def pm_to_aqi(pm25, pm10):
    si25 = _sub_index(pm25, BREAKPOINTS["PM2_5"])
    si10 = _sub_index(pm10, BREAKPOINTS["PM10"])
    return float(max(si25, si10)), float(si25), float(si10)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    store.check_and_reload()
    meta = store.dual_meta or {}
    return {
        "status":       "ok",
        "models_loaded": store.dual_pm25 is not None,
        "trained_at":   meta.get("trained_at"),
        "rows_used":    meta.get("rows_used"),
        "data_range":   meta.get("data_range"),
        "last_reload":  store._reload_ts,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.get("/aqi/now", response_model=AQIPrediction)
def predict_now():
    store.check_and_reload()   # Hot-swap check on every request
    now = pd.Timestamp(datetime.utcnow())

    try:
        weather = get_current_weather()
        air     = get_current_air_pollution()
        comp    = air.get("list", [{}])[0].get("components", {})
    except Exception as e:
        raise HTTPException(502, f"OWM error: {e}")

    pm25_live, pm10_live = get_safe_pm(comp, now)
    log.info(f"OWM live → PM2.5={pm25_live:.1f}  PM10={pm10_live:.1f}")

    live_hist = build_live_history(pm25_live, pm10_live, now, weather)
    row_df = build_inference_row(
        owm_current={"main":{"temp":weather["main"]["temp"],"humidity":weather["main"]["humidity"]},
                     "wind":{"speed":weather["wind"]["speed"],"deg":weather["wind"].get("deg",0)}},
        owm_air={"list":[{"components":comp}]},
        history_df=live_hist,
        timestamp=now, mode="dual",
    )

    pm25_pred, pm10_pred = predict_dual(row_df)
    aqi, si25, si10      = pm_to_aqi(pm25_pred, pm10_pred)
    aqi = max(0.0, min(500.0, aqi))

    meta = store.dual_meta or {}
    log.info(f"PREDICTION → PM2.5={pm25_pred:.1f}  PM10={pm10_pred:.1f}  AQI={aqi:.1f}")

    return AQIPrediction(
        timestamp=now.isoformat(),
        aqi_predicted=round(aqi, 1),
        aqi_category=aqi_to_category(aqi),
        health_message=aqi_health_message(aqi),
        pm25_predicted=round(pm25_pred, 2),
        pm10_predicted=round(pm10_pred, 2),
        pm25_subindex=round(si25, 1),
        pm10_subindex=round(si10, 1),
        model_used="dual_xgboost_cpcb_v3",
        data_rows_used=meta.get("rows_used"),
        model_trained_at=meta.get("trained_at"),
        confidence_note=(
            f"Trained on {meta.get('rows_used','?')} rows "
            f"({meta.get('data_range',{}).get('start','?')[:10]} to "
            f"{meta.get('data_range',{}).get('end','?')[:10]}). "
            f"Auto-retrains weekly."
        ),
    )


@app.get("/aqi/forecast", response_model=ForecastResponse)
def predict_forecast():
    store.check_and_reload()
    try:
        forecast_df = get_merged_forecast()
    except Exception as e:
        raise HTTPException(502, f"OWM forecast error: {e}")

    now = pd.Timestamp(datetime.utcnow())
    try:
        air      = get_current_air_pollution()
        comp_now = air.get("list",[{}])[0].get("components",{})
    except Exception:
        comp_now = {}

    pm25_seed, pm10_seed = get_safe_pm(comp_now, now)
    running_history = build_live_history(pm25_seed, pm10_seed, now)
    results = []

    for _, frow in forecast_df.iterrows():
        ts = frow["Timestamp"]
        fp25 = frow.get("PM2_5") or pm25_seed
        fp10 = frow.get("PM10")  or pm10_seed
        if fp25 < 1: fp25 = pm25_seed
        if fp10 < 1: fp10 = pm10_seed

        try:
            row_df = build_inference_row(
                owm_current={"main":{"temp":frow.get("Temperature",28),"humidity":frow.get("Humidity",55)},
                             "wind":{"speed":frow.get("WindSpeed",1.5),"deg":frow.get("WindDeg",0)}},
                owm_air={"list":[{"components":{"pm2_5":fp25,"pm10":fp10,
                    "co":frow.get("CO",0),"no":frow.get("NO",0),"no2":frow.get("NO2",0),
                    "o3":frow.get("O3",0),"so2":frow.get("SO2",0),"nh3":frow.get("NH3",0)}}]},
                history_df=running_history,
                timestamp=ts, mode="dual",
            )
        except Exception:
            continue

        pm25_pred, pm10_pred = predict_dual(row_df)
        aqi, _, _            = pm_to_aqi(pm25_pred, pm10_pred)
        aqi = max(0.0, min(500.0, aqi))

        new_row = pd.DataFrame([{"Timestamp":ts,"PM2_5":pm25_pred,"PM10":pm10_pred,
            "AQI_computed":aqi,"Temperature":frow.get("Temperature",28),
            "Humidity":frow.get("Humidity",55),"WindSpeed":frow.get("WindSpeed",1.5),
            "CO":0,"NO":0,"NO2":0,"O3":0,"SO2":0,"NH3":0}])
        running_history = pd.concat([running_history, new_row], ignore_index=True)
        pm25_seed, pm10_seed = pm25_pred, pm10_pred

        results.append(ForecastItem(
            timestamp=ts.isoformat(),
            aqi_predicted=round(aqi, 1),
            aqi_category=aqi_to_category(aqi),
            pm25=round(pm25_pred, 2),
            pm10=round(pm10_pred, 2),
        ))

    return ForecastResponse(
        location="Dhanbad, Jharkhand, India",
        generated_at=datetime.utcnow().isoformat(),
        forecasts=results,
    )


@app.get("/aqi/daily", response_model=list[DailySummary])
def daily_summary():
    forecast = predict_forecast()
    df = pd.DataFrame([f.model_dump() for f in forecast.forecasts])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["aqi"]  = df["aqi_predicted"]
    summaries  = []
    for date, group in df.groupby("date"):
        summaries.append(DailySummary(
            date=str(date), aqi_mean=round(group["aqi"].mean(),1),
            aqi_max=round(group["aqi"].max(),1), aqi_min=round(group["aqi"].min(),1),
            dominant_cat=aqi_to_category(group["aqi"].mean()),
        ))
    return summaries


@app.get("/pipeline/status", response_model=PipelineStatus)
def pipeline_status():
    """See how many rows the model was trained on, when, and training history."""
    store.check_and_reload()
    meta     = store.dual_meta or {}
    log_file = MODELS_DIR / "training_log.json"
    runs     = json.loads(log_file.read_text()) if log_file.exists() else []

    # Count master CSV rows
    master_csv = DATA_DIR / "master_data.csv"
    csv_rows   = None
    csv_updated = None
    if master_csv.exists():
        import os
        csv_rows    = sum(1 for _ in open(master_csv)) - 1
        csv_updated = datetime.fromtimestamp(master_csv.stat().st_mtime).isoformat()

    return PipelineStatus(
        model_trained_at   = meta.get("trained_at"),
        data_rows_used     = meta.get("rows_used"),
        data_date_range    = meta.get("data_range"),
        last_reload        = store._reload_ts,
        training_runs      = runs[-5:],   # Last 5 weekly runs
        master_csv_rows    = csv_rows,
        master_csv_updated = csv_updated,
    )


@app.get("/debug/owm")
def debug_owm():
    """Check what OWM returns live — useful for debugging PM values."""
    store.check_and_reload()
    try:
        weather = get_current_weather()
        air     = get_current_air_pollution()
        comp    = air.get("list",[{}])[0].get("components",{})
        now     = pd.Timestamp(datetime.utcnow())
        pm25, pm10 = get_safe_pm(comp, now)
        return {
            "owm_raw_pm25": comp.get("pm2_5"),
            "owm_raw_pm10": comp.get("pm10"),
            "used_pm25":    pm25, "used_pm10": pm10,
            "temperature":  weather["main"]["temp"],
            "humidity":     weather["main"]["humidity"],
            "source": "live" if (comp.get("pm2_5") or 0) > 1 else "seasonal_fallback",
        }
    except Exception as e:
        raise HTTPException(502, str(e))
