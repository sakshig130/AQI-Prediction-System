"""
feature_engineering.py
Dhanbad AQI Prediction — Feature Engineering Pipeline
Handles: CPCB AQI computation, lag/rolling features, cyclical encoding,
         seasonal flags, and inference-time feature construction from
         OpenWeatherMap API response.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. CPCB AQI Sub-index Formula (India standard)
# ──────────────────────────────────────────────

BREAKPOINTS = {
    "PM2_5": [
        (0,   30,   0,   50),
        (30,  60,   50,  100),
        (60,  90,   100, 200),
        (90,  120,  200, 300),
        (120, 250,  300, 400),
        (250, 500,  400, 500),
    ],
    "PM10": [
        (0,   50,   0,   50),
        (50,  100,  50,  100),
        (100, 250,  100, 200),
        (250, 350,  200, 300),
        (350, 430,  300, 400),
        (430, 600,  400, 500),
    ],
    "NO2": [
        (0,   40,   0,   50),
        (40,  80,   50,  100),
        (80,  180,  100, 200),
        (180, 280,  200, 300),
        (280, 400,  300, 400),
        (400, 800,  400, 500),
    ],
    "SO2": [
        (0,   40,   0,   50),
        (40,  80,   50,  100),
        (80,  380,  100, 200),
        (380, 800,  200, 300),
        (800, 1600, 300, 400),
        (1600,2100, 400, 500),
    ],
    "CO": [
        (0,    1000,  0,   50),
        (1000, 2000,  50,  100),
        (2000, 10000, 100, 200),
        (10000,17000, 200, 300),
        (17000,34000, 300, 400),
        (34000,50000, 400, 500),
    ],
    "O3": [
        (0,   50,   0,   50),
        (50,  100,  50,  100),
        (100, 168,  100, 200),
        (168, 208,  200, 300),
        (208, 748,  300, 400),
        (748, 1000, 400, 500),
    ],
}


def _sub_index(value: float, breakpoints: list) -> float:
    """Compute pollutant sub-index using linear interpolation between breakpoints."""
    if value < 0:
        return np.nan
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= value <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (value - bp_lo) + i_lo
    # Beyond highest breakpoint → cap at 500
    return 500.0


def compute_aqi(row: pd.Series) -> float:
    """Compute overall AQI as max of all available pollutant sub-indices."""
    sub_indices = []
    for pollutant, bps in BREAKPOINTS.items():
        col = pollutant  # e.g. PM2_5, PM10, NO2 ...
        if col in row and not pd.isna(row[col]):
            si = _sub_index(float(row[col]), bps)
            if not np.isnan(si):
                sub_indices.append(si)
    return max(sub_indices) if sub_indices else np.nan


def add_computed_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'AQI_computed' column using CPCB formula from pollutant columns."""
    df = df.copy()
    df["AQI_computed"] = df.apply(compute_aqi, axis=1)
    return df


# ──────────────────────────────────────────────
# 2. Time & Seasonal Features
# ──────────────────────────────────────────────

# Approximate Jharkhand festival/event dates that spike pollution
# (Diwali, Chhath Puja, crop burning season)
HIGH_POLLUTION_MONTH_DAYS = {
    (10, 20), (10, 21), (10, 22),  # Diwali window
    (11, 1), (11, 2), (11, 3),     # Chhath Puja window
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["Timestamp"])

    df["hour"]       = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_year"] = ts.dt.dayofyear
    df["month"]      = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding (avoids discontinuity at 23→0, Dec→Jan)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Jharkhand seasons
    # Winter (haze peak): Nov–Feb | Summer: Mar–May | Monsoon: Jun–Sep | Post-monsoon: Oct
    df["is_winter"]   = ts.dt.month.isin([11, 12, 1, 2]).astype(int)
    df["is_monsoon"]  = ts.dt.month.isin([6, 7, 8, 9]).astype(int)

    # Peak pollution hours (morning commute + cooking: 6-10, evening: 18-22)
    df["is_peak_hour"] = ts.dt.hour.apply(
        lambda h: 1 if (6 <= h <= 10 or 18 <= h <= 22) else 0
    )

    # Festival flag
    df["is_festival"] = ts.apply(
        lambda t: 1 if (t.month, t.day) in HIGH_POLLUTION_MONTH_DAYS else 0
    )

    return df


# ──────────────────────────────────────────────
# 3. Lag & Rolling Features
# ──────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    Add lag and rolling window features.
    Assumes df is sorted by Timestamp ascending.
    targets: list of column names to create lags for (e.g. ['PM2_5','PM10','AQI_computed'])
    """
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)

    lag_hours = [1, 3, 6, 12, 24, 48]
    roll_windows = [6, 12, 24]

    for col in targets:
        if col not in df.columns:
            continue
        for lag in lag_hours:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
        for w in roll_windows:
            df[f"{col}_roll{w}h_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll{w}h_std"]  = df[col].shift(1).rolling(w).std()

    return df


# ──────────────────────────────────────────────
# 4. Wind Direction Encoding
# ──────────────────────────────────────────────

def encode_wind_direction(df: pd.DataFrame, col: str = "WindDeg") -> pd.DataFrame:
    """Convert wind direction (degrees 0–360) to sin/cos components."""
    df = df.copy()
    if col in df.columns:
        rad = np.deg2rad(df[col])
        df["wind_sin"] = np.sin(rad)
        df["wind_cos"] = np.cos(rad)
    return df


# ──────────────────────────────────────────────
# 5. Full Training Pipeline
# ──────────────────────────────────────────────

LAG_TARGETS_SINGLE  = ["AQI_computed"]
LAG_TARGETS_DUAL    = ["PM2_5", "PM10"]

FEATURE_COLS_BASE = [
    "Temperature", "Humidity", "WindSpeed",
    "CO", "NO", "NO2", "O3", "SO2", "NH3",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "day_of_year",
    "is_weekend", "is_winter", "is_monsoon",
    "is_peak_hour", "is_festival",
]


def build_features_training(df: pd.DataFrame, mode: str = "single") -> pd.DataFrame:
    """
    Full feature engineering pipeline for training data.
    mode: 'single' → add AQI lags | 'dual' → add PM2.5 & PM10 lags
    """
    df = add_computed_aqi(df)

    # Drop the bad PM10 sentinel value
    df = df[df["PM10"] > -999].copy()

    df = add_time_features(df)
    df = encode_wind_direction(df)

    lag_targets = LAG_TARGETS_SINGLE if mode == "single" else LAG_TARGETS_DUAL
    df = add_lag_features(df, lag_targets)

    # Drop rows where lags are NaN (first 48h)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame, mode: str = "single") -> list[str]:
    """Return the list of feature column names used for training/inference."""
    lag_targets = LAG_TARGETS_SINGLE if mode == "single" else LAG_TARGETS_DUAL
    lag_cols = []
    for col in lag_targets:
        for lag in [1, 3, 6, 12, 24, 48]:
            lag_cols.append(f"{col}_lag{lag}h")
        for w in [6, 12, 24]:
            lag_cols += [f"{col}_roll{w}h_mean", f"{col}_roll{w}h_std"]

    wind_cols = ["wind_sin", "wind_cos"] if "wind_sin" in df.columns else []
    return FEATURE_COLS_BASE + wind_cols + lag_cols


# ──────────────────────────────────────────────
# 6. Inference Feature Builder (from OpenWeatherMap API)
# ──────────────────────────────────────────────

def build_inference_row(
    owm_current: dict,
    owm_air: dict,
    history_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    mode: str = "single"
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference.

    owm_current : dict from OWM /weather endpoint
    owm_air     : dict from OWM /air_pollution endpoint
    history_df  : recent historical data (last 48h minimum), same schema as training
    timestamp   : prediction target timestamp
    mode        : 'single' or 'dual'
    """
    # Extract weather from OWM
    row = {
        "Timestamp":   timestamp,
        "Temperature": owm_current.get("main", {}).get("temp", np.nan),
        "Humidity":    owm_current.get("main", {}).get("humidity", np.nan),
        "WindSpeed":   owm_current.get("wind", {}).get("speed", np.nan),
        "WindDeg":     owm_current.get("wind", {}).get("deg", 0),
    }

    # Extract air quality components from OWM air_pollution endpoint
    comp = owm_air.get("list", [{}])[0].get("components", {})
    row["CO"]  = comp.get("co",  np.nan)
    row["NO"]  = comp.get("no",  np.nan)
    row["NO2"] = comp.get("no2", np.nan)
    row["O3"]  = comp.get("o3",  np.nan)
    row["SO2"] = comp.get("so2", np.nan)
    row["NH3"] = comp.get("nh3", np.nan)

    # For dual mode, OWM also provides pm2_5 and pm10
    row["PM2_5"] = comp.get("pm2_5", np.nan)
    row["PM10"]  = comp.get("pm10",  np.nan)

    # Compute AQI from available pollutants
    row_series = pd.Series(row)
    row["AQI_computed"] = compute_aqi(row_series)

    df_row = pd.DataFrame([row])
    df_row = add_time_features(df_row)
    df_row = encode_wind_direction(df_row)

    # Append to history and compute lags from it
    lag_targets = LAG_TARGETS_SINGLE if mode == "single" else LAG_TARGETS_DUAL
    combined = pd.concat([history_df, df_row], ignore_index=True).sort_values("Timestamp")
    combined = add_lag_features(combined, lag_targets)

    # Return only the last row (the inference row)
    return combined.tail(1).reset_index(drop=True)
