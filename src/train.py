"""
src/train.py  —  FIXED VERSION
Fixes 3 problems:
  FIX 1: early_stopping_rounds=50 added → stops overfitting
  FIX 2: Train from 2023 onwards only → removes concept drift
  FIX 3: 80/10/10 split → proper validation set for early stopping

Run:
  python train.py --mode dual        # recommended — dual PM models only
  python train.py --mode both        # trains both single + dual
  python train.py --mode dual --tune # with Optuna tuning (~30 min)
"""

import argparse, os, json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent
ROOT_DIR   = SRC_DIR.parent
DATA_PATH  = ROOT_DIR / "data" / "master_data.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(SRC_DIR))
from feature_engineering import build_features_training, get_feature_columns

# ── FIX 1: early_stopping_rounds added ───────────────────────────────────────
BASE_PARAMS = {
    "objective":            "reg:squarederror",
    "tree_method":          "hist",
    "n_estimators":         1000,          # high ceiling — early stopping cuts it
    "early_stopping_rounds": 50,           # FIX 1: stop when val doesn't improve
    "max_depth":            6,
    "learning_rate":        0.05,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "reg_alpha":            0.1,
    "reg_lambda":           1.0,
    "min_child_weight":     5,
    "random_state":         42,
    "n_jobs":               -1,
    "verbosity":            0,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test_log, label):
    pred_log  = model.predict(X_test)
    pred_orig = np.expm1(pred_log)
    y_orig    = np.expm1(y_test_log)
    rmse = float(mean_squared_error(y_orig, pred_orig) ** 0.5)
    mae  = float(mean_absolute_error(y_orig, pred_orig))
    r2   = float(r2_score(y_orig, pred_orig))
    print(f"  [{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  "
          f"(best_iter={model.best_iteration})")
    return {"rmse": rmse, "mae": mae, "r2": r2,
            "best_iteration": model.best_iteration}


def load_and_filter(data_path, cutoff_year=2023):
    """
    FIX 2: Load data and keep only rows from cutoff_year onwards.
    This removes concept drift from 2020-2022 when Dhanbad was
    52% more polluted than today.
    """
    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    total = len(df)
    df = df[df["Timestamp"].dt.year >= cutoff_year].copy()
    print(f"  Loaded {total:,} total rows → kept {len(df):,} rows "
          f"(from {cutoff_year} onwards)  [FIX 2: concept drift removed]")

    # FIX 4: Replace -9999 sentinels with NaN
    for col in ["CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3"]:
        if col in df.columns:
            df[col] = df[col].replace(-9999, np.nan)

    return df


def split_80_10_10(X, y):
    """
    FIX 3: Proper 80/10/10 time-based split.
    - 80% train
    - 10% validation  → used for early stopping
    - 10% test        → used for final honest RMSE
    Never shuffle time-series data!
    """
    n      = len(X)
    s1     = int(n * 0.80)
    s2     = int(n * 0.90)
    X_train = X.iloc[:s1]
    X_val   = X.iloc[s1:s2]
    X_test  = X.iloc[s2:]
    y_train = y.iloc[:s1]
    y_val   = y.iloc[s1:s2]
    y_test  = y.iloc[s2:]
    print(f"  Split: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}  [FIX 3]")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def optuna_tune(X_train, y_train, X_val, y_val, n_trials=60):
    print(f"  Running Optuna ({n_trials} trials)...")
    def objective(trial):
        params = {
            "objective":            "reg:squarederror",
            "tree_method":          "hist",
            "n_estimators":         trial.suggest_int("n_estimators", 200, 1000),
            "early_stopping_rounds": 30,
            "max_depth":            trial.suggest_int("max_depth", 3, 9),
            "learning_rate":        trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":            trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":     trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":            trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":           trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":     trial.suggest_int("min_child_weight", 1, 20),
            "random_state":         42,
            "verbosity":            0,
            "n_jobs":               -1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_val)
        return float(mean_squared_error(y_val, pred) ** 0.5)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"objective":"reg:squarederror","tree_method":"hist",
                 "early_stopping_rounds":50,"random_state":42,
                 "n_jobs":-1,"verbosity":0})
    print(f"  Best val RMSE (log scale): {study.best_value:.4f}")
    return best


# ── Train dual models ─────────────────────────────────────────────────────────

def train_dual(df, tune=False):
    print("\n=== Training Dual PM Models (PM2.5 + PM10) ===")
    df_feat   = build_features_training(df, mode="dual")
    feat_cols = get_feature_columns(df_feat, mode="dual")
    results   = {}
    models    = {}

    for target in ["PM2_5", "PM10"]:
        print(f"\n  --- Target: {target} ---")
        X = df_feat[feat_cols]
        y = np.log1p(df_feat[target])

        # FIX 3: 80/10/10 split
        X_train, X_val, X_test, y_train, y_val, y_test = split_80_10_10(X, y)

        params = BASE_PARAMS.copy()
        if tune and OPTUNA_AVAILABLE:
            params = optuna_tune(X_train, y_train, X_val, y_val)
        elif tune:
            print("  Optuna not installed — skipping tuning. Run: pip install optuna")

        model = xgb.XGBRegressor(**params)
        # FIX 1: eval_set with separate val set enables early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        metrics = evaluate(model, X_test, y_test, target)
        results[target] = metrics
        models[target]  = model

        fname = f"model_dual_{target.lower()}.joblib"
        joblib.dump(model, MODELS_DIR / fname)
        print(f"  Saved → models/{fname}")

    meta = {
        "mode":            "dual",
        "targets":         ["PM2_5", "PM10"],
        "log_transform":   True,
        "feature_cols":    feat_cols,
        "metrics":         results,
        "training_cutoff": "2023-01-01",
        "trained_at":      pd.Timestamp.now().isoformat(),
        "rows_used":       len(df_feat),
        "data_range": {
            "start": str(df["Timestamp"].min().date()),
            "end":   str(df["Timestamp"].max().date()),
        },
    }
    with open(MODELS_DIR / "model_dual_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("DUAL MODEL SUMMARY")
    print("="*50)
    for t, m in results.items():
        r2_status = "✓ Excellent" if m["r2"] > 0.90 else "⚠ Check" if m["r2"] > 0.70 else "✗ Poor"
        print(f"  {t:<8} RMSE={m['rmse']:.2f} µg/m³  R²={m['r2']:.4f}  {r2_status}")
    print("="*50)
    return models, feat_cols, results


# ── Train single AQI model ────────────────────────────────────────────────────

def train_single(df, tune=False):
    print("\n=== Training Single AQI Model ===")
    print("  NOTE: Dual models (R²=0.97) are recommended over this.")
    print("  Single AQI model is harder to train accurately due to")
    print("  CPCB formula non-linearity. Use dual model for production.\n")

    df_feat   = build_features_training(df, mode="single")
    feat_cols = get_feature_columns(df_feat, mode="single")

    X = df_feat[feat_cols]
    y = np.log1p(df_feat["AQI_computed"])

    # FIX 3: 80/10/10 split
    X_train, X_val, X_test, y_train, y_val, y_test = split_80_10_10(X, y)

    params = BASE_PARAMS.copy()
    if tune and OPTUNA_AVAILABLE:
        params = optuna_tune(X_train, y_train, X_val, y_val)

    model = xgb.XGBRegressor(**params)
    # FIX 1: proper eval_set for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    metrics = evaluate(model, X_test, y_test, "AQI_computed")

    # Category accuracy on original scale
    pred_log  = model.predict(X_test)
    pred_orig = np.expm1(pred_log)
    y_orig    = np.expm1(y_test)

    def cat(v):
        if v <= 50:  return "Good"
        if v <= 100: return "Satisfactory"
        if v <= 200: return "Moderate"
        if v <= 300: return "Poor"
        if v <= 400: return "Very Poor"
        return "Severe"

    cat_acc = sum(cat(a)==cat(b) for a,b in zip(y_orig,pred_orig)) / len(y_orig)
    print(f"  Category accuracy (CPCB): {cat_acc:.3f} ({cat_acc*100:.1f}%)")
    metrics["category_accuracy"] = cat_acc

    joblib.dump(model, MODELS_DIR / "model_single_aqi.joblib")
    meta = {
        "mode": "single", "target": "AQI_computed",
        "log_transform": True, "feature_cols": feat_cols,
        "metrics": metrics, "training_cutoff": "2023-01-01",
        "trained_at": pd.Timestamp.now().isoformat(),
        "rows_used": len(df_feat),
        "data_range": {
            "start": str(df["Timestamp"].min().date()),
            "end":   str(df["Timestamp"].max().date()),
        },
    }
    with open(MODELS_DIR / "model_single_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → models/model_single_aqi.joblib")
    return model, feat_cols, metrics


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dual",
                        choices=["single","dual","both"],
                        help="Which model(s) to train")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning (~30 min)")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="Path to training CSV")
    parser.add_argument("--from-year", type=int, default=2023,
                        help="Only train on data from this year onwards (default: 2023)")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_and_filter(args.data, cutoff_year=args.from_year)
    print(f"Date range: {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
    print(f"PM2.5 mean: {df['PM2_5'].mean():.1f} µg/m³ (test set should be similar)\n")

    if args.mode in ("dual", "both"):
        train_dual(df, tune=args.tune)

    if args.mode in ("single", "both"):
        train_single(df, tune=args.tune)

    print("\n✅ Training complete. Models saved to:", MODELS_DIR)
    print("\nRecommended next step:")
    print("  Restart your API server to load the new models:")
    print("  cd api && uvicorn main:app --port 8000 --reload")
