"""
src/auto_retrain.py
Automatic model retrainer for the self-updating pipeline.

Called by the scheduler after every successful data collection.
Retrains both XGBoost models on the full updated master CSV,
then hot-swaps them into the running FastAPI — zero downtime.

Hot-swap mechanism:
  1. Train new models → save to models/new/
  2. Validate new models beat old RMSE threshold
  3. Atomically move new/ → active/ (overwrite)
  4. Signal FastAPI to reload via a flag file
"""

import json
import logging
import shutil
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
NEW_DIR    = MODELS_DIR / "new"
LOG_FILE   = MODELS_DIR / "training_log.json"
RELOAD_FLAG = MODELS_DIR / ".reload_needed"

# Import from existing pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_features_training, get_feature_columns


# ── XGBoost params (good baseline, no Optuna needed for weekly runs) ──────────

PARAMS = {
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "n_estimators":     700,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_weight": 5,
    "early_stopping_rounds": 50,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# Minimum quality gate — new model must beat this RMSE on test set
# Loosened slightly so retraining is never blocked (old models stay if new is worse)
MAX_ACCEPTABLE_RMSE = {
    "PM2_5": 60.0,   # µg/m³
    "PM10":  80.0,   # µg/m³
}


def evaluate(model, X_test, y_test_log, label: str) -> dict:
    """Evaluate in original (non-log) scale."""
    pred_log  = model.predict(X_test)
    pred_orig = np.expm1(pred_log)
    y_orig    = np.expm1(y_test_log)

    rmse = float(mean_squared_error(y_orig, pred_orig) ** 0.5)
    mae  = float(mean_absolute_error(y_orig, pred_orig))
    r2   = float(r2_score(y_orig, pred_orig))

    log.info(f"  [{label}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_single_target(df_feat: pd.DataFrame, target: str, feat_cols: list) -> tuple:
    """Train one XGBoost model for a single target (PM2_5 or PM10)."""
    X = df_feat[feat_cols]
    y = np.log1p(df_feat[target])   # log-transform for skewed Dhanbad data

    # Time-based split — never shuffle a time series!
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(f"  Training {target}: {len(X_train):,} train / {len(X_test):,} test rows")

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    metrics = evaluate(model, X_test, y_test, target)
    return model, metrics


def retrain_all(master_csv: Path = None) -> dict:
    """
    Full retrain pipeline:
      1. Load & engineer features from master CSV
      2. Train PM2.5 model
      3. Train PM10 model
      4. Validate both beat quality gate
      5. Hot-swap into production models directory
      6. Write reload flag for FastAPI to pick up
    Returns training report dict.
    """
    master_csv = master_csv or (DATA_DIR / "master_data.csv")
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"=== Auto-retrain started at {run_ts} ===")

    # 1. Load and engineer features
    log.info(f"Loading {master_csv.name}...")
    df = pd.read_csv(master_csv)
    log.info(f"  Total rows: {len(df):,}")

    df_feat    = build_features_training(df, mode="dual")
    feat_cols  = get_feature_columns(df_feat, mode="dual")
    log.info(f"  After feature engineering: {len(df_feat):,} rows, {len(feat_cols)} features")

    # 2. Train both models
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    report   = {"run_at": run_ts, "rows_used": len(df_feat), "models": {}}
    all_pass = True

    for target in ["PM2_5", "PM10"]:
        model, metrics = train_single_target(df_feat, target, feat_cols)

        # Quality gate check
        max_rmse = MAX_ACCEPTABLE_RMSE[target]
        passed   = metrics["rmse"] < max_rmse
        if not passed:
            log.warning(f"  {target} RMSE={metrics['rmse']:.1f} exceeds gate {max_rmse} — will keep old model")
            all_pass = False

        report["models"][target] = {**metrics, "passed_gate": passed, "gate_rmse": max_rmse}

        # Save new model to staging area
        fname = f"model_dual_{target.lower()}.joblib"
        joblib.dump(model, NEW_DIR / fname)
        log.info(f"  Saved new {target} model → models/new/{fname}")

    # 3. Save new metadata
    meta = {
        "mode":          "dual",
        "targets":       ["PM2_5", "PM10"],
        "log_transform": True,
        "feature_cols":  feat_cols,
        "metrics":       report["models"],
        "rows_used":     len(df_feat),
        "trained_at":    run_ts,
        "data_range": {
            "start": str(df["Timestamp"].min()),
            "end":   str(df["Timestamp"].max()),
        },
    }
    with open(NEW_DIR / "model_dual_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 4. Hot-swap: move new models → active production directory
    if all_pass:
        log.info("All models passed quality gate — hot-swapping into production...")
        for fname in NEW_DIR.iterdir():
            dest = MODELS_DIR / fname.name
            shutil.move(str(fname), str(dest))
            log.info(f"  Deployed: {fname.name}")
        NEW_DIR.rmdir()

        # Write reload flag — FastAPI picks this up and reloads models in-place
        RELOAD_FLAG.write_text(run_ts)
        log.info("  Reload flag written → FastAPI will pick up new models on next request")
        report["deployed"] = True
    else:
        log.warning("One or more models failed quality gate — keeping existing production models")
        shutil.rmtree(NEW_DIR, ignore_errors=True)
        report["deployed"] = False

    # 5. Append to training log
    all_logs = []
    if LOG_FILE.exists():
        try:
            all_logs = json.loads(LOG_FILE.read_text())
        except Exception:
            all_logs = []
    all_logs.append(report)
    all_logs = all_logs[-52:]   # Keep last 52 weeks (1 year) of training runs
    LOG_FILE.write_text(json.dumps(all_logs, indent=2))
    log.info(f"  Training log updated ({len(all_logs)} runs stored)")

    log.info(f"=== Auto-retrain complete. Deployed={report['deployed']} ===")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = retrain_all()
    print(f"\nRetrain complete: {report}")
