"""
src/scheduler.py
Weekly pipeline orchestrator — the heart of the self-updating system.

Runs as a background process alongside your FastAPI server.
Every Sunday at 2:00 AM (India time) it:
  1. Collects the past 7 days of OWM data
  2. Appends to master_data.csv
  3. Retrains both XGBoost models
  4. Hot-swaps new models into FastAPI (zero downtime)
  5. Logs the full run to logs/pipeline.log

Start it in a separate terminal:
  python scheduler.py

Or run it immediately for testing:
  python scheduler.py --now

On Windows (for production): use Task Scheduler pointing to this script.
On Linux/Mac: use cron:  0 2 * * 0  python /path/to/scheduler.py --now
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

# Setup logging to both console and file
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from data_collector import collect_weekly_data
from auto_retrain   import retrain_all


def run_pipeline():
    """
    Full weekly pipeline: collect → retrain → hot-swap.
    This is the function APScheduler calls every Sunday at 2 AM.
    """
    run_start = datetime.now()
    log.info("=" * 60)
    log.info(f"WEEKLY PIPELINE STARTED at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # ── Step 1: Collect last 7 days of data ──────────────────────
    log.info("STEP 1/2 — Collecting 7 days of OWM historical data...")
    try:
        new_rows, csv_path = collect_weekly_data(days_back=7)
        if new_rows == 0:
            log.warning("No new data collected this week — skipping retrain")
            log.info("Pipeline finished (no new data).")
            return
        log.info(f"  ✓ Collected {new_rows} new rows → {csv_path.name}")
    except Exception as e:
        log.error(f"  ✗ Data collection failed: {e}", exc_info=True)
        log.error("Pipeline aborted — will retry next week.")
        return

    # ── Step 2: Retrain and deploy ────────────────────────────────
    log.info("STEP 2/2 — Retraining XGBoost models on updated data...")
    try:
        report = retrain_all(master_csv=csv_path)
        if report["deployed"]:
            log.info("  ✓ New models deployed and hot-swapped into FastAPI")
            for target, m in report["models"].items():
                log.info(f"    {target}: RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}")
        else:
            log.warning("  ⚠ New models did not pass quality gate — keeping existing models")
    except Exception as e:
        log.error(f"  ✗ Retraining failed: {e}", exc_info=True)
        log.error("  Existing models remain in production — no disruption to API")
        return

    # ── Done ──────────────────────────────────────────────────────
    elapsed = (datetime.now() - run_start).total_seconds()
    log.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AQI self-updating pipeline scheduler")
    parser.add_argument("--now", action="store_true",
                        help="Run the pipeline immediately (for testing)")
    parser.add_argument("--day",  default="sun",
                        help="Day of week to run (mon/tue/.../sun). Default: sun")
    parser.add_argument("--hour", type=int, default=2,
                        help="Hour to run (24h, IST). Default: 2")
    args = parser.parse_args()

    if args.now:
        log.info("Running pipeline immediately (--now flag)...")
        run_pipeline()
        return

    if not APSCHEDULER_AVAILABLE:
        log.error("APScheduler not installed. Run:  pip install apscheduler")
        log.error("Or use --now to run immediately, or set up cron/Task Scheduler manually.")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(
            day_of_week=args.day,
            hour=args.hour,
            minute=0,
            timezone="Asia/Kolkata",
        ),
        id="weekly_aqi_pipeline",
        name="Weekly AQI data collection + model retrain",
        misfire_grace_time=3600,     # If server was off, run within 1hr of scheduled time
        coalesce=True,               # Never run twice if missed multiple times
    )

    log.info(f"Scheduler started — pipeline will run every {args.day.capitalize()} at {args.hour:02d}:00 IST")
    log.info("Keep this terminal open. Press Ctrl+C to stop.")
    log.info(f"Logs: {LOG_DIR / 'pipeline.log'}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
