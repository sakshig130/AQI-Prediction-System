import os
import signal
import subprocess
import sys

import uvicorn


scheduler_process = None


def start_scheduler():
    if os.environ.get("RUN_SCHEDULER", "true").lower() not in {"1", "true", "yes"}:
        return None
    return subprocess.Popen([sys.executable, "src/scheduler.py"])


def shutdown(signum, frame):
    if scheduler_process and scheduler_process.poll() is None:
        scheduler_process.terminate()
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    scheduler_process = start_scheduler()
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
