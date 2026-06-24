# AQI Dhanbad Backend Deployment

This deployment is backend-only: model artifacts, FastAPI API, `src` pipeline scripts, and Python requirements. The frontend is not required.

## Required Environment Variables

Set these in your hosting provider or in a local `.env` file:

```bash
OWM_API_KEY=your_openweathermap_key
CORS_ORIGINS=*
```

`OWM_API_KEY` is required for `/aqi/now`, `/aqi/forecast`, `/aqi/daily`, `/debug/owm`, data collection, and scheduled retraining. `/health` can run without calling OpenWeatherMap.

## Files That Must Be Deployed

```text
api/main.py
src/auto_retrain.py
src/data_collector.py
src/feature_engineering.py
src/owm_client.py
src/scheduler.py
src/train.py
start.py
models/model_dual_pm2_5.joblib
models/model_dual_pm10.joblib
models/model_dual_meta.json
models/training_log.json
data/master_data.csv
data/final_clean_merged_dataset.csv
requirements.txt
render.yaml
Dockerfile
Procfile
runtime.txt
.env.example
```

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python start.py
```

For API-only local testing in PowerShell, run `$env:RUN_SCHEDULER='false'` before `python start.py`.

Health check:

```bash
curl http://localhost:8000/health
```

## Render Deployment

Use `render.yaml` as the blueprint, or create services manually.

Web service:

```bash
pip install -r requirements.txt
python start.py
```

`start.py` starts the FastAPI API and, by default, starts `src/scheduler.py` beside it so retrained models are written to the same filesystem the API reads from. To run the API without the scheduler:

```bash
RUN_SCHEDULER=false python start.py
```

The scheduler runs every Sunday at 02:00 IST. On free hosting plans that sleep, scheduled retraining only runs while the service is awake. For reliable weekly retraining, use an always-on instance or run `python src/scheduler.py --now` from the same deployed filesystem.

## Docker Deployment

Build and run:

```bash
docker build -t aqi-dhanbad-api .
docker run -p 8000:8000 --env-file .env aqi-dhanbad-api
```

## Notes

- Do not commit `.env` or real API keys.
- The API loads active models from `models/`.
- The weekly pipeline appends OWM data to `data/master_data.csv`, retrains dual PM models, and writes `models/.reload_needed` so the FastAPI process hot-reloads models on the next request.
