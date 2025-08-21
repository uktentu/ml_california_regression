import time, joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from api.schemas import HouseFeatures
from src.utils import get_logger, get_db
import os

app = FastAPI(title="House Price API", version="1.0")
logger = get_logger("api")
conn = get_db(); cur = conn.cursor()

# Load the full pipeline, not a dict
pipeline = joblib.load("models/model.pkl")

# Serve static HTML for the test form
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HTML_PATH = os.path.join(STATIC_DIR, "home.html")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_form():
    with open(HTML_PATH, "r") as f:
        return f.read()

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: HouseFeatures):
    start = time.time()
    df = pd.DataFrame([payload.dict()])
    yhat = float(pipeline.predict(df)[0])
    latency_ms = (time.time() - start) * 1000.0

    cur.execute("""
        INSERT INTO api_logs(longitude, latitude, housing_median_age, total_rooms,
                             total_bedrooms, population, households, median_income,
                             prediction, latency_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        payload.longitude, payload.latitude, payload.housing_median_age,
        payload.total_rooms, payload.total_bedrooms, payload.population,
        payload.households, payload.median_income, yhat, latency_ms
    ))
    conn.commit()

    logger.info(f"pred={yhat:.2f} | latency_ms={latency_ms:.1f}")
    return {"prediction": yhat, "latency_ms": latency_ms}