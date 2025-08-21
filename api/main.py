import time, joblib, pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from api.schemas import HouseFeatures
from src.utils import get_logger, get_db

app = FastAPI(title="House Price API", version="1.0")
logger = get_logger("api")
conn = get_db(); cur = conn.cursor()

bundle = joblib.load("models/model.pkl")
pipe = bundle["preproc"]; model = bundle["estimator"]

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: HouseFeatures):
    start = time.time()
    df = pd.DataFrame([payload.dict()])
    X = pipe.transform(df)
    yhat = float(model.predict(X)[0])
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
