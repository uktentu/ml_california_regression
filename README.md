# 🏡 California Housing Regression – MLOps Project

Predict **`median_house_value`** using the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).  
This repo demonstrates a full **MLOps pipeline** with data versioning, experiment tracking, API serving, Dockerization, CI/CD, and monitoring.

---

## 📂 Project Structure

```
.
├── api/                  # FastAPI app
│   ├── main.py           # REST endpoints
│   └── schemas.py        # Pydantic models (input validation)
├── src/                  # ML pipeline code
│   ├── prep_data.py      # preprocessing (scaler + one-hot encoder)
│   ├── train.py          # training + MLflow logging
│   ├── evaluate.py       # evaluation
│   └── utils.py          # logging + DB utils
├── data/                 # dataset (DVC tracked)
│   └── cal_housing.csv
├── models/               # local saved models
├── logs/                 # log files (app.log)
├── db/                   # SQLite DB (api_logs, train_runs)
├── .github/workflows/ci.yml  # CI/CD pipeline
├── Dockerfile
├── dvc.yaml              # optional DVC pipeline
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

- **Data & Code Versioning** → Git + [DVC](https://dvc.org/)
- **Experiment Tracking** → [MLflow](https://mlflow.org/) (metrics + model registry)
- **Model Serving** → [FastAPI](https://fastapi.tiangolo.com/) + Pydantic
- **Containerization** → Docker
- **CI/CD** → GitHub Actions
- **Logging & Storage** → Python logging + SQLite
- **Monitoring** → Prometheus + Grafana (optional)

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/ml_california_regression.git
cd ml_california_regression
```

### 2. Create env & install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Initialize DVC

```bash
dvc init
dvc add data/cal_housing.csv
git add data/.gitignore data/cal_housing.csv.dvc dvc.yaml
git commit -m "Track dataset with DVC"
```

---

## 📊 Training & Experiment Tracking

### Train model (Linear Regression / Random Forest)

```bash
python src/train.py --model rf
```

### Evaluate model

```bash
python src/evaluate.py
```

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Visit → [http://127.0.0.1:5000](http://127.0.0.1:5000)  
You’ll see:

- Params (`model=rf`)
- Metrics (`rmse`, `mae`, `r2`)
- Artifacts (`model.pkl`, signature, conda env)
- Model Registry (`housing_model` with version history)

---

## 🌐 API Service

### Run FastAPI locally

```bash
uvicorn api.main:app --reload
```

### Endpoints

- `GET /health` → health check
- `POST /predict` → predict house value
- `GET /logs` → last 50 log lines (plain text)
- `GET /logs/db` → logs from SQLite (JSON)
- `GET /logs/db/html` → logs as an HTML table
- `GET /metrics` → Prometheus metrics

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "<1H OCEAN"
}'
```

Response:

```json
{
  "prediction": 256789.23,
  "latency_ms": 12.45
}
```

---

## 🐳 Docker

### Build

```bash
docker build -t housing-api:latest .
```

### Run

```bash
docker run -p 8000:8000 housing-api:latest
```

Visit → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🚀 CI/CD (GitHub Actions)

The pipeline (`.github/workflows/ci.yml`) runs on every push/PR:

1. Install deps
2. Lint Python files
3. Train model (`src/train.py`)
4. Evaluate model (`src/evaluate.py`)
5. Build Docker image
6. (Optional) Push to GHCR or Docker Hub

### Enable GHCR push

1. Add secret `GHCR_PAT` (or use built-in `GITHUB_TOKEN` with `packages: write`)
2. Image will be available at:
   ```
   ghcr.io/<owner>/<repo>:latest
   ```

---

## 📈 Monitoring (Optional)

- **Prometheus** scrapes metrics from `/metrics`.
- **Grafana** dashboards → request count, prediction latency, error rates, distribution.

---

## 📑 Logs

- File logs: `logs/app.log`
- Structured logs in SQLite: `db/app.db` (`api_logs`, `train_runs`)

---

## ✅ Next Steps

- Connect Prometheus + Grafana for full monitoring
- Add pytest test suite (`tests/`)
- Extend CI to auto-promote best model in MLflow registry
- Deploy API with Kubernetes or Docker Compose

---

## ✨ Summary

This repo shows how to take a **simple regression task** and wrap it in **full MLOps best practices**:

- Reproducible data & code (Git + DVC)
- Experiment tracking + registry (MLflow)
- Scalable API (FastAPI + Docker)
- Automated pipeline (GitHub Actions)
- Observability (logs, SQLite, Prometheus/Grafana)
