# California Housing – MLOps Regression

Predict `median_house_value` with full pipeline: DVC (data), MLflow (experiments), FastAPI (serving), Docker, GitHub Actions, logs→SQLite, Prometheus metrics. Based on the capstone outline.

## Quickstart (local)

1. Python env + deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

dvc init
dvc add data/cal_housing.csv
git add data/.gitignore data/cal_housing.csv.dvc dvc.yaml
git commit -m "track dataset with DVC"
# set a remote: dvc remote add -d origin <your-remote> && dvc push

mlflow ui --backend-store-uri sqlite:///mlflow.db

python src/train.py --model rf
python src/evaluate.py

uvicorn api.main:app --reload
# Test
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
 -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41,"total_rooms":880,"total_bedrooms":129,"population":322,"households":126,"median_income":8.3252}'

docker build -t housing-api:latest .
docker run -p 8000:8000 housing-api:latest
```
