import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

from prep_data import load_data, get_xy, make_pipeline, split
from utils import get_logger

# Define available models
MODELS = {
    "linreg": LinearRegression,
    "rf": RandomForestRegressor,
}


def train(model_name="linreg", data_path="data/cal_housing.csv", models_dir="models"):
    logger = get_logger("train")
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)
    X, y = get_xy(df)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # Build pipeline (preprocessing + model)
    ModelCls = MODELS[model_name]
    estimator = ModelCls()
    preproc = make_pipeline()
    pipeline = Pipeline(steps=[
        ("preproc", preproc),
        ("model", estimator)
    ])

    # Ensure experiment exists or create it
    experiment_name = "housing-regression"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        # Fit model
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)

        # Compute metrics
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae  = float(mean_absolute_error(y_te, preds))
        r2   = float(r2_score(y_te, preds))

        # Log params + metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Use first row as input example
        if isinstance(X_tr, pd.DataFrame):
            input_example = X_tr.iloc[:1]
        else:
            input_example = pd.DataFrame(X_tr[:1])

        # Log full pipeline to MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="housing_model"
        )

        # Also save locally for API fallback
        out_path = Path(models_dir) / "model.pkl"
        import joblib
        joblib.dump(pipeline, out_path)
        logger.info(f"Saved model locally: {out_path} | r2={r2:.4f}, rmse={rmse:.2f}, mae={mae:.2f}")

    print(f"✅ Trained {model_name} | R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="linreg")
    ap.add_argument("--data", default="data/cal_housing.csv")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()
    train(args.model, args.data, args.models_dir)