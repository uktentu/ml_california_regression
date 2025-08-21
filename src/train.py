import argparse, json, joblib, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow, mlflow.sklearn

from prep_data import load_data, get_xy, make_pipeline, split
from utils import get_logger, get_db

MODELS = {
    "linreg": (LinearRegression, {"n_jobs": None}),
    "rf": (RandomForestRegressor, {"n_estimators": 300, "max_depth": None, "random_state": 42, "n_jobs": -1}),
}

def train(model_name="linreg", data_path="data/cal_housing.csv", models_dir="models"):
    logger = get_logger("train")
    conn = get_db(); cur = conn.cursor()
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y = get_xy(df)
    X_tr, X_te, y_tr, y_te = split(X, y)

    pipe = make_pipeline()
    X_tr = pipe.fit_transform(X_tr)
    X_te = pipe.transform(X_te)

    cls, default_params = MODELS[model_name]
    model = cls(**{k:v for k,v in default_params.items() if k in cls().__dict__ or True})  # tolerant

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(default_params)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae  = float(mean_absolute_error(y_te, preds))
        r2   = float(r2_score(y_te, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # persist artifacts locally AND in MLflow
        bundle = {"preproc": pipe, "estimator": model}
        out_path = Path(models_dir) / "model.pkl"
        joblib.dump(bundle, out_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info(f"Saved model to {out_path} | r2={r2:.4f}, rmse={rmse:.2f}, mae={mae:.2f}")

        cur.execute(
            "INSERT INTO train_runs(model, r2, rmse, mae, params) VALUES(?,?,?,?,?)",
            (model_name, r2, rmse, mae, json.dumps(default_params))
        )
        conn.commit()

    print(f"✅ Trained {model_name} | R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="linreg")
    ap.add_argument("--data", default="data/cal_housing.csv")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()
    train(args.model, args.data, args.models_dir)
