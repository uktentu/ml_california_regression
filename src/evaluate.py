import joblib, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prep_data import load_data, get_xy, split
from utils import get_logger

def evaluate(model_path="models/model.pkl", data_path="data/cal_housing.csv"):
    logger = get_logger("evaluate")
    df = load_data(data_path)
    X, y = get_xy(df)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # Load the full pipeline directly (not a dict with 'preproc' and 'estimator')
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae  = float(mean_absolute_error(y_te, preds))
    r2   = float(r2_score(y_te, preds))
    logger.info(f"Evaluation | R2={r2:.4f} RMSE={rmse:.2f} MAE={mae:.2f}")
    print(f"📊 Evaluation | R2={r2:.4f} RMSE={rmse:.2f} MAE={mae:.2f}")

if __name__ == "__main__":
    evaluate()