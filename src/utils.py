import logging, os, sqlite3
from pathlib import Path

LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR = Path("db"); DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "app.log"
DB_PATH = DB_DIR / "app.db"

def get_logger(name="mlapp"):
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            longitude REAL, latitude REAL, housing_median_age REAL,
            total_rooms REAL, total_bedrooms REAL, population REAL,
            households REAL, median_income REAL,
            prediction REAL, latency_ms REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS train_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            model TEXT, r2 REAL, rmse REAL, mae REAL, params TEXT
        )
    """)
    conn.commit()
    return conn
