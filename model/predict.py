import os
import pickle
from pathlib import Path
import duckdb
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")
PRICE_AREA = os.getenv("PRICE_AREA", "SE3").upper()


def predict_tomorrow() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found at {MODEL_PATH}. Run train.py first.")

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    features_list = payload["features"]

    con = duckdb.connect()
    con.sql("INSTALL postgres; LOAD postgres;")
    con.sql(f"ATTACH '{DATABASE_URL}' AS pg (TYPE postgres);")

    query = f"""
        SELECT * FROM pg.feat__daily
        WHERE price_area = '{PRICE_AREA}'
        ORDER BY target_date DESC
        LIMIT 1;
    """
    df = con.sql(query).df()
    if df.empty:
        raise SystemExit("No feature rows found in feat__daily for inference.")

    X = df[features_list]
    prediction = model.predict(X)[0]
    target_date = df["target_date"].values[0]
    target_type = payload["target"]

    print(f"Model: HistGradientBoostingRegressor ({target_type})")
    print(f"Price Area: {PRICE_AREA}")
    print(f"Target Date: {target_date}")
    print(f"Predicted Price: {prediction:.4f} SEK/kWh")


if __name__ == "__main__":
    predict_tomorrow()