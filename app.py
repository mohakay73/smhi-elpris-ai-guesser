import os
import pickle
from pathlib import Path
import duckdb
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import traceback
from datetime import datetime, timedelta

import pandas as pd

load_dotenv()

app = FastAPI(
    title="Sweden Electricity Price Predictor API",
    description="Predicts tomorrow's electricity prices using SMHI weather data and historical trends.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (or you can specify your github pages domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("model/model.pkl")
DATABASE_URL = os.getenv("DATABASE_URL")
PRICE_AREA = os.getenv("PRICE_AREA", "SE3").upper()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "price_area": PRICE_AREA,
        "message": "Welcome to the Electricity Price Predictor API!"
    }


@app.get("/predict")
def get_prediction():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model file not found. Run training first.")
    
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not set in environment.")

    try:
        # Load serialized model payload
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        model = payload["model"]
        features_list = payload["features"]
        target_type = payload["target"]

        # Connect via DuckDB to query Neon database features
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
            raise HTTPException(status_code=404, detail="No feature rows found in feat__daily for inference.")

        # Extract features and predict
        X = df[features_list]
        prediction = float(model.predict(X)[0])
        target_date = str(df["target_date"].values[0])

        base_date = pd.to_datetime(df["target_date"].values[0])
        forecast_date = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")

        return {
            "price_area": PRICE_AREA,
            "target_date": forecast_date,
            "target_type": target_type,
            "predicted_price_sek_per_kwh": round(prediction, 4),
            "metrics": payload.get("metrics", {})
        }

    except Exception as e:
        print("ERROR DETAILS:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))