import os
import pickle
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.train import enrich_domain_features

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")
PRICE_AREA = os.getenv("PRICE_AREA", "SE3").upper()


def predict_with_intervals(payload: dict, df_row) -> dict:
    df_enriched = enrich_domain_features(df_row)
    features_list = payload["features"]
    X = df_enriched[features_list]

    model = payload["model"]
    mean_pred = float(model.predict(X)[0])

    q_models = payload.get("quantile_models") or {}
    p10 = float(q_models["p10"].predict(X)[0]) if "p10" in q_models else mean_pred * 0.65
    p50 = float(q_models["p50"].predict(X)[0]) if "p50" in q_models else mean_pred
    p90 = float(q_models["p90"].predict(X)[0]) if "p90" in q_models else mean_pred * 1.45

    # Ensure monotonic ordering p10 <= p50 <= p90
    p10, p50, p90 = sorted([p10, p50, p90])

    return {
        "prediction": mean_pred,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "enriched_row": df_enriched.iloc[0].to_dict(),
    }


def predict_tomorrow() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found at {MODEL_PATH}. Run train.py first.")

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

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

    res = predict_with_intervals(payload, df)
    prediction = res["prediction"]
    target_date = df["target_date"].values[0]
    target_type = payload["target"]
    row = res["enriched_row"]

    print(f"Model: HistGradientBoostingRegressor ({target_type})")
    print(f"Price Area: {PRICE_AREA} | Target Date: {target_date}")
    print(f"Predicted Price (Mean):   {prediction:.4f} SEK/kWh ({prediction * 100:.1f} öre/kWh)")
    print(f"Predicted Price (Median): {res['p50']:.4f} SEK/kWh")
    print(f"80% Confidence Band:      [{res['p10']:.4f} .. {res['p90']:.4f}] SEK/kWh (P10–P90)")
    print("-" * 58)
    print(
        f"Domain Signals: Heating Degree Days = {row.get('heating_degree_days', 0):.1f}°C-d | "
        f"Wind Power Proxy = {row.get('wind_power_proxy', 0):.1f} m³/s³"
    )
    ev_cost = prediction * 60.0
    avg_7d_cost = float(row.get("price_7d_mean", prediction)) * 60.0
    diff_pct = ((ev_cost - avg_7d_cost) / max(avg_7d_cost, 0.01)) * 100
    advice = "🟢 CHARGE TOMORROW" if diff_pct <= 0 else "🟡 WAIT IF POSSIBLE"
    print(f"EV 60 kWh Charge: ~{ev_cost:.2f} SEK ({diff_pct:+.1f}% vs 7d avg) -> {advice}")


if __name__ == "__main__":
    predict_tomorrow()