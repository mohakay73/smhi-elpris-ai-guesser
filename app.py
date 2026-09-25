from __future__ import annotations

import math
import os
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model.predict import predict_with_intervals
from model.train import enrich_domain_features

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
INDEX_HTML_PATH = BASE_DIR / "index.html"
DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_AREA = os.getenv("PRICE_AREA", "SE2").upper()

app = FastAPI(
    title="SMHI Elpris AI Forecaster & What-If Simulator",
    description=(
        "FastAPI inference service with Quantile Uncertainty Bands (P10/P50/P90), "
        "Domain Feature Engineering, and Interactive Weather Simulator."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    tomorrow_temp_mean: float = Field(default=8.5, description="Tomorrow mean temp (°C)")
    tomorrow_temp_spread: float = Field(
        default=5.0, ge=0.0, description="Tomorrow max - min temp (°C)"
    )
    tomorrow_wind_mean: float = Field(
        default=3.5, ge=0.0, description="Tomorrow mean wind speed (m/s)"
    )
    price_today: float = Field(
        default=0.35, description="Today's mean electricity price (SEK/kWh)"
    )
    price_yesterday: float = Field(default=0.38, description="Yesterday's mean price (SEK/kWh)")
    price_7d_ago: float = Field(default=0.42, description="Price 7 days ago (SEK/kWh)")
    price_7d_mean: float = Field(default=0.45, description="7-day rolling mean price (SEK/kWh)")
    peak_today: float = Field(default=0.68, description="Today's peak hourly price (SEK/kWh)")
    dayofweek: int = Field(default=2, ge=0, le=6, description="0=Mon .. 6=Sun")
    month: int = Field(default=9, ge=1, le=12, description="Month 1..12")
    is_weekend: int = Field(default=0, ge=0, le=1, description="1 if Saturday/Sunday else 0")


def _clean_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else round(f, 5)
    except Exception:
        return default


def load_model_payload() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run `uv run python model/train.py` first.",
        )
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


def build_consumer_advisor(
    pred_mean: float, p10: float, p90: float, row: dict[str, Any]
) -> dict[str, Any]:
    price_7d_mean = _clean_float(row.get("price_7d_mean"), pred_mean)
    price_today = _clean_float(row.get("price_today"), pred_mean)
    heating_dd = _clean_float(row.get("heating_degree_days"), 0.0)
    wind_proxy = _clean_float(row.get("wind_power_proxy"), 0.0)

    ev_60kwh_sek = round(pred_mean * 60.0, 2)
    ev_7d_avg_sek = round(price_7d_mean * 60.0, 2)
    ev_diff_pct = round(
        ((pred_mean - price_7d_mean) / max(abs(price_7d_mean), 0.05)) * 100.0, 1
    )

    sauna_8kwh_sek = round(pred_mean * 8.0, 2)
    laundry_2kwh_sek = round(pred_mean * 2.5, 2)

    band_width = max(0.0, p90 - p10)
    if p90 > max(1.0, price_7d_mean * 1.6):
        spike_risk = "HIGH"
    elif p90 > price_7d_mean * 1.25:
        spike_risk = "MODERATE"
    else:
        spike_risk = "LOW"

    if pred_mean <= price_today * 0.92 and pred_mean <= price_7d_mean:
        verdict_badge = "CHARGE & RUN HEAVY LOADS TOMORROW"
        verdict_color = "emerald"
        verdict_reason = (
            f"Tomorrow is forecasted {abs(ev_diff_pct):.1f}% below the 7-day average "
            "and cheaper than today."
        )
    elif pred_mean >= price_today * 1.12:
        verdict_badge = "USE POWER TODAY INSTEAD OF TOMORROW"
        verdict_color = "amber"
        pct_rise = ((pred_mean - price_today) / max(price_today, 0.05)) * 100
        verdict_reason = f"Prices are expected to rise tomorrow (+{pct_rise:.1f}% vs today)."
    else:
        verdict_badge = "NORMAL GRID CONDITIONS"
        verdict_color = "sky"
        verdict_reason = (
            "Tomorrow's price is stable near recent levels; "
            "schedule loads during off-peak night hours."
        )

    return {
        "ev_60kwh_sek": ev_60kwh_sek,
        "ev_7d_avg_sek": ev_7d_avg_sek,
        "ev_diff_pct": ev_diff_pct,
        "sauna_8kwh_sek": sauna_8kwh_sek,
        "laundry_2kwh_sek": laundry_2kwh_sek,
        "band_width": round(band_width, 4),
        "spike_risk": spike_risk,
        "heating_degree_days": round(heating_dd, 1),
        "wind_power_proxy": round(wind_proxy, 1),
        "verdict_badge": verdict_badge,
        "verdict_color": verdict_color,
        "verdict_reason": verdict_reason,
    }


def fetch_history_from_db(area: str, limit: int = 35) -> tuple[pd.DataFrame, str]:
    if DATABASE_URL:
        try:
            with psycopg2.connect(DATABASE_URL, connect_timeout=6) as conn:
                query = """
                    SELECT *
                    FROM feat__daily
                    WHERE price_area = %s
                    ORDER BY target_date DESC
                    LIMIT %s
                """
                df = pd.read_sql_query(query, conn, params=(area, limit))
                if not df.empty:
                    df = df.sort_values("target_date").reset_index(drop=True)
                    return df, "neon_postgres"
        except Exception as exc:
            print(f"[app] Database fallback triggered: {exc}")

    # Fallback sample dataframe if DB is unreachable
    fallback_rows = [
        {
            "target_date": date(2026, 9, 25),
            "tomorrow_temp_mean": 8.775,
            "tomorrow_temp_min": 7.1,
            "tomorrow_temp_max": 12.2,
            "tomorrow_wind_mean": 2.17,
            "price_today": 0.0985,
            "price_yesterday": 0.0434,
            "price_7d_ago": 0.7495,
            "price_7d_mean": 0.6042,
            "peak_today": 0.4688,
            "dayofweek": 6,
            "month": 9,
            "is_weekend": 1,
            "y": None,
        }
    ]
    return pd.DataFrame(fallback_rows), "local_fallback"


@app.get("/api/overview")
def get_overview() -> dict[str, Any]:
    payload = load_model_payload()
    area = payload.get("price_area", DEFAULT_AREA)
    df, source = fetch_history_from_db(area, limit=35)
    df_enriched = enrich_domain_features(df)

    features_list = payload["features"]
    for col in features_list:
        if col not in df_enriched.columns:
            df_enriched[col] = 0.0
    df_enriched[features_list] = df_enriched[features_list].ffill().bfill().fillna(0.0)

    # Latest row is tomorrow's prediction candidate
    latest_df = df_enriched.iloc[[-1]].copy()
    latest_pred = predict_with_intervals(payload, latest_df)
    latest_row = latest_pred["enriched_row"]
    target_date_str = str(latest_row.get("target_date", date.today().isoformat()))

    # Evaluate historical series
    X_all = df_enriched[features_list]
    preds_mean = payload["model"].predict(X_all)
    q_models = payload.get("quantile_models") or {}
    preds_p10 = q_models["p10"].predict(X_all) if "p10" in q_models else preds_mean * 0.65
    preds_p50 = q_models["p50"].predict(X_all) if "p50" in q_models else preds_mean
    preds_p90 = q_models["p90"].predict(X_all) if "p90" in q_models else preds_mean * 1.45

    history_records = []
    ai_wins = 0
    eval_days = 0

    for idx, row_s in df_enriched.iterrows():
        p_m = _clean_float(preds_mean[idx])
        p_10, p_50, p_90 = sorted([
            _clean_float(preds_p10[idx]),
            _clean_float(preds_p50[idx]),
            _clean_float(preds_p90[idx]),
        ])
        actual_y = row_s.get("y")
        actual_val = None if pd.isna(actual_y) else _clean_float(actual_y)
        naive_val = _clean_float(
            row_s.get("price_today") if payload.get("target") == "mean" else row_s.get("peak_today")
        )

        if actual_val is not None:
            eval_days += 1
            if abs(actual_val - p_m) <= abs(actual_val - naive_val):
                ai_wins += 1

        history_records.append(
            {
                "date": str(row_s.get("target_date", idx)),
                "actual": actual_val,
                "predicted": p_m,
                "p10": p_10,
                "p50": p_50,
                "p90": p_90,
                "naive": naive_val,
                "wind": _clean_float(row_s.get("tomorrow_wind_mean")),
                "temp": _clean_float(row_s.get("tomorrow_temp_mean")),
            }
        )

    win_rate = round((ai_wins / eval_days) * 100.0, 1) if eval_days > 0 else None

    advisor = build_consumer_advisor(
        latest_pred["prediction"],
        latest_pred["p10"],
        latest_pred["p90"],
        latest_row,
    )

    metrics = payload.get("metrics", {})
    return {
        "data_source": source,
        "price_area": area,
        "target": payload.get("target", "mean"),
        "trained_at": payload.get("trained_at", "unknown"),
        "n_train": payload.get("n_train"),
        "features": payload.get("features", []),
        "domain_features": payload.get("domain_features", []),
        "metrics": {
            "mae": _clean_float(metrics.get("mae")),
            "p50_mae": _clean_float(metrics.get("p50_mae", metrics.get("mae"))),
            "naive_mae": _clean_float(metrics.get("naive_mae")),
            "r2": _clean_float(metrics.get("r2")),
            "picp_80": _clean_float(metrics.get("picp_80", 0.80)),
            "live_win_rate_pct": win_rate,
            "eval_days": eval_days,
        },
        "latest": {
            "target_date": target_date_str,
            "prediction": _clean_float(latest_pred["prediction"]),
            "p10": _clean_float(latest_pred["p10"]),
            "p50": _clean_float(latest_pred["p50"]),
            "p90": _clean_float(latest_pred["p90"]),
            "naive": _clean_float(latest_row.get("price_today")),
            "features": {
                "tomorrow_temp_mean": _clean_float(latest_row.get("tomorrow_temp_mean"), 8.5),
                "tomorrow_temp_min": _clean_float(latest_row.get("tomorrow_temp_min"), 6.0),
                "tomorrow_temp_max": _clean_float(latest_row.get("tomorrow_temp_max"), 11.0),
                "tomorrow_temp_spread": _clean_float(latest_row.get("tomorrow_temp_spread"), 5.0),
                "tomorrow_wind_mean": _clean_float(latest_row.get("tomorrow_wind_mean"), 3.5),
                "price_today": _clean_float(latest_row.get("price_today"), 0.35),
                "price_yesterday": _clean_float(latest_row.get("price_yesterday"), 0.35),
                "price_7d_ago": _clean_float(latest_row.get("price_7d_ago"), 0.40),
                "price_7d_mean": _clean_float(latest_row.get("price_7d_mean"), 0.42),
                "peak_today": _clean_float(latest_row.get("peak_today"), 0.65),
                "dayofweek": int(_clean_float(latest_row.get("dayofweek"), 2)),
                "month": int(_clean_float(latest_row.get("month"), 9)),
                "is_weekend": int(_clean_float(latest_row.get("is_weekend"), 0)),
                "heating_degree_days": _clean_float(latest_row.get("heating_degree_days"), 8.5),
                "wind_power_proxy": _clean_float(latest_row.get("wind_power_proxy"), 42.9),
            },
            "advisor": advisor,
        },
        "history": history_records,
    }


@app.post("/api/simulate")
def simulate_weather(req: SimulationRequest) -> dict[str, Any]:
    payload = load_model_payload()

    half_spread = req.tomorrow_temp_spread / 2.0
    base_row = {
        "tomorrow_temp_mean": req.tomorrow_temp_mean,
        "tomorrow_temp_min": req.tomorrow_temp_mean - half_spread,
        "tomorrow_temp_max": req.tomorrow_temp_mean + half_spread,
        "tomorrow_wind_mean": req.tomorrow_wind_mean,
        "price_today": req.price_today,
        "price_yesterday": req.price_yesterday,
        "price_7d_ago": req.price_7d_ago,
        "price_7d_mean": req.price_7d_mean,
        "peak_today": req.peak_today,
        "dayofweek": 6 if req.is_weekend else req.dayofweek,
        "month": req.month,
        "is_weekend": req.is_weekend,
    }

    point_res = predict_with_intervals(payload, pd.DataFrame([base_row]))
    advisor = build_consumer_advisor(
        point_res["prediction"],
        point_res["p10"],
        point_res["p90"],
        point_res["enriched_row"],
    )

    # 1. Wind sensitivity curve (0 m/s .. 18 m/s)
    wind_grid = [round(w * 1.0, 1) for w in range(0, 19)]
    wind_rows = []
    for w in wind_grid:
        r = dict(base_row)
        r["tomorrow_wind_mean"] = w
        wind_rows.append(r)
    wind_df = enrich_domain_features(pd.DataFrame(wind_rows))[payload["features"]]
    w_mean = payload["model"].predict(wind_df)
    q_models = payload.get("quantile_models") or {}
    w_p10 = q_models["p10"].predict(wind_df) if "p10" in q_models else w_mean * 0.65
    w_p90 = q_models["p90"].predict(wind_df) if "p90" in q_models else w_mean * 1.45

    wind_curve = [
        {
            "wind_ms": w,
            "mean": _clean_float(w_mean[i]),
            "p10": _clean_float(min(w_p10[i], w_mean[i])),
            "p90": _clean_float(max(w_p90[i], w_mean[i])),
        }
        for i, w in enumerate(wind_grid)
    ]

    # 2. Temperature sensitivity curve (-20°C .. +24°C)
    temp_grid = [round(-20.0 + i * 2.5, 1) for i in range(19)]
    temp_rows = []
    for t in temp_grid:
        r = dict(base_row)
        r["tomorrow_temp_mean"] = t
        r["tomorrow_temp_min"] = t - half_spread
        r["tomorrow_temp_max"] = t + half_spread
        temp_rows.append(r)
    temp_df = enrich_domain_features(pd.DataFrame(temp_rows))[payload["features"]]
    t_mean = payload["model"].predict(temp_df)
    t_p10 = q_models["p10"].predict(temp_df) if "p10" in q_models else t_mean * 0.65
    t_p90 = q_models["p90"].predict(temp_df) if "p90" in q_models else t_mean * 1.45

    temp_curve = [
        {
            "temp_c": t,
            "mean": _clean_float(t_mean[i]),
            "p10": _clean_float(min(t_p10[i], t_mean[i])),
            "p90": _clean_float(max(t_p90[i], t_mean[i])),
        }
        for i, t in enumerate(temp_grid)
    ]

    erow = point_res["enriched_row"]
    return {
        "prediction": _clean_float(point_res["prediction"]),
        "p10": _clean_float(point_res["p10"]),
        "p50": _clean_float(point_res["p50"]),
        "p90": _clean_float(point_res["p90"]),
        "domain_features": {
            "heating_degree_days": _clean_float(erow.get("heating_degree_days")),
            "wind_power_proxy": _clean_float(erow.get("wind_power_proxy")),
            "tomorrow_temp_spread": _clean_float(erow.get("tomorrow_temp_spread")),
            "price_momentum": _clean_float(erow.get("price_momentum")),
            "peak_to_mean_ratio": _clean_float(erow.get("peak_to_mean_ratio")),
        },
        "advisor": advisor,
        "wind_curve": wind_curve,
        "temp_curve": temp_curve,
    }


@app.get("/predict")
def predict_endpoint() -> dict[str, Any]:
    overview = get_overview()
    latest = overview["latest"]
    return {
        "price_area": overview["price_area"],
        "target_date": latest["target_date"],
        "predicted_price_sek_per_kwh": latest["prediction"],
        "p10_sek_per_kwh": latest["p10"],
        "p50_sek_per_kwh": latest["p50"],
        "p90_sek_per_kwh": latest["p90"],
        "advisor": latest["advisor"],
    }


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)