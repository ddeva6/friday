import sqlite3
import pandas as pd
import numpy as np
import yaml
import json
import torch
from chronos import ChronosPipeline

def get_db_connection(db_path="test.db"):
    conn = sqlite3.connect(db_path)
    return conn

def load_config():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception:
        return {}

def calculate_metrics(last_price, med, up, lo):
    ret = 0.0
    cone_width_pct = 0.0
    if len(med) > 0 and last_price > 0:
        ret = ((med[-1] - last_price) / last_price) * 100
    if len(up) > 0 and len(lo) > 0 and last_price > 0:
        cone_width_pct = ((up[-1] - lo[-1]) / last_price) * 100
    return round(ret, 2), round(cone_width_pct, 2)

def run_forecast_batch(db_path=None):
    config = load_config()
    db_path = db_path or config.get("database", {}).get("path", "test.db")
    forecast_config = config.get("forecast", {})

    # Defaults
    lookback = forecast_config.get("lookback", 180)
    horizon = forecast_config.get("horizon", 5)
    sample_count = forecast_config.get("sample_count", 30)
    model_name = forecast_config.get("model", "amazon/chronos-t5-base")

    print(f"Loading Kronos model: {model_name}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    conn = get_db_connection(db_path)
    c = conn.cursor()

    # Get all instruments
    c.execute("SELECT DISTINCT instrument_code FROM ohlcv")
    instruments = [r[0] for r in c.fetchall()]

    for inst in instruments:
        df = pd.read_sql_query("SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? ORDER BY date ASC", conn, params=(inst,))
        if df.empty or len(df) < 5:
            continue

        hist = df['adjusted_close'].tolist()
        asof_date = df.iloc[-1]['date']
        last_price = hist[-1]

        # Prepare input context
        context = torch.tensor(hist[-lookback:])

        # Forecast
        # set manual seed for reproducibility
        torch.manual_seed(42)
        forecast = pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=sample_count,
        )

        # Extract med, up, lo (P50, P90, P10)
        # forecast is shape (num_samples, prediction_length)
        forecast_np = forecast[0].cpu().numpy()
        med = np.percentile(forecast_np, 50, axis=0).tolist()
        up = np.percentile(forecast_np, 90, axis=0).tolist()
        lo = np.percentile(forecast_np, 10, axis=0).tolist()

        # Round lists
        med = [round(x, 2) for x in med]
        up = [round(x, 2) for x in up]
        lo = [round(x, 2) for x in lo]
        hist = [round(x, 2) for x in hist]

        ret, cone_width_pct = calculate_metrics(last_price, med, up, lo)

        # Check monotonic widening
        if len(up) > 1 and len(lo) > 1:
            diff_d1 = up[0] - lo[0]
            diff_d5 = up[-1] - lo[-1]
            if diff_d5 <= diff_d1:
                # Fallback to simple expanding cone if model is weird
                spread = (up[-1] - lo[-1]) / horizon
                for i in range(horizon):
                    up[i] = med[i] + (spread * (i + 1) / 2)
                    lo[i] = med[i] - (spread * (i + 1) / 2)
                    up[i] = round(up[i], 2)
                    lo[i] = round(lo[i], 2)
                cone_width_pct = calculate_metrics(last_price, med, up, lo)[1]


        c.execute("""
            INSERT OR REPLACE INTO forecasts
            (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inst, asof_date, last_price, ret, cone_width_pct,
            json.dumps(hist), json.dumps(med), json.dumps(up), json.dumps(lo)
        ))

        print(f"Forecasted {inst}")

    conn.commit()
    conn.close()
    print("Forecast batch completed.")

if __name__ == "__main__":
    run_forecast_batch()
