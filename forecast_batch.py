import sqlite3
import pandas as pd
import numpy as np
import yaml
import json
import logging


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

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

def ensure_monotonic_widening(med, up, lo, horizon):
    if len(up) > 1 and len(lo) > 1:
        diff_d1 = up[0] - lo[0]
        diff_d5 = up[-1] - lo[-1]
        if diff_d5 <= diff_d1:
            spread = (up[-1] - lo[-1]) / horizon
            for i in range(horizon):
                up[i] = round(med[i] + (spread * (i + 1) / 2), 2)
                lo[i] = round(med[i] - (spread * (i + 1) / 2), 2)
    return up, lo

def run_forecast_batch(db_path=None):
    import torch
    from chronos import ChronosPipeline

    config = load_config()
    db_path = db_path or config.get("database", {}).get("path", "test.db")
    forecast_config = config.get("forecast", {})

    lookback = forecast_config.get("lookback", 180)
    horizon = forecast_config.get("horizon", 5)
    sample_count = forecast_config.get("sample_count", 30)
    model_name = forecast_config.get("model", "amazon/chronos-t5-base")
    seed = forecast_config.get("seed", 42)

    logging.info("FRIDAY forecast batch starting")
    logging.info(f"Model: {model_name}")
    logging.info(f"Config: lookback={lookback}, horizon={horizon}, sample_count={sample_count}")
    logging.info(f"Seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logging.info(f"Device: {device}")

    # Chronos-t5-base: ~200M params, ~0.8 GB VRAM, ~1-2 min for 50 instruments on T4
    # Chronos-t5-tiny: ~8M params, runs on CPU, ~5-10 min for 50 instruments
    logging.info(f"Loading model: {model_name}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
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
        torch.manual_seed(seed)
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

        up, lo = ensure_monotonic_widening(med, up, lo, horizon)
        if up != lo:
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

    c.execute("SELECT MAX(asof_date) FROM forecasts")
    max_asof = c.fetchone()[0]

    conn.close()
    logging.info(f"Forecast batch complete: {len(instruments)} instruments, as-of {max_asof}")

if __name__ == "__main__":
    run_forecast_batch()
