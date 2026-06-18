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

def apply_calibration(med, up, lo, last_price, calibration):
    """Shift med/up/lo by the calibration's bias_pct (relative to last_price)
    and scale the up/lo half-widths by cone_scale. Neutral calibration
    (bias_pct=0, cone_scale=1.0) leaves the cone unchanged."""
    bias_pct = calibration.get("bias_pct", 0.0)
    cone_scale = calibration.get("cone_scale", 1.0)

    half_up = [u - m for u, m in zip(up, med)]
    half_lo = [m - l for m, l in zip(med, lo)]
    shift = last_price * (bias_pct / 100.0)

    new_med = [round(m + shift, 2) for m in med]
    new_up = [round(new_med[i] + half_up[i] * cone_scale, 2) for i in range(len(med))]
    new_lo = [round(new_med[i] - half_lo[i] * cone_scale, 2) for i in range(len(med))]

    return new_med, new_up, new_lo


def prices_to_log_returns(prices):
    prices = np.array(prices, dtype=np.float64)
    prices = prices[prices > 0]
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


def log_returns_to_prices(last_price, log_return_forecasts):
    prices = []
    p = last_price
    for lr in log_return_forecasts:
        p = p * np.exp(lr)
        prices.append(p)
    return prices


def compute_adaptive_lookback(prices, min_lb=60, max_lb=365):
    if len(prices) < min_lb:
        return len(prices)
    recent = prices[-60:]
    daily_returns = np.diff(recent) / recent[:-1]
    vol = np.std(daily_returns) * np.sqrt(252)
    if vol > 0.45:
        return min_lb
    elif vol > 0.30:
        return 120
    elif vol > 0.15:
        return 180
    else:
        return min(max_lb, len(prices))


def ema_forecast(prices, horizon, span=20):
    series = pd.Series(prices)
    ema = series.ewm(span=span, adjust=False).mean().iloc[-1]
    last = prices[-1]
    forecasts = []
    for i in range(1, horizon + 1):
        blend = last + (ema - last) * (i / horizon)
        forecasts.append(blend)
    return forecasts


def drift_forecast(prices, horizon):
    prices = np.array(prices, dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    if len(log_ret) < 5:
        return [prices[-1]] * horizon
    mu = np.mean(log_ret)
    last = prices[-1]
    return [last * np.exp(mu * i) for i in range(1, horizon + 1)]


def ensemble_with_baselines(chronos_med, prices, horizon, chronos_weight=0.6):
    ema = ema_forecast(prices, horizon)
    drift = drift_forecast(prices, horizon)
    baseline_weight = (1.0 - chronos_weight) / 2.0
    ensembled = []
    for i in range(horizon):
        val = (chronos_weight * chronos_med[i] +
               baseline_weight * ema[i] +
               baseline_weight * drift[i])
        ensembled.append(round(val, 2))
    return ensembled


def run_forecast_batch(db_path=None):
    import torch
    from chronos import ChronosPipeline

    config = load_config()
    db_path = db_path or config.get("database", {}).get("path", "test.db")
    forecast_config = config.get("forecast", {})

    lookback = forecast_config.get("lookback", 180)
    horizon = forecast_config.get("horizon", 1)
    sample_count = forecast_config.get("sample_count", 50)
    model_name = forecast_config.get("model", "amazon/chronos-t5-base")
    seed = forecast_config.get("seed", 42)
    use_log_returns = forecast_config.get("use_log_returns", True)
    ensemble_lookbacks = forecast_config.get("ensemble_lookbacks", None)
    adaptive = forecast_config.get("adaptive_lookback", False)

    logging.info("FRIDAY forecast batch starting")
    logging.info(f"Model: {model_name}")
    logging.info(f"Config: lookback={lookback}, horizon={horizon}, sample_count={sample_count}")
    logging.info(f"Log-returns: {use_log_returns}, Adaptive: {adaptive}, Ensemble: {ensemble_lookbacks}")
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

    # *.db is recreated fresh on every pipeline run, so restore pending
    # forecasts and accuracy history from the git-tracked log before
    # comparing past forecasts vs actuals. This keeps calibration below
    # informed by the freshest accuracy history.
    from accuracy import evaluate_pending_forecasts, get_calibration, load_forecast_log, seed_from_log, save_forecast_log
    seed_from_log(conn, load_forecast_log())
    n_evaluated = evaluate_pending_forecasts(conn, horizon=horizon)
    logging.info(f"Evaluated {n_evaluated} pending forecasts against actuals")

    # Get all instruments
    c.execute("SELECT DISTINCT instrument_code FROM ohlcv")
    instruments = [r[0] for r in c.fetchall()]

    for inst in instruments:
        df = pd.read_sql_query(
            "SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? ORDER BY date ASC",
            conn, params=(inst,))
        if df.empty or len(df) < 5:
            continue

        hist = df['adjusted_close'].tolist()
        asof_date = df.iloc[-1]['date']
        last_price = hist[-1]

        if adaptive:
            effective_lookback = compute_adaptive_lookback(hist, min_lb=60, max_lb=lookback)
        else:
            effective_lookback = lookback

        lookbacks_to_run = [effective_lookback]
        if ensemble_lookbacks and len(hist) > min(ensemble_lookbacks):
            lookbacks_to_run = [lb for lb in ensemble_lookbacks if lb <= len(hist)]
            if not lookbacks_to_run:
                lookbacks_to_run = [effective_lookback]

        all_samples = []

        for lb in lookbacks_to_run:
            window = hist[-lb:] if lb <= len(hist) else hist

            if use_log_returns and len(window) > 3:
                context_data = prices_to_log_returns(window)
                use_lr_this = len(context_data) >= 3
            else:
                context_data = np.array(window, dtype=np.float32)
                use_lr_this = False

            if not use_lr_this:
                context_data = np.array(window, dtype=np.float32)

            context = torch.tensor(context_data)
            torch.manual_seed(seed)
            forecast = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=sample_count,
            )
            forecast_np = forecast[0].cpu().numpy()

            if use_lr_this:
                price_samples = np.zeros_like(forecast_np)
                for s in range(forecast_np.shape[0]):
                    price_samples[s] = log_returns_to_prices(last_price, forecast_np[s])
                all_samples.append(price_samples)
            else:
                all_samples.append(forecast_np)

        combined = np.concatenate(all_samples, axis=0)

        med = np.percentile(combined, 50, axis=0).tolist()
        up = np.percentile(combined, 90, axis=0).tolist()
        lo = np.percentile(combined, 10, axis=0).tolist()

        med = ensemble_with_baselines(med, hist, horizon, chronos_weight=0.6)

        med = [round(x, 2) for x in med]
        up = [round(x, 2) for x in up]
        lo = [round(x, 2) for x in lo]
        hist = [round(x, 2) for x in hist]

        ret, cone_width_pct = calculate_metrics(last_price, med, up, lo)

        up, lo = ensure_monotonic_widening(med, up, lo, horizon)
        if up != lo:
            cone_width_pct = calculate_metrics(last_price, med, up, lo)[1]

        calibration = get_calibration(conn, inst)
        med, up, lo = apply_calibration(med, up, lo, last_price, calibration)
        ret, cone_width_pct = calculate_metrics(last_price, med, up, lo)

        c.execute("""
            INSERT OR REPLACE INTO forecasts
            (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json, calibration_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inst, asof_date, last_price, ret, cone_width_pct,
            json.dumps(hist), json.dumps(med), json.dumps(up), json.dumps(lo), json.dumps(calibration)
        ))

        print(f"Forecasted {inst}")

    conn.commit()

    save_forecast_log(conn, horizon=horizon)

    c.execute("SELECT MAX(asof_date) FROM forecasts")
    max_asof = c.fetchone()[0]

    conn.close()
    logging.info(f"Forecast batch complete: {len(instruments)} instruments, as-of {max_asof}")

if __name__ == "__main__":
    run_forecast_batch()
