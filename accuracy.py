import sqlite3
import json
import logging
import os

import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

DEFAULT_LOG_PATH = os.path.join("data", "_forecast_log.json")


def get_db_connection(db_path="test.db"):
    return sqlite3.connect(db_path)


def evaluate_pending_forecasts(conn, horizon=1):
    """Compare archived forecasts against actuals once `horizon` trading
    sessions of OHLCV data exist past asof_date, writing results into
    forecast_accuracy. Returns the number of forecasts evaluated."""
    c = conn.cursor()
    c.execute("SELECT instrument_code, asof_date, last_price, med_json, up_json, lo_json FROM forecasts")
    forecasts = c.fetchall()

    evaluated = 0
    for inst, asof_date, last_price, med_json, up_json, lo_json in forecasts:
        c.execute("SELECT 1 FROM forecast_accuracy WHERE instrument_code = ? AND asof_date = ?", (inst, asof_date))
        if c.fetchone():
            continue

        med = json.loads(med_json) if med_json else []
        up = json.loads(up_json) if up_json else []
        lo = json.loads(lo_json) if lo_json else []
        if len(med) < horizon or len(up) < horizon or len(lo) < horizon:
            continue

        df = pd.read_sql_query(
            "SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? AND date > ? AND adjusted_close IS NOT NULL ORDER BY date ASC LIMIT ?",
            conn, params=(inst, asof_date, horizon)
        )
        if len(df) < horizon:
            continue

        target_date = df.iloc[horizon - 1]['date']
        actual_close = df.iloc[horizon - 1]['adjusted_close']
        if pd.isna(actual_close):
            continue
        actual_close = float(actual_close)

        predicted_med = med[horizon - 1]
        predicted_up = up[horizon - 1]
        predicted_lo = lo[horizon - 1]

        error_pct = ((actual_close - predicted_med) / predicted_med) * 100 if predicted_med else 0.0
        in_cone = 1 if predicted_lo <= actual_close <= predicted_up else 0
        direction_correct = 1 if (predicted_med - last_price >= 0) == (actual_close - last_price >= 0) else 0

        c.execute("""
            INSERT OR REPLACE INTO forecast_accuracy
            (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inst, asof_date, target_date, horizon, last_price,
            predicted_med, predicted_up, predicted_lo, actual_close,
            round(error_pct, 4), in_cone, direction_correct
        ))
        evaluated += 1

    conn.commit()
    return evaluated


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_calibration(conn, instrument_code, lookback_n=20):
    """Adaptive calibration derived from recently-evaluated forecasts.

    Cold start (fewer than 5 evaluated forecasts): neutral, no adjustment.
    Otherwise: bias_pct is the mean % error of P50 vs actual, and cone_scale
    widens the cone when coverage is too low or narrows it when coverage is
    consistently high.
    """
    c = conn.cursor()
    c.execute("""
        SELECT error_pct, in_cone FROM forecast_accuracy
        WHERE instrument_code = ?
        ORDER BY asof_date DESC LIMIT ?
    """, (instrument_code, lookback_n))
    rows = c.fetchall()
    n = len(rows)

    if n < 5:
        return {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": n}

    errors = [r[0] for r in rows]
    in_cones = [r[1] for r in rows]
    bias_pct = sum(errors) / n
    coverage = sum(in_cones) / n

    if coverage < 0.8:
        cone_scale = clamp(0.8 / max(coverage, 0.1), 1.0, 2.0)
    elif coverage > 0.9:
        cone_scale = 0.8
    else:
        cone_scale = 1.0

    return {"bias_pct": round(bias_pct, 4), "coverage": round(coverage, 4), "cone_scale": cone_scale, "n": n}


def load_forecast_log(path=DEFAULT_LOG_PATH):
    """Load the persisted forecast/accuracy log. Since *.db is recreated
    fresh on every pipeline run, this git-tracked JSON file is what carries
    pending forecasts and accuracy history forward across runs."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def seed_from_log(conn, log):
    """Seed a fresh DB's forecasts/forecast_accuracy tables from a persisted
    log so evaluate_pending_forecasts() and get_calibration() have history
    to work with."""
    c = conn.cursor()
    for inst, rec in log.items():
        for fc in rec.get("pending", []):
            c.execute("""
                INSERT OR IGNORE INTO forecasts
                (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json, calibration_json)
                VALUES (?, ?, ?, 0, 0, '[]', ?, ?, ?, NULL)
            """, (inst, fc["asof_date"], fc["last_price"], json.dumps(fc["med"]), json.dumps(fc["up"]), json.dumps(fc["lo"])))
        for a in rec.get("accuracy_history", []):
            c.execute("""
                INSERT OR IGNORE INTO forecast_accuracy
                (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inst, a["asof_date"], a["target_date"], a.get("horizon", 1), a.get("last_price", 0),
                a["predicted_med"], a.get("predicted_up", a["predicted_med"]), a.get("predicted_lo", a["predicted_med"]),
                a["actual_close"], a["error_pct"], int(a["in_cone"]), int(a["direction_correct"])
            ))
    conn.commit()


def save_forecast_log(conn, path=DEFAULT_LOG_PATH, horizon=1, lookback_n=20):
    """Persist not-yet-evaluated forecasts (so they get a chance to be
    scored once `horizon` sessions pass) and recent accuracy history (for
    get_calibration's lookback) to a git-tracked JSON file."""
    c = conn.cursor()
    c.execute("SELECT DISTINCT instrument_code FROM forecasts")
    instruments = [r[0] for r in c.fetchall()]

    log = {}
    for inst in instruments:
        c.execute("""
            SELECT f.asof_date, f.last_price, f.med_json, f.up_json, f.lo_json
            FROM forecasts f
            WHERE f.instrument_code = ?
              AND NOT EXISTS (
                  SELECT 1 FROM forecast_accuracy a
                  WHERE a.instrument_code = f.instrument_code AND a.asof_date = f.asof_date
              )
            ORDER BY f.asof_date DESC LIMIT ?
        """, (inst, horizon + 2))
        pending = [{
            "asof_date": r[0], "last_price": r[1],
            "med": json.loads(r[2]) if r[2] else [],
            "up": json.loads(r[3]) if r[3] else [],
            "lo": json.loads(r[4]) if r[4] else [],
        } for r in c.fetchall()]

        c.execute("""
            SELECT asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct
            FROM forecast_accuracy WHERE instrument_code = ? ORDER BY asof_date DESC LIMIT ?
        """, (inst, lookback_n))
        accuracy_history = [{
            "asof_date": r[0], "target_date": r[1], "horizon": r[2], "last_price": r[3],
            "predicted_med": r[4], "predicted_up": r[5], "predicted_lo": r[6],
            "actual_close": r[7], "error_pct": r[8], "in_cone": bool(r[9]), "direction_correct": bool(r[10])
        } for r in c.fetchall()]

        if pending or accuracy_history:
            log[inst] = {"pending": pending, "accuracy_history": accuracy_history}

    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    from forecast_batch import load_config
    config = load_config()
    db_path = config.get("database", {}).get("path", "test.db")
    horizon = config.get("forecast", {}).get("horizon", 1)
    conn = get_db_connection(db_path)
    n = evaluate_pending_forecasts(conn, horizon=horizon)
    logging.info(f"Evaluated {n} pending forecasts")
    conn.close()
