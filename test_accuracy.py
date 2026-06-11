import json
import sqlite3

import pytest
import yaml

from accuracy import (
    evaluate_pending_forecasts,
    get_calibration,
    load_forecast_log,
    seed_from_log,
    save_forecast_log,
)
from forecast_batch import apply_calibration


@pytest.fixture
def db_conn(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    yield conn, str(db_file)
    conn.close()


def insert_ohlcv(conn, code, dates_closes):
    c = conn.cursor()
    for date, close in dates_closes:
        c.execute("""
            INSERT INTO ohlcv (instrument_code, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (code, date, close, close, close, close, close, 100000))
    conn.commit()


def insert_forecast(conn, code, asof_date, last_price, med, up, lo):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json)
        VALUES (?, ?, ?, 0, 0, '[]', ?, ?, ?)
    """, (code, asof_date, last_price, json.dumps(med), json.dumps(up), json.dumps(lo)))
    conn.commit()


# --- evaluate_pending_forecasts ---

def test_evaluate_pending_forecasts_writes_accuracy_row(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    dates_closes = [(f"2024-01-{i:02}", 100 + i) for i in range(1, 11)]
    insert_ohlcv(conn, "TEST", dates_closes)

    # Forecast made as-of 2024-01-05 (close=105), targeting 5 sessions ahead
    insert_forecast(
        conn, "TEST", "2024-01-05", 105.0,
        med=[106, 107, 108, 109, 110],
        up=[108, 109, 110, 111, 112],
        lo=[104, 105, 106, 107, 108],
    )

    n = evaluate_pending_forecasts(conn, horizon=5)
    assert n == 1

    c.execute("SELECT * FROM forecast_accuracy WHERE instrument_code = 'TEST'")
    row = c.fetchone()
    assert row is not None

    cols = [d[0] for d in c.description]
    rec = dict(zip(cols, row))
    assert rec["asof_date"] == "2024-01-05"
    assert rec["target_date"] == "2024-01-10"
    assert rec["horizon"] == 5
    assert rec["actual_close"] == 110.0
    assert rec["predicted_med"] == 110
    assert rec["error_pct"] == 0.0
    assert rec["in_cone"] == 1
    assert rec["direction_correct"] == 1


def test_evaluate_pending_forecasts_skips_not_enough_sessions(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    dates_closes = [(f"2024-01-{i:02}", 100 + i) for i in range(1, 11)]
    insert_ohlcv(conn, "TEST", dates_closes)

    # Only 2 sessions (01-09, 01-10) exist after 01-08, need 5
    insert_forecast(
        conn, "TEST", "2024-01-08", 108.0,
        med=[109, 110, 111, 112, 113],
        up=[111, 112, 113, 114, 115],
        lo=[107, 108, 109, 110, 111],
    )

    n = evaluate_pending_forecasts(conn, horizon=5)
    assert n == 0

    c.execute("SELECT COUNT(*) FROM forecast_accuracy")
    assert c.fetchone()[0] == 0


def test_evaluate_pending_forecasts_idempotent(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    dates_closes = [(f"2024-01-{i:02}", 100 + i) for i in range(1, 11)]
    insert_ohlcv(conn, "TEST", dates_closes)
    insert_forecast(
        conn, "TEST", "2024-01-05", 105.0,
        med=[106, 107, 108, 109, 110],
        up=[108, 109, 110, 111, 112],
        lo=[104, 105, 106, 107, 108],
    )

    assert evaluate_pending_forecasts(conn, horizon=5) == 1
    # Second call: already evaluated, nothing new
    assert evaluate_pending_forecasts(conn, horizon=5) == 0

    c.execute("SELECT COUNT(*) FROM forecast_accuracy")
    assert c.fetchone()[0] == 1


def test_evaluate_pending_forecasts_direction_and_cone_miss(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # Price drops sharply on day 10, well below the predicted cone
    dates_closes = [(f"2024-01-{i:02}", 100) for i in range(1, 10)] + [("2024-01-10", 50)]
    insert_ohlcv(conn, "TEST", dates_closes)

    insert_forecast(
        conn, "TEST", "2024-01-05", 100.0,
        med=[101, 102, 103, 104, 105],  # predicts up
        up=[103, 104, 105, 106, 107],
        lo=[99, 100, 101, 102, 103],
    )

    n = evaluate_pending_forecasts(conn, horizon=5)
    assert n == 1

    c.execute("SELECT actual_close, in_cone, direction_correct FROM forecast_accuracy WHERE instrument_code = 'TEST'")
    actual_close, in_cone, direction_correct = c.fetchone()
    assert actual_close == 50.0
    assert in_cone == 0  # 50 is not within [103, 105]
    assert direction_correct == 0  # predicted up, actual down


# --- get_calibration ---

def insert_accuracy_rows(conn, code, error_pcts, in_cones):
    c = conn.cursor()
    for i, (err, cone) in enumerate(zip(error_pcts, in_cones)):
        c.execute("""
            INSERT INTO forecast_accuracy
            (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
            VALUES (?, ?, ?, 5, 100, 100, 110, 90, 100, ?, ?, 1)
        """, (code, f"2024-01-{i+1:02}", f"2024-01-{i+10:02}", err, cone))
    conn.commit()


def test_get_calibration_cold_start(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    insert_accuracy_rows(conn, "TEST", error_pcts=[1, 2, 3, 4], in_cones=[1, 1, 1, 1])

    cal = get_calibration(conn, "TEST")
    assert cal == {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": 4}


def test_get_calibration_no_history(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    cal = get_calibration(conn, "TEST")
    assert cal == {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": 0}


def test_get_calibration_low_coverage_widens_cone(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # 3/5 in cone -> coverage 0.6 -> cone_scale = clamp(0.8/0.6, 1.0, 2.0)
    insert_accuracy_rows(conn, "TEST", error_pcts=[2, 2, 2, 2, 2], in_cones=[1, 1, 1, 0, 0])

    cal = get_calibration(conn, "TEST")
    assert cal["bias_pct"] == 2.0
    assert cal["coverage"] == 0.6
    assert cal["cone_scale"] == pytest.approx(0.8 / 0.6)
    assert cal["n"] == 5


def test_get_calibration_high_coverage_narrows_cone(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # 5/5 in cone -> coverage 1.0 (>0.9) -> cone_scale narrows to 0.8
    insert_accuracy_rows(conn, "TEST", error_pcts=[-1, -1, -1, -1, -1], in_cones=[1, 1, 1, 1, 1])

    cal = get_calibration(conn, "TEST")
    assert cal["bias_pct"] == -1.0
    assert cal["coverage"] == 1.0
    assert cal["cone_scale"] == 0.8
    assert cal["n"] == 5


def test_get_calibration_cone_scale_clamped_to_2(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # 0/5 in cone -> coverage 0.0 -> 0.8/max(0.0,0.1)=8.0 -> clamped to 2.0
    insert_accuracy_rows(conn, "TEST", error_pcts=[0, 0, 0, 0, 0], in_cones=[0, 0, 0, 0, 0])

    cal = get_calibration(conn, "TEST")
    assert cal["coverage"] == 0.0
    assert cal["cone_scale"] == 2.0


def test_get_calibration_lookback_limits_window(db_conn):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # 25 rows, all in_cone, error_pct = 0. With lookback_n=20, only the most
    # recent 20 (by asof_date DESC) should be considered.
    error_pcts = [0] * 25
    in_cones = [1] * 25
    for i, (err, cone) in enumerate(zip(error_pcts, in_cones)):
        c.execute("""
            INSERT INTO forecast_accuracy
            (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
            VALUES (?, ?, ?, 5, 100, 100, 110, 90, 100, ?, ?, 1)
        """, ("TEST", f"2024-{(i//28)+1:02}-{(i%28)+1:02}", "2024-12-31", err, cone))
    conn.commit()

    cal = get_calibration(conn, "TEST", lookback_n=20)
    assert cal["n"] == 20


# --- apply_calibration ---

def test_apply_calibration_neutral_is_noop():
    med = [100.0, 101.0, 102.0, 103.0, 104.0]
    up = [102.0, 103.0, 104.0, 105.0, 106.0]
    lo = [98.0, 99.0, 100.0, 101.0, 102.0]
    cal = {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": 0}

    new_med, new_up, new_lo = apply_calibration(med, up, lo, last_price=100.0, calibration=cal)
    assert new_med == med
    assert new_up == up
    assert new_lo == lo


def test_apply_calibration_shifts_by_bias():
    med = [100.0, 101.0, 102.0, 103.0, 104.0]
    up = [102.0, 103.0, 104.0, 105.0, 106.0]
    lo = [98.0, 99.0, 100.0, 101.0, 102.0]
    cal = {"bias_pct": 1.0, "coverage": 0.6, "cone_scale": 1.0, "n": 10}

    # shift = last_price * bias_pct/100 = 100 * 0.01 = 1.0
    new_med, new_up, new_lo = apply_calibration(med, up, lo, last_price=100.0, calibration=cal)
    assert new_med == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert new_up == [103.0, 104.0, 105.0, 106.0, 107.0]
    assert new_lo == [99.0, 100.0, 101.0, 102.0, 103.0]


def test_apply_calibration_scales_cone_width():
    med = [100.0, 100.0]
    up = [102.0, 102.0]   # half-width 2
    lo = [98.0, 98.0]     # half-width 2
    cal = {"bias_pct": 0.0, "coverage": 0.0, "cone_scale": 2.0, "n": 10}

    new_med, new_up, new_lo = apply_calibration(med, up, lo, last_price=100.0, calibration=cal)
    assert new_med == [100.0, 100.0]
    assert new_up == [104.0, 104.0]  # half-width doubled to 4
    assert new_lo == [96.0, 96.0]


# --- forecast log persistence (DB is recreated fresh every run) ---

def test_load_forecast_log_missing_file_returns_empty(tmp_path):
    assert load_forecast_log(str(tmp_path / "nope.json")) == {}


def test_save_forecast_log_pending_excludes_evaluated(db_conn, tmp_path):
    conn, _ = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    # Already evaluated forecast
    insert_forecast(conn, "TEST", "2024-01-01", 100.0, med=[101]*5, up=[103]*5, lo=[99]*5)
    c.execute("""
        INSERT INTO forecast_accuracy
        (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
        VALUES ('TEST', '2024-01-01', '2024-01-08', 5, 100, 101, 103, 99, 101, 0.0, 1, 1)
    """)
    # Still-pending forecast
    insert_forecast(conn, "TEST", "2024-01-05", 105.0, med=[106]*5, up=[108]*5, lo=[104]*5)
    conn.commit()

    log_path = tmp_path / "log.json"
    save_forecast_log(conn, path=str(log_path), horizon=5, lookback_n=20)

    log = load_forecast_log(str(log_path))
    assert [p["asof_date"] for p in log["TEST"]["pending"]] == ["2024-01-05"]
    assert [a["asof_date"] for a in log["TEST"]["accuracy_history"]] == ["2024-01-01"]


def test_seed_from_log_repopulates_fresh_db(tmp_path):
    log = {
        "TEST": {
            "pending": [{"asof_date": "2024-01-05", "last_price": 105.0,
                          "med": [106, 107, 108, 109, 110],
                          "up": [108, 109, 110, 111, 112],
                          "lo": [104, 105, 106, 107, 108]}],
            "accuracy_history": [{"asof_date": "2024-01-01", "target_date": "2024-01-06",
                                   "horizon": 5, "last_price": 100,
                                   "predicted_med": 101, "predicted_up": 103, "predicted_lo": 99,
                                   "actual_close": 101, "error_pct": 0.0,
                                   "in_cone": True, "direction_correct": True}]
        }
    }

    db_path = tmp_path / "fresh.db"
    conn = sqlite3.connect(str(db_path))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    conn.commit()

    seed_from_log(conn, log)

    c.execute("SELECT asof_date, med_json FROM forecasts WHERE instrument_code = 'TEST'")
    row = c.fetchone()
    assert row[0] == "2024-01-05"
    assert json.loads(row[1]) == [106, 107, 108, 109, 110]

    c.execute("SELECT COUNT(*) FROM forecast_accuracy WHERE instrument_code = 'TEST'")
    assert c.fetchone()[0] == 1
    conn.close()


def test_forecast_log_enables_cross_run_evaluation(tmp_path):
    # "Run 1": fresh DB, OHLCV only through 2024-01-05, generate a forecast.
    db1 = tmp_path / "run1.db"
    conn1 = sqlite3.connect(str(db1))
    with open("schema.sql", "r") as f:
        conn1.executescript(f.read())
    c1 = conn1.cursor()
    c1.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    insert_ohlcv(conn1, "TEST", [(f"2024-01-{i:02}", 100 + i) for i in range(1, 6)])
    insert_forecast(
        conn1, "TEST", "2024-01-05", 105.0,
        med=[106, 107, 108, 109, 110],
        up=[108, 109, 110, 111, 112],
        lo=[104, 105, 106, 107, 108],
    )

    log_path = tmp_path / "forecast_log.json"
    save_forecast_log(conn1, path=str(log_path), horizon=5, lookback_n=20)
    conn1.close()

    log = load_forecast_log(str(log_path))
    assert len(log["TEST"]["pending"]) == 1
    assert log["TEST"]["accuracy_history"] == []

    # "Run 2": brand-new DB (as happens in CI), but OHLCV now extends
    # through 2024-01-10 — 5 sessions past the prior forecast's asof_date.
    db2 = tmp_path / "run2.db"
    conn2 = sqlite3.connect(str(db2))
    with open("schema.sql", "r") as f:
        conn2.executescript(f.read())
    c2 = conn2.cursor()
    c2.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    insert_ohlcv(conn2, "TEST", [(f"2024-01-{i:02}", 100 + i) for i in range(1, 11)])

    seed_from_log(conn2, log)
    n = evaluate_pending_forecasts(conn2, horizon=5)
    assert n == 1

    c2.execute("SELECT actual_close FROM forecast_accuracy WHERE instrument_code = 'TEST'")
    assert c2.fetchone()[0] == 110.0

    cal = get_calibration(conn2, "TEST")
    assert cal["n"] == 1  # cold start: only 1 evaluated forecast so far


# --- export_json integration ---

def test_export_includes_calibration_and_accuracy(tmp_path, monkeypatch):
    # Use GOLDBEES (an "is_index"-style symbol in export_json.py) so the
    # fundamentals fetch path skips yfinance entirely and stays offline.
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    c = conn.cursor()

    c.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTY 50', 'Broad Market', 'market')")
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDETF', 'Gold ETFs', 'index')")
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDBEES', 'GOLDBEES', 'stock')")
    c.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('GOLDETF', 'GOLDBEES', '2024-01-01')")

    for i in range(1, 11):
        c.execute("""
            INSERT INTO ohlcv (instrument_code, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ('GOLDBEES', f'2024-01-{i:02}', 100, 100, 100, 100, 100 + i, 100000))

    # Forecast (most recent), with calibration_json
    calibration = {"bias_pct": 0.5, "coverage": 0.6, "cone_scale": 1.33, "n": 5}
    c.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json, calibration_json)
        VALUES ('GOLDBEES', '2024-01-10', 110.0, 1.0, 5.0, '[]', '[111,112,113,114,115]', '[113,114,115,116,117]', '[109,110,111,112,113]', ?)
    """, (json.dumps(calibration),))

    # Accuracy history row from an earlier evaluated forecast
    c.execute("""
        INSERT INTO forecast_accuracy
        (instrument_code, asof_date, target_date, horizon, last_price, predicted_med, predicted_up, predicted_lo, actual_close, error_pct, in_cone, direction_correct)
        VALUES ('GOLDBEES', '2024-01-01', '2024-01-06', 5, 101, 106, 108, 104, 106, 0.0, 1, 1)
    """)
    conn.commit()
    conn.close()

    config = {'database': {'path': str(db_path)}}
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    import os
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    from export_json import export_data
    data_dir = tmp_path / "data"
    export_data(output_dir=str(data_dir))

    with open(data_dir / "GOLDBEES.json", "r") as f:
        data = json.load(f)

    assert data["calibration"] == calibration

    assert data["accuracy"]["n"] == 1
    assert data["accuracy"]["mae_pct"] == 0.0
    assert data["accuracy"]["directional_accuracy_pct"] == 100.0
    assert data["accuracy"]["calibration_pct"] == 100.0
    hist = data["accuracy"]["history"][0]
    assert hist["asof_date"] == "2024-01-01"
    assert hist["target_date"] == "2024-01-06"
    assert hist["predicted_med"] == 106
    assert hist["actual_close"] == 106
    assert hist["in_cone"] is True
    assert hist["direction_correct"] is True
