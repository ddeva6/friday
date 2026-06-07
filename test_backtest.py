import sqlite3
import json
import pytest
from backtest import summarize_backtest, print_report


def test_summarize_backtest_basic():
    results = [
        {"instrument_code": "TCS", "total_trials": 10, "model_hits": 7,
         "model_hit_rate": 70.0, "naive_hits": 4, "naive_hit_rate": 40.0, "edge": 30.0},
        {"instrument_code": "INFY", "total_trials": 10, "model_hits": 5,
         "model_hit_rate": 50.0, "naive_hits": 6, "naive_hit_rate": 60.0, "edge": -10.0},
    ]
    s = summarize_backtest(results)
    assert s["total_stocks"] == 2
    assert s["total_trials"] == 20
    assert s["model_hits"] == 12
    assert s["model_hit_rate"] == 60.0
    assert s["naive_hits"] == 10
    assert s["naive_hit_rate"] == 50.0
    assert s["edge"] == 10.0


def test_summarize_backtest_empty():
    s = summarize_backtest([])
    assert s["total_stocks"] == 0
    assert s["total_trials"] == 0
    assert s["model_hit_rate"] == 0
    assert s["edge"] == 0


def test_summarize_backtest_single():
    results = [
        {"instrument_code": "SBIN", "total_trials": 5, "model_hits": 3,
         "model_hit_rate": 60.0, "naive_hits": 2, "naive_hit_rate": 40.0, "edge": 20.0},
    ]
    s = summarize_backtest(results)
    assert s["total_stocks"] == 1
    assert s["model_hit_rate"] == 60.0
    assert s["edge"] == 20.0


def test_backtest_schema():
    conn = sqlite3.connect(":memory:")
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    c = conn.cursor()

    c.execute("""
        INSERT INTO backtest_results
        (instrument_code, run_date, total_trials, model_hits, model_hit_rate,
         naive_hits, naive_hit_rate, edge, config_json, trials_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("TCS", "2024-06-01", 20, 12, 60.0, 10, 50.0, 10.0,
          json.dumps({"lookback": 180}), json.dumps([])))
    conn.commit()

    c.execute("SELECT * FROM backtest_results WHERE instrument_code = 'TCS'")
    row = c.fetchone()
    assert row is not None
    assert row[0] == "TCS"
    assert row[2] == 20
    assert row[4] == 60.0
    assert row[7] == 10.0
    conn.close()


def test_backtest_config():
    from forecast_batch import load_config
    config = load_config()
    bt = config.get("backtest", {})
    assert bt.get("step") == 5


def test_print_report_runs(capsys):
    results = [
        {"instrument_code": "TCS", "total_trials": 10, "model_hits": 6,
         "model_hit_rate": 60.0, "naive_hits": 5, "naive_hit_rate": 50.0, "edge": 10.0},
    ]
    summary = summarize_backtest(results)
    print_report(summary, results)
    captured = capsys.readouterr()
    assert "Directional Hit-Rate Report" in captured.out
    assert "TCS" in captured.out
    assert "60.0%" in captured.out


@pytest.mark.integration
def test_backtest_e2e(tmp_path):
    from backtest import run_backtest
    import numpy as np

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test Stock', 'stock')")

    np.random.seed(42)
    price = 100.0
    for i in range(250):
        price *= 1 + np.random.normal(0.001, 0.02)
        date = f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
        c.execute("INSERT OR IGNORE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  ('TEST', date, price, price*1.02, price*0.98, price, price, 100000))
    conn.commit()
    conn.close()

    results = run_backtest(db_path=db_path)
    assert results is not None
    assert len(results) > 0
    assert results[0]["total_trials"] > 0
    assert 0 <= results[0]["model_hit_rate"] <= 100
