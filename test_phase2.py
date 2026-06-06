import sqlite3
import pandas as pd
import pytest
import json
import os
from jsonschema import validate
from forecast_batch import run_forecast_batch, get_db_connection
from fetch_eod import run_fetcher, init_instruments
from adjust_corporate_actions import run_adjustments
from export_json import export_data

@pytest.fixture
def db_conn(tmp_path):
    db_file = tmp_path / "test.db"

    # Initialize schema
    conn = sqlite3.connect(str(db_file))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    yield conn, str(db_file)
    conn.close()


@pytest.fixture
def sample_data(db_conn):
    conn, db_path = db_conn
    c = conn.cursor()
    # Market must exist for export_json to work
    c.execute("INSERT OR IGNORE INTO instruments (code, name, level) VALUES ('NIFTY 50', 'NIFTY 50', 'market')")
    c.execute("INSERT OR IGNORE INTO instruments (code, name, level) VALUES ('TCS', 'Tata Consultancy', 'stock')")

    # insert dummy data so chronos-t5-base has something to forecast
    base_val = 100
    for i in range(1, 20):
        c.execute("""
            INSERT OR REPLACE INTO ohlcv (instrument_code, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ('TCS', f'2024-01-{i:02}', base_val, base_val+5, base_val-5, base_val, base_val + (i*2), 1000))

    conn.commit()
    return conn, db_path

@pytest.mark.integration
def test_monotonic_widening(sample_data):
    conn, db_path = sample_data
    run_forecast_batch(db_path)

    c = conn.cursor()
    c.execute("SELECT up_json, lo_json FROM forecasts WHERE instrument_code = 'TCS'")
    res = c.fetchone()

    assert res is not None, "Forecast was not generated"
    up = json.loads(res[0])
    lo = json.loads(res[1])

    assert len(up) > 1 and len(lo) > 1

    diff_d1 = up[0] - lo[0]
    diff_d5 = up[-1] - lo[-1]

    assert diff_d5 > diff_d1, "Uncertainty cone does not widen monotonically"

@pytest.mark.integration
def test_json_schema(sample_data, tmp_path, monkeypatch):
    conn, db_path = sample_data
    run_forecast_batch(db_path)

    # patch export_data's read of config.yaml
    def mock_get_db_connection(mock_db_path="test.db"):
        return sqlite3.connect(db_path)

    monkeypatch.setattr("export_json.get_db_connection", mock_get_db_connection)

    export_data(output_dir=str(tmp_path))

    # Verify export file exists
    tcs_file = tmp_path / "TCS.json"
    assert tcs_file.exists()

    with open(tcs_file, "r") as f:
        data = json.load(f)

    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "last": {"type": "number"},
            "hist": {
                "type": "array",
                "items": {"type": "number"}
            },
            "med": {
                "type": "array",
                "items": {"type": "number"}
            },
            "up": {
                "type": "array",
                "items": {"type": "number"}
            },
            "lo": {
                "type": "array",
                "items": {"type": "number"}
            },
            "ret": {"type": "number"},
            "cone_width_pct": {"type": "number"},
            "asof": {"type": "string"}
        },
        "required": ["code", "last", "hist", "med", "up", "lo", "ret", "cone_width_pct", "asof"]
    }

    # Should not raise exception
    validate(instance=data, schema=schema)
