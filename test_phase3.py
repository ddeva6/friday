import pytest
import os
import json
import sqlite3
import pandas as pd
from export_json import export_data

@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    # Add market
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTY 50', 'Broad Market', 'market')")
    # Add index
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTYBANK', 'Nifty Bank', 'index')")
    # Add stock
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('HDFCBANK', 'HDFC Bank', 'stock')")
    conn.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('NIFTYBANK', 'HDFCBANK', '2024-01-01')")

    # Add OHLCV
    dates = pd.date_range(start="2024-01-01", periods=100).strftime('%Y-%m-%d')
    for d in dates:
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close) VALUES ('HDFCBANK', ?, 1500.0)", (d,))
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close) VALUES ('NIFTYBANK', ?, 45000.0)", (d,))
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close) VALUES ('NIFTY 50', ?, 22000.0)", (d,))

    # Add forecast
    conn.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, med_json, up_json, lo_json)
        VALUES ('HDFCBANK', '2024-04-09', 1500.0, 2.5, 5.0, '[1510, 1520]', '[1530, 1540]', '[1490, 1480]')
    """)

    conn.commit()
    conn.close()
    return db_path

def test_no_synthetic_fallback():
    with open("friday-forecast-terminal.html", "r") as f:
        content = f.read()
    assert "FALLBACK_UNIVERSE" not in content

def test_export_json_structure(sample_db, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    # Mock config
    config = {'database': {'path': str(sample_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    # Copy schema.sql to tmp_path so export_data can find it
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    export_data(output_dir=str(data_dir))

    assert (data_dir / "index.json").exists()
    assert (data_dir / "HDFCBANK.json").exists()

    with open(data_dir / "HDFCBANK.json", "r") as f:
        data = json.load(f)

    # test_export_json_has_fundamentals
    assert "fundamentals" in data
    assert isinstance(data["fundamentals"]["hi_52w"], (int, float))
    assert isinstance(data["fundamentals"]["lo_52w"], (int, float))

    # test_export_json_has_status
    assert "status" in data
    assert isinstance(data["status"], list)

    # test_export_json_has_forecast_fields
    assert "med" in data
    assert "up" in data
    assert "lo" in data
    assert "ret" in data
    assert "cone_width_pct" in data

def test_screener_risk_adjusted_default():
    with open("friday-forecast-terminal.html", "r") as f:
        content = f.read()
    assert "sortKey='risk'" in content

def test_not_investment_advice_notice():
    with open("friday-forecast-terminal.html", "r") as f:
        content = f.read()
    assert "Not investment advice" in content

def test_watch_avoid_labels():
    with open("friday-forecast-terminal.html", "r") as f:
        content = f.read()
    assert "WATCH" in content
    assert "AVOID" in content
    assert "BUY" not in content
    assert "SELL" not in content

def test_data_contract_schema(sample_db, tmp_path, monkeypatch):
    from jsonschema import validate
    data_dir = tmp_path / "data"

    config = {'database': {'path': str(sample_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    export_data(output_dir=str(data_dir))

    with open(data_dir / "HDFCBANK.json", "r") as f:
        data = json.load(f)

    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "last": {"type": "number"},
            "hist": {"type": "array", "items": {"type": "number"}},
            "med": {"type": "array", "items": {"type": "number"}},
            "up": {"type": "array", "items": {"type": "number"}},
            "lo": {"type": "array", "items": {"type": "number"}},
            "ret": {"type": "number"},
            "cone_width_pct": {"type": "number"},
            "asof": {"type": "string"},
            "fundamentals": {
                "type": "object",
                "properties": {
                    "mcap_cr": {"type": ["number", "string"]},
                    "pe": {"type": ["number", "string"]},
                    "roe": {"type": ["number", "string"]},
                    "de": {"type": ["number", "string"]},
                    "sales_growth_yoy": {"type": ["number", "string"]},
                    "hi_52w": {"type": "number"},
                    "lo_52w": {"type": "number"}
                }
            },
            "status": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["code", "last", "hist", "med", "up", "lo", "ret", "cone_width_pct", "asof", "fundamentals", "status"]
    }

    validate(instance=data, schema=schema)
