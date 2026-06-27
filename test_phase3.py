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

@pytest.fixture
def gold_etf_db(tmp_path):
    db_path = tmp_path / "test_gold.db"
    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTY 50', 'Broad Market', 'market')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDETF', 'Gold ETFs', 'index')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDBEES', 'GOLDBEES', 'stock')")
    conn.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('GOLDETF', 'GOLDBEES', '2024-01-01')")

    dates = pd.date_range(start="2024-01-01", periods=100).strftime('%Y-%m-%d')
    for d in dates:
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close, volume) VALUES ('GOLDBEES', ?, 65.0, 500000)", (d,))

    conn.commit()
    conn.close()
    return db_path


def test_gold_etf_skips_fundamentals_fetch(gold_etf_db, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    config = {'database': {'path': str(gold_etf_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    # No yfinance mock needed: GOLDBEES is treated as an index-like instrument,
    # so the fundamentals fetch (and any network call) is skipped entirely.
    export_data(output_dir=str(data_dir))

    assert (data_dir / "GOLDBEES.json").exists()
    with open(data_dir / "GOLDBEES.json", "r") as f:
        data = json.load(f)

    assert data["fundamentals"]["mcap_cr"] == "N/A"
    assert data["fundamentals"]["pe"] == "N/A"

    with open(data_dir / "index.json", "r") as f:
        idx = json.load(f)
    gold_group = next((c for c in idx["children"] if c["code"] == "GOLDETF"), None)
    assert gold_group is not None
    assert "GOLDBEES" in gold_group["stocks"]


def test_gold_etf_handles_null_adjusted_close(gold_etf_db, tmp_path, monkeypatch):
    # Simulate yfinance returning a NaN/NULL close for the most recent session,
    # which previously leaked into the JSON export as a bare `NaN` token and
    # broke JSON.parse() in the dashboard.
    conn = sqlite3.connect(gold_etf_db)
    conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close, volume) VALUES ('GOLDBEES', '2024-04-10', NULL, 500000)")
    conn.commit()
    conn.close()

    data_dir = tmp_path / "data"

    config = {'database': {'path': str(gold_etf_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    export_data(output_dir=str(data_dir))

    raw = (data_dir / "GOLDBEES.json").read_text()
    assert "NaN" not in raw

    data = json.loads(raw)
    assert data["last"] == 65.0
    assert data["asof"] == "2024-04-09"
    assert all(v == 65.0 for v in data["hist"])


@pytest.fixture
def synthetic_index_db(tmp_path):
    db_path = tmp_path / "test_synth.db"
    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTY 50', 'Broad Market', 'market')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDETF', 'Gold ETFs', 'index')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('GOLDBEES', 'GOLDBEES', 'stock')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('AXISGOLD', 'AXISGOLD', 'stock')")
    conn.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('GOLDETF', 'GOLDBEES', '2024-01-01')")
    conn.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('GOLDETF', 'AXISGOLD', '2024-01-01')")

    dates = pd.date_range(start="2024-01-01", periods=100).strftime('%Y-%m-%d')
    for d in dates:
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close, volume) VALUES ('GOLDBEES', ?, 65.0, 500000)", (d,))
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close, volume) VALUES ('AXISGOLD', ?, 100.0, 500000)", (d,))

    conn.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, med_json, up_json, lo_json)
        VALUES ('GOLDBEES', '2024-04-09', 65.0, 2.0, 5.0, '[66.3]', '[67.0]', '[64.0]')
    """)
    conn.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, med_json, up_json, lo_json)
        VALUES ('AXISGOLD', '2024-04-09', 100.0, 2.0, 5.0, '[102.0]', '[103.0]', '[101.0]')
    """)

    conn.commit()
    conn.close()
    return db_path


def test_synthetic_index_built_from_constituents(synthetic_index_db, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    config = {'database': {'path': str(synthetic_index_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    export_data(output_dir=str(data_dir))

    assert (data_dir / "GOLDETF.json").exists()
    with open(data_dir / "GOLDETF.json", "r") as f:
        data = json.load(f)

    # Both constituents are rebased to 100 and have an identical 2% forecast
    # move, so the synthetic index should also be at 100 with a ~2% forecast.
    assert data["last"] == 100.0
    assert data["med"] == [102.0]
    assert data["ret"] == pytest.approx(2.0)
    assert data["fundamentals"]["pe"] == "N/A"
    assert any("Synthetic index" in s for s in data["status"])


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

    # test_export_json_has_generated_at
    with open(data_dir / "index.json", "r") as f:
        idx_data = json.load(f)
    assert "generated_at" in idx_data

    # Check if generated_at is a valid ISO 8601 string ending in 'Z'
    dt_str = idx_data["generated_at"]
    assert dt_str.endswith("Z")

    from datetime import datetime
    try:
        datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        pytest.fail(f"generated_at '{dt_str}' is not in valid ISO-8601 format.")

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


def test_rsi_bounds(sample_db, tmp_path, monkeypatch):
    # sample_db's HDFCBANK history is a flat 1500.0 for every session, so RSI
    # has no price movement to measure and must come back as None, not NaN.
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

    rsi = data["trading"]["momentum"]["rsi_14"]
    assert rsi is None or (isinstance(rsi, (int, float)) and 0 <= rsi <= 100)


@pytest.fixture
def favorable_short_history_db(tmp_path):
    """A stock with a <60-row (short) history but otherwise favorable signals:
    a high risk:reward forecast, a large expected move, and volume well above
    its own average. Without the data-quality override, conviction would land
    in the HIGH tier and the verdict would be FAVORABLE — this fixture exists
    to prove the short-history flag forces CAUTION anyway."""
    db_path = tmp_path / "test_favorable_short.db"
    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTY 50', 'Broad Market', 'market')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NIFTYBANK', 'Nifty Bank', 'index')")
    conn.execute("INSERT INTO instruments (code, name, level) VALUES ('NEWBANK', 'New Bank', 'stock')")
    conn.execute("INSERT INTO index_membership (index_code, stock_code, start_date) VALUES ('NIFTYBANK', 'NEWBANK', '2024-01-01')")

    dates = pd.date_range(start="2024-01-01", periods=55).strftime('%Y-%m-%d')
    for i, d in enumerate(dates):
        is_last = i == len(dates) - 1
        conn.execute(
            "INSERT INTO ohlcv (instrument_code, date, adjusted_close, volume) VALUES ('NEWBANK', ?, 1500.0, ?)",
            (d, 200000 if is_last else 100000),
        )
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close) VALUES ('NIFTYBANK', ?, 45000.0)", (d,))
        conn.execute("INSERT INTO ohlcv (instrument_code, date, adjusted_close) VALUES ('NIFTY 50', ?, 22000.0)", (d,))

    conn.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, med_json, up_json, lo_json)
        VALUES ('NEWBANK', ?, 1500.0, 3.0, 5.0, '[1525]', '[1530]', '[1490]')
    """, (dates[-1],))

    conn.commit()
    conn.close()
    return db_path


def test_verdict_stale_data_forces_caution(favorable_short_history_db, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    config = {'database': {'path': str(favorable_short_history_db)}}
    import yaml
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    monkeypatch.chdir(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
        with open(tmp_path / "schema.sql", "w") as f_out:
            f_out.write(f_in.read())

    export_data(output_dir=str(data_dir))

    with open(data_dir / "NEWBANK.json", "r") as f:
        data = json.load(f)

    trading = data["trading"]
    # Confirm the fixture really is otherwise favorable, so the CAUTION below
    # can only be coming from the data-quality override, not from weak inputs.
    assert trading["conviction"] >= 70
    assert trading["risk_reward"] >= 2
    assert any(s.startswith("Short history") for s in data["status"])

    assert trading["verdict"]["label"] == "CAUTION"
    assert any(r.startswith(("Stale data", "Short history")) for r in trading["verdict"]["reasons"])


def test_verdict_label_values(sample_db, gold_etf_db, tmp_path, monkeypatch):
    for db, code in [(sample_db, "HDFCBANK"), (gold_etf_db, "GOLDBEES")]:
        data_dir = tmp_path / f"data_{code}"
        config = {'database': {'path': str(db)}}
        import yaml
        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        monkeypatch.chdir(tmp_path)
        with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f_in:
            with open(tmp_path / "schema.sql", "w") as f_out:
                f_out.write(f_in.read())

        export_data(output_dir=str(data_dir))

        with open(data_dir / f"{code}.json", "r") as f:
            data = json.load(f)

        label = data["trading"]["verdict"]["label"]
        assert label in ("FAVORABLE", "NEUTRAL", "CAUTION")
        assert "BUY" not in label and "SELL" not in label
