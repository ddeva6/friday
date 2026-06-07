import sqlite3
import json
import pytest
from forecast_batch import calculate_metrics, ensure_monotonic_widening, load_config
from jsonschema import validate


@pytest.fixture
def db_conn(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    yield conn, str(db_file)
    conn.close()


@pytest.fixture
def sample_data(db_conn):
    conn, db_path = db_conn
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO instruments (code, name, level) VALUES ('NIFTY 50', 'NIFTY 50', 'market')")
    c.execute("INSERT OR IGNORE INTO instruments (code, name, level) VALUES ('TCS', 'Tata Consultancy', 'stock')")
    base_val = 100
    for i in range(1, 20):
        c.execute("""
            INSERT OR REPLACE INTO ohlcv (instrument_code, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ('TCS', f'2024-01-{i:02}', base_val, base_val+5, base_val-5, base_val, base_val + (i*2), 1000))
    conn.commit()
    return conn, db_path


# --- Offline unit tests (no torch/GPU required) ---

def test_calculate_metrics_basic():
    last = 100.0
    med = [101.0, 102.0, 103.0, 104.0, 105.0]
    up  = [102.0, 104.0, 106.0, 108.0, 110.0]
    lo  = [100.0, 100.0, 100.0, 100.0, 100.0]
    ret, cone = calculate_metrics(last, med, up, lo)
    assert ret == 5.0
    assert cone == 10.0


def test_calculate_metrics_zero_price():
    ret, cone = calculate_metrics(0.0, [1.0], [2.0], [0.5])
    assert ret == 0.0
    assert cone == 0.0


def test_calculate_metrics_empty():
    ret, cone = calculate_metrics(100.0, [], [], [])
    assert ret == 0.0
    assert cone == 0.0


def test_monotonic_widening_already_monotonic():
    med = [100.0, 101.0, 102.0, 103.0, 104.0]
    up  = [101.0, 103.0, 105.0, 107.0, 110.0]
    lo  = [99.0,  98.0,  97.0,  96.0,  95.0]
    new_up, new_lo = ensure_monotonic_widening(med, up, lo, 5)
    assert new_up == up
    assert new_lo == lo


def test_monotonic_widening_fixes_non_monotonic():
    med = [100.0, 101.0, 102.0, 103.0, 104.0]
    up  = [110.0, 108.0, 106.0, 105.0, 104.5]
    lo  = [90.0,  92.0,  98.0,  101.0, 103.5]
    new_up, new_lo = ensure_monotonic_widening(med, up, lo, 5)
    diffs = [new_up[i] - new_lo[i] for i in range(5)]
    for i in range(1, 5):
        assert diffs[i] > diffs[i-1], f"Cone not widening at step {i}: {diffs}"


def test_config_has_forecast_params():
    config = load_config()
    fc = config.get("forecast", {})
    assert fc.get("lookback") == 180
    assert fc.get("horizon") == 5
    assert fc.get("sample_count") in (20, 30)
    assert fc.get("model") is not None


def test_forecast_json_contract(db_conn):
    conn, db_path = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('TEST', 'Test', 'stock')")
    for i in range(1, 11):
        c.execute("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  ('TEST', f'2024-01-{i:02}', 100, 105, 95, 100, 100+i, 1000))

    med = [101.0, 102.0, 103.0, 104.0, 105.0]
    up  = [102.0, 104.0, 106.0, 108.0, 110.0]
    lo  = [100.0, 99.0, 98.0, 97.0, 96.0]
    hist = [100+i for i in range(1, 11)]
    ret, cone = calculate_metrics(110.0, med, up, lo)

    c.execute("""
        INSERT INTO forecasts (instrument_code, asof_date, last_price, ret, cone_width_pct, hist_json, med_json, up_json, lo_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ('TEST', '2024-01-10', 110.0, ret, cone,
          json.dumps(hist), json.dumps(med), json.dumps(up), json.dumps(lo)))
    conn.commit()

    c.execute("SELECT * FROM forecasts WHERE instrument_code = 'TEST'")
    row = c.fetchone()
    assert row is not None

    data = {
        "code": "TEST",
        "last": row[2],
        "hist": json.loads(row[5]),
        "med": json.loads(row[6]),
        "up": json.loads(row[7]),
        "lo": json.loads(row[8]),
        "ret": row[3],
        "cone_width_pct": row[4],
        "asof": row[1]
    }
    schema = {
        "type": "object",
        "required": ["code", "last", "hist", "med", "up", "lo", "ret", "cone_width_pct", "asof"],
        "properties": {
            "code": {"type": "string"},
            "last": {"type": "number"},
            "hist": {"type": "array", "items": {"type": "number"}},
            "med":  {"type": "array", "items": {"type": "number"}, "minItems": 5, "maxItems": 5},
            "up":   {"type": "array", "items": {"type": "number"}, "minItems": 5, "maxItems": 5},
            "lo":   {"type": "array", "items": {"type": "number"}, "minItems": 5, "maxItems": 5},
            "ret": {"type": "number"},
            "cone_width_pct": {"type": "number"},
            "asof": {"type": "string"}
        }
    }
    validate(instance=data, schema=schema)


# --- Integration tests (require torch + chronos + GPU) ---

@pytest.mark.integration
def test_monotonic_widening_e2e(sample_data):
    from forecast_batch import run_forecast_batch
    conn, db_path = sample_data
    run_forecast_batch(db_path)

    c = conn.cursor()
    c.execute("SELECT up_json, lo_json FROM forecasts WHERE instrument_code = 'TCS'")
    res = c.fetchone()
    assert res is not None, "Forecast was not generated"
    up = json.loads(res[0])
    lo = json.loads(res[1])
    assert len(up) > 1 and len(lo) > 1
    assert (up[-1] - lo[-1]) > (up[0] - lo[0]), "Uncertainty cone does not widen monotonically"


@pytest.mark.integration
def test_export_with_forecasts(sample_data, tmp_path, monkeypatch):
    from forecast_batch import run_forecast_batch
    from export_json import export_data
    conn, db_path = sample_data
    run_forecast_batch(db_path)

    def mock_get_db_connection(mock_db_path="test.db"):
        return sqlite3.connect(db_path)
    monkeypatch.setattr("export_json.get_db_connection", mock_get_db_connection)
    export_data(output_dir=str(tmp_path))

    tcs_file = tmp_path / "TCS.json"
    assert tcs_file.exists()
    with open(tcs_file, "r") as f:
        data = json.load(f)
    assert len(data["med"]) == 5
    assert len(data["up"]) == 5
    assert len(data["lo"]) == 5
