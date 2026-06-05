import sqlite3
import pandas as pd
import pytest
from datetime import datetime, timedelta
from fetch_eod import run_fetcher, init_instruments, get_db_connection
from adjust_corporate_actions import run_adjustments

@pytest.fixture
def db_conn(tmp_path):
    db_file = tmp_path / "test.db"

    # Initialize schema
    conn = sqlite3.connect(str(db_file))
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())

    yield conn, str(db_file)
    conn.close()

def test_fetcher_idempotency(db_conn):
    conn, db_path = db_conn

    # Use dates in the past (e.g. 1 month ago) so yfinance returns data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    run_fetcher(db_path, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM ohlcv")
    count1 = c.fetchone()[0]

    # Run fetcher again
    run_fetcher(db_path, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

    c.execute("SELECT COUNT(*) FROM ohlcv")
    count2 = c.fetchone()[0]

    assert count1 > 0
    assert count1 == count2, "Duplicate records were added on second run!"

def test_split_continuity(db_conn):
    conn, db_path = db_conn
    c = conn.cursor()
    c.execute("INSERT INTO instruments (code, name, level) VALUES ('SPLIT_STOCK', 'Split Stock', 'stock')")

    # Pre-split
    data = [
        ('SPLIT_STOCK', '2026-01-01', 100, 100, 100, 100, 100, 1000),
        ('SPLIT_STOCK', '2026-01-02', 100, 100, 100, 100, 100, 1000),
    # Post-split (2:1)
        ('SPLIT_STOCK', '2026-01-03', 50, 50, 50, 50, 50, 2000),
        ('SPLIT_STOCK', '2026-01-04', 50, 50, 50, 50, 50, 2000)
    ]
    c.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)

    c.execute("INSERT INTO corporate_actions (instrument_code, ex_date, action_type, ratio) VALUES ('SPLIT_STOCK', '2026-01-03', 'split', 2.0)")
    conn.commit()

    run_adjustments(db_path)

    df = pd.read_sql_query("SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = 'SPLIT_STOCK' ORDER BY date", conn)

    assert df.loc[0, 'adjusted_close'] == 50.0
    assert df.loc[1, 'adjusted_close'] == 50.0
    assert df.loc[2, 'adjusted_close'] == 50.0
    assert df.loc[3, 'adjusted_close'] == 50.0

def test_survivorship(db_conn):
    conn, db_path = db_conn

    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    run_fetcher(db_path, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

    # OLDITSTOCK was inserted as a member of NIFTYIT in the past
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM index_membership WHERE stock_code = 'OLDITSTOCK'")
    assert c.fetchone()[0] == 1, "Delisted symbol OLDITSTOCK missing from history!"

def test_holiday_calendar(db_conn):
    conn, db_path = db_conn

    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    run_fetcher(db_path, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

    c = conn.cursor()
    c.execute("SELECT * FROM holidays WHERE date = '2024-05-01'")
    holiday = c.fetchone()

    assert holiday is not None
    assert holiday[1] == "Maharashtra Day"

def test_universe_coverage(db_conn):
    conn, db_path = db_conn
    from fetch_eod import fetch_all_constituents_from_nse, init_instruments, init_index_membership

    nse_universe = fetch_all_constituents_from_nse()
    init_instruments(conn, nse_universe)
    init_index_membership(conn, nse_universe)

    from fetch_eod import get_universe_from_db
    UNIVERSE = get_universe_from_db(conn)
    import requests, csv, io
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get('https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv', headers=headers)
    reader = csv.DictReader(io.StringIO(r.text))
    expected_symbols = set([row['Symbol'].strip() for row in reader if row.get('Symbol')])

    universe_symbols = set()
    for child in UNIVERSE["children"]:
        for stock in child["stocks"]:
            universe_symbols.add(stock)

    assert len(expected_symbols) == 50, "Expected exactly 50 symbols from NSE"
    assert len(universe_symbols) == 50, "Universe does not have exactly 50 symbols"

    missing_from_universe = expected_symbols - universe_symbols
    assert len(missing_from_universe) == 0, f"Symbols missing from universe: {missing_from_universe}"

    extra_in_universe = universe_symbols - expected_symbols
    assert len(extra_in_universe) == 0, f"Extra symbols in universe not in Nifty 50: {extra_in_universe}"


def test_symbol_resolvability(db_conn):
    conn, db_path = db_conn
    from fetch_eod import get_universe_from_db, fetch_all_constituents_from_nse, init_instruments, init_index_membership
    nse_universe = fetch_all_constituents_from_nse()
    init_instruments(conn, nse_universe)
    init_index_membership(conn, nse_universe)
    UNIVERSE = get_universe_from_db(conn)
    from fetch_eod import fetch_data_for_symbol, INDEX_MAP
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    failed_symbols = []

    # Test market
    market_code = UNIVERSE["code"]
    market_yf = INDEX_MAP.get(market_code)
    if market_yf:
        df = fetch_data_for_symbol(market_code, market_yf, start_str, end_str)
        if df is None or df.empty:
            failed_symbols.append(market_code)

    for child in UNIVERSE["children"]:
        idx_code = child["code"]
        idx_yf = INDEX_MAP.get(idx_code)
        if idx_yf:
            df = fetch_data_for_symbol(idx_code, idx_yf, start_str, end_str)
            if df is None or df.empty:
                failed_symbols.append(idx_code)

        for stock in child["stocks"]:
            yf_sym = f"{stock}.NS"
            df = fetch_data_for_symbol(stock, yf_sym, start_str, end_str)
            if df is None or df.empty:
                failed_symbols.append(stock)

    assert len(failed_symbols) == 0, f"Symbols failed to resolve: {failed_symbols}"
