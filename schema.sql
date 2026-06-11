CREATE TABLE IF NOT EXISTS instruments (
    code TEXT PRIMARY KEY,
    name TEXT,
    level TEXT CHECK(level IN ('market', 'index', 'stock')),
    base_val REAL -- Optional baseline, mainly for indices if needed
);

CREATE TABLE IF NOT EXISTS ohlcv (
    instrument_code TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    PRIMARY KEY (instrument_code, date),
    FOREIGN KEY (instrument_code) REFERENCES instruments(code)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    instrument_code TEXT,
    ex_date TEXT,
    action_type TEXT CHECK(action_type IN ('split', 'bonus', 'dividend')),
    ratio REAL, -- e.g., 2.0 for a 2:1 split
    PRIMARY KEY (instrument_code, ex_date, action_type),
    FOREIGN KEY (instrument_code) REFERENCES instruments(code)
);

CREATE TABLE IF NOT EXISTS index_membership (
    index_code TEXT,
    stock_code TEXT,
    start_date TEXT,
    end_date TEXT, -- NULL means currently active
    PRIMARY KEY (index_code, stock_code, start_date),
    FOREIGN KEY (index_code) REFERENCES instruments(code),
    FOREIGN KEY (stock_code) REFERENCES instruments(code)
);

CREATE TABLE IF NOT EXISTS holidays (
    date TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    instrument_code TEXT,
    asof_date TEXT,
    last_price REAL,
    ret REAL,
    cone_width_pct REAL,
    hist_json TEXT, -- JSON array of history
    med_json TEXT, -- JSON array of median forecast
    up_json TEXT,  -- JSON array of upper cone
    lo_json TEXT,  -- JSON array of lower cone
    calibration_json TEXT, -- JSON: {"bias_pct", "cone_scale", "coverage", "n"} applied to this forecast
    PRIMARY KEY (instrument_code, asof_date),
    FOREIGN KEY (instrument_code) REFERENCES instruments(code)
);

-- Tracks predicted-vs-actual outcomes for each forecast, evaluated once
-- `horizon` trading sessions of real OHLCV data are available past asof_date.
CREATE TABLE IF NOT EXISTS forecast_accuracy (
    instrument_code TEXT,
    asof_date TEXT,      -- the date the forecast was made
    target_date TEXT,    -- the actual trading date the forecast targeted
    horizon INTEGER,      -- trading sessions ahead (e.g. 5)
    last_price REAL,      -- price as of asof_date
    predicted_med REAL,
    predicted_up REAL,
    predicted_lo REAL,
    actual_close REAL,
    error_pct REAL,        -- (actual - predicted_med) / predicted_med * 100
    in_cone INTEGER,        -- 1 if predicted_lo <= actual_close <= predicted_up
    direction_correct INTEGER, -- 1 if sign(predicted_med - last_price) == sign(actual_close - last_price)
    PRIMARY KEY (instrument_code, asof_date),
    FOREIGN KEY (instrument_code) REFERENCES instruments(code)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    instrument_code TEXT,
    run_date TEXT,
    total_trials INTEGER,
    model_hits INTEGER,
    model_hit_rate REAL,
    naive_hits INTEGER,
    naive_hit_rate REAL,
    edge REAL,
    config_json TEXT,
    trials_json TEXT,
    PRIMARY KEY (instrument_code, run_date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    instrument_code TEXT PRIMARY KEY,
    mcap_cr REAL,
    pe REAL,
    roe REAL,
    de REAL,
    sales_growth_yoy REAL,
    updated_at TEXT,
    FOREIGN KEY (instrument_code) REFERENCES instruments(code)
);
