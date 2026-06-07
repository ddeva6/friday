import sqlite3
import pandas as pd
import json
import os
import yaml

def get_db_connection(db_path="test.db"):
    conn = sqlite3.connect(db_path)
    import os
    if os.path.exists("schema.sql"):
        with open("schema.sql", "r") as f:
            conn.executescript(f.read())
    return conn


def export_data(output_dir="data"):
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        db_path = config.get("database", {}).get("path", "test.db")
    except Exception:
        db_path = "test.db"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = get_db_connection(db_path)
    c = conn.cursor()


    # Dump hierarchy to index.json
    c.execute("SELECT code, name FROM instruments WHERE level = 'market'")
    market = c.fetchone()

    if not market:
        print("No market found in DB")
        return

    universe = {"code": market[0], "name": market[1], "children": []}

    c.execute("SELECT code, name FROM instruments WHERE level = 'index'")
    indices = c.fetchall()

    for idx_code, idx_name in indices:
        c.execute("SELECT stock_code FROM index_membership WHERE index_code = ? AND end_date IS NULL ORDER BY stock_code", (idx_code,))
        stocks = [r[0] for r in c.fetchall()]
        if stocks:
            universe["children"].append({
                "code": idx_code,
                "name": idx_name,
                "stocks": stocks
            })

    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(universe, f, indent=2)

    # Dump instrument data
    c.execute("SELECT DISTINCT instrument_code FROM ohlcv")
    instruments = [r[0] for r in c.fetchall()]

    for inst in instruments:
        df = pd.read_sql_query("SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? ORDER BY date ASC", conn, params=(inst,))
        if df.empty:
            continue

        hist = df['adjusted_close'].tolist()
        last = hist[-1]

        # Get forecast from DB
        c.execute("SELECT ret, cone_width_pct, med_json, up_json, lo_json FROM forecasts WHERE instrument_code = ? ORDER BY asof_date DESC LIMIT 1", (inst,))
        f_row = c.fetchone()

        med = []
        up = []
        lo = []
        ret = 0.0
        cone_width_pct = 0.0

        if f_row:
            ret, cone_width_pct, med_json, up_json, lo_json = f_row
            if med_json: med = json.loads(med_json)
            if up_json: up = json.loads(up_json)
            if lo_json: lo = json.loads(lo_json)

        # Get 52w high/low
        # Approx 252 trading days in a year
        df_52w = df.tail(252)
        hi_52w = float(df_52w['adjusted_close'].max())
        lo_52w = float(df_52w['adjusted_close'].min())

        # Forecast contract shape with populated forecast cones
        data = {
            "code": inst,
            "last": last,
            "hist": hist,
            "med": med,
            "up": up,
            "lo": lo,
            "ret": ret,
            "cone_width_pct": cone_width_pct,
            "asof": df.iloc[-1]['date'],
            "fundamentals": {
                "mcap_cr": "N/A",
                "pe": "N/A",
                "roe": "N/A",
                "de": "N/A",
                "sales_growth_yoy": "N/A",
                "hi_52w": hi_52w,
                "lo_52w": lo_52w
            },
            "status": [
                "Liquidity OK",
                "Data as of " + df.iloc[-1]['date']
            ]
        }

        with open(os.path.join(output_dir, f"{inst}.json"), "w") as f:
            json.dump(data, f, indent=2)

    conn.close()
    print(f"Exported to {output_dir}")

if __name__ == "__main__":
    export_data()
