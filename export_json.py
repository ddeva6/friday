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

        # Forecast contract shape with empty forecast cones
        data = {
            "code": inst,
            "last": last,
            "hist": hist,
            "med": [],
            "up": [],
            "lo": [],
            "ret": 0.0,
            "cone_width_pct": 0.0,
            "asof": df.iloc[-1]['date']
        }

        with open(os.path.join(output_dir, f"{inst}.json"), "w") as f:
            json.dump(data, f)

    conn.close()
    print(f"Exported to {output_dir}")

if __name__ == "__main__":
    export_data()
