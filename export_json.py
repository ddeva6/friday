import sqlite3
import pandas as pd
import json
import os
import yaml
import yfinance as yf
from datetime import datetime

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
        # Fetch fundamentals from yfinance if not in DB or stale
        c.execute("SELECT mcap_cr, pe, roe, de, sales_growth_yoy, updated_at FROM fundamentals WHERE instrument_code = ?", (inst,))
        f_data = c.fetchone()

        # Simple caching: re-fetch if not found or updated > 30 days ago
        needs_fetch = True
        if f_data:
            updated_at = datetime.strptime(f_data[5], '%Y-%m-%d')
            if (datetime.now() - updated_at).days < 30:
                needs_fetch = False

        is_index = inst.startswith("NIFTY") or inst == "BANKNIFTY"
        if needs_fetch and not is_index:
            try:
                ticker = yf.Ticker(inst + ".NS")
                info = ticker.info
                mcap_cr = info.get('marketCap', 0) / 1e7
                pe = info.get('trailingPE')
                roe = info.get('returnOnEquity')
                if roe: roe *= 100
                de = info.get('debtToEquity')
                if de: de /= 100
                sales_growth_yoy = info.get('revenueGrowth')
                if sales_growth_yoy: sales_growth_yoy *= 100

                updated_at_str = datetime.now().strftime('%Y-%m-%d')
                c.execute("""
                    INSERT INTO fundamentals (instrument_code, mcap_cr, pe, roe, de, sales_growth_yoy, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(instrument_code) DO UPDATE SET
                        mcap_cr=excluded.mcap_cr,
                        pe=excluded.pe,
                        roe=excluded.roe,
                        de=excluded.de,
                        sales_growth_yoy=excluded.sales_growth_yoy,
                        updated_at=excluded.updated_at
                """, (inst, mcap_cr, pe, roe, de, sales_growth_yoy, updated_at_str))
                conn.commit()
                f_data = (mcap_cr, pe, roe, de, sales_growth_yoy, updated_at_str)
            except Exception as e:
                print(f"Failed to fetch fundamentals for {inst}: {e}")

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

        # Real data-status flags
        status_flags = []

        # 1. Low liquidity: avg volume last 20 sessions < 100k
        # We need volume from ohlcv table
        df_vol = pd.read_sql_query("SELECT volume FROM ohlcv WHERE instrument_code = ? ORDER BY date DESC LIMIT 20", conn, params=(inst,))
        if not df_vol.empty and df_vol['volume'].mean() < 100000:
            status_flags.append("Low liquidity")

        # 2. Data gap: last date > 3 calendar days ago
        last_date_str = df.iloc[-1]['date']
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
        if (datetime.now() - last_date).days > 3:
            status_flags.append("Stale data")

        # 3. Short history: < 60 data points
        if len(df) < 60:
            status_flags.append("Short history — forecast less reliable")

        if not status_flags:
            status_flags.append("Data OK")

        status_flags.append("Data as of " + last_date_str)

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
            "asof": last_date_str,
            "fundamentals": {
                "mcap_cr": f_data[0] if f_data and f_data[0] is not None and f_data[0] > 0 else "N/A",
                "pe": f_data[1] if f_data and f_data[1] is not None else "N/A",
                "roe": f_data[2] if f_data and f_data[2] is not None else "N/A",
                "de": f_data[3] if f_data and f_data[3] is not None else "N/A",
                "sales_growth_yoy": f_data[4] if f_data and f_data[4] is not None else "N/A",
                "hi_52w": hi_52w,
                "lo_52w": lo_52w
            },
            "status": status_flags
        }

        with open(os.path.join(output_dir, f"{inst}.json"), "w") as f:
            json.dump(data, f, indent=2)

    conn.close()
    print(f"Exported to {output_dir}")

if __name__ == "__main__":
    export_data()
