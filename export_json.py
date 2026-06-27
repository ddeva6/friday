import sqlite3
import pandas as pd
import json
import os
import yaml
import yfinance as yf
from datetime import datetime, timezone

from forecast_batch import calculate_metrics

GOLD_ETF_SYMBOLS = {"GOLDBEES", "HDFCGOLD", "SETFGOLD", "AXISGOLD", "GOLD1", "IVZINGOLD", "QGOLDHALF"}

# Status flags whose presence must force a CAUTION verdict regardless of how
# favorable every other signal looks — the data itself can't be trusted.
DATA_QUALITY_PREFIXES = ("Stale data", "Short history")


def calculate_rsi(prices, period=14):
    """Wilder's RSI from a price series. Returns (rsi_value, label); rsi_value
    is None when there isn't enough history or the price hasn't moved at all
    (flat series), avoiding a NaN/divide-by-zero result."""
    if len(prices) < period + 1:
        return None, "N/A"

    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None, "N/A"
    if avg_loss == 0:
        if avg_gain == 0:
            return None, "N/A"
        return 100.0, "OVERBOUGHT"

    rs = avg_gain / avg_loss
    rsi = round(float(100 - (100 / (1 + rs))), 1)
    if rsi >= 70:
        label = "OVERBOUGHT"
    elif rsi <= 30:
        label = "OVERSOLD"
    else:
        label = "NEUTRAL"
    return rsi, label


def compute_verdict(conviction, risk_reward, trend_label, ret, vol_confirmed, status_flags):
    """Composite, plain-English decision verdict combining conviction, trend
    alignment, risk:reward and volume. Data-quality issues are checked first
    and always win — no combination of favorable signals can override them."""
    data_quality_reason = next(
        (f for f in status_flags if f.startswith(DATA_QUALITY_PREFIXES)), None
    )
    if data_quality_reason:
        return {"label": "CAUTION", "reasons": [data_quality_reason]}

    high_conviction = conviction >= 70
    low_conviction = conviction < 40
    conv_tier = "HIGH" if high_conviction else ("LOW" if low_conviction else "MEDIUM")

    trend_aligned = (trend_label == "BULLISH" and ret > 0) or (trend_label == "BEARISH" and ret < 0)
    trend_conflicts = (trend_label == "BULLISH" and ret < 0) or (trend_label == "BEARISH" and ret > 0)
    good_rr = risk_reward >= 1

    reasons = [f"Conviction {conviction}/100 ({conv_tier})"]
    if trend_aligned:
        reasons.append(f"Trend ({trend_label}) agrees with the forecast")
    elif trend_conflicts:
        reasons.append(f"Trend ({trend_label}) conflicts with the forecast")
    reasons.append(f"R:R {'favorable' if good_rr else 'unfavorable'} (1:{risk_reward:.2f})" if risk_reward else "R:R not available")
    reasons.append("Volume confirms the move" if vol_confirmed else "Volume does not confirm the move")

    if high_conviction and not trend_conflicts and good_rr:
        label = "FAVORABLE"
    elif low_conviction or trend_conflicts or not good_rr:
        label = "CAUTION"
    else:
        label = "NEUTRAL"

    return {"label": label, "reasons": reasons}


def build_synthetic_index(code, name, stock_codes, cache):
    """Build an equal-weighted synthetic index from constituent stock data,
    for index-level instruments (e.g. NIFTYFINSERVICE, GOLDETF) that have no
    OHLCV/forecast of their own. Each constituent's history and forecast are
    rebased to a common scale before averaging."""
    constituents = [cache[s] for s in stock_codes if s in cache and cache[s]['hist']]
    if not constituents:
        return None

    min_len = min(len(c['hist']) for c in constituents)
    if min_len == 0:
        return None

    rebased_hists = [
        [v / c['hist'][-min_len] * 100 for v in c['hist'][-min_len:]]
        for c in constituents if c['hist'][-min_len] != 0
    ]
    if not rebased_hists:
        return None

    hist = [round(sum(vals) / len(vals), 2) for vals in zip(*rebased_hists)]
    last = hist[-1]

    rebased_meds, rebased_ups, rebased_los = [], [], []
    for c in constituents:
        if c['med'] and c['up'] and c['lo'] and c['last']:
            rebased_meds.append([last * (v / c['last']) for v in c['med']])
            rebased_ups.append([last * (v / c['last']) for v in c['up']])
            rebased_los.append([last * (v / c['last']) for v in c['lo']])

    if rebased_meds:
        med = [round(sum(vals) / len(vals), 2) for vals in zip(*rebased_meds)]
        up = [round(sum(vals) / len(vals), 2) for vals in zip(*rebased_ups)]
        lo = [round(sum(vals) / len(vals), 2) for vals in zip(*rebased_los)]
    else:
        med, up, lo = [], [], []

    ret, cone_width_pct = calculate_metrics(last, med, up, lo)
    asof = max(c['asof'] for c in constituents)

    return {
        "code": code,
        "name": name,
        "last": last,
        "hist": hist,
        "med": med,
        "up": up,
        "lo": lo,
        "ret": ret,
        "cone_width_pct": cone_width_pct,
        "asof": asof,
        "fundamentals": {
            "mcap_cr": "N/A", "pe": "N/A", "roe": "N/A", "de": "N/A",
            "sales_growth_yoy": "N/A", "hi_52w": max(hist), "lo_52w": min(hist)
        },
        "status": [
            f"Synthetic index — equal-weighted average of {len(constituents)} constituent(s)",
            "Data as of " + asof
        ],
        "calibration": {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": 0},
        "accuracy": {"history": [], "n": 0, "mae_pct": None, "directional_accuracy_pct": None, "calibration_pct": None}
    }

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

    universe["generated_at"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(universe, f, indent=2)

    # Dump instrument data
    c.execute("SELECT DISTINCT instrument_code FROM ohlcv")
    instruments = [r[0] for r in c.fetchall()]

    inst_data_cache = {}

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

        is_index = inst.startswith("NIFTY") or inst == "BANKNIFTY" or inst in GOLD_ETF_SYMBOLS
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

        # Get instrument name
        c.execute("SELECT name FROM instruments WHERE code = ?", (inst,))
        name_row = c.fetchone()
        inst_name = name_row[0] if name_row else inst

        df = pd.read_sql_query("SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? ORDER BY date ASC", conn, params=(inst,))
        df = df.dropna(subset=['adjusted_close'])
        if df.empty:
            continue

        hist = [round(x, 2) for x in df['adjusted_close'].tolist()]
        last = hist[-1]

        # Get forecast from DB
        c.execute("SELECT ret, cone_width_pct, med_json, up_json, lo_json, calibration_json FROM forecasts WHERE instrument_code = ? ORDER BY asof_date DESC LIMIT 1", (inst,))
        f_row = c.fetchone()

        med = []
        up = []
        lo = []
        ret = 0.0
        cone_width_pct = 0.0
        calibration = {"bias_pct": 0.0, "coverage": None, "cone_scale": 1.0, "n": 0}

        if f_row:
            ret, cone_width_pct, med_json, up_json, lo_json, calibration_json = f_row
            if med_json: med = json.loads(med_json)
            if up_json: up = json.loads(up_json)
            if lo_json: lo = json.loads(lo_json)
            if calibration_json: calibration = json.loads(calibration_json)

        # Get accuracy history: predicted-vs-actual outcomes for past forecasts
        c.execute("""
            SELECT asof_date, target_date, predicted_med, actual_close, error_pct, in_cone, direction_correct
            FROM forecast_accuracy WHERE instrument_code = ? ORDER BY asof_date DESC LIMIT 20
        """, (inst,))
        acc_rows = c.fetchall()

        accuracy_history = [{
            "asof_date": r[0],
            "target_date": r[1],
            "predicted_med": r[2],
            "actual_close": r[3],
            "error_pct": r[4],
            "in_cone": bool(r[5]),
            "direction_correct": bool(r[6])
        } for r in acc_rows]

        if acc_rows:
            mae_pct = sum(abs(r[4]) for r in acc_rows) / len(acc_rows)
            directional_accuracy_pct = sum(r[6] for r in acc_rows) / len(acc_rows) * 100
            calibration_pct = sum(r[5] for r in acc_rows) / len(acc_rows) * 100
        else:
            mae_pct = None
            directional_accuracy_pct = None
            calibration_pct = None

        accuracy = {
            "history": accuracy_history,
            "n": len(acc_rows),
            "mae_pct": round(mae_pct, 2) if mae_pct is not None else None,
            "directional_accuracy_pct": round(directional_accuracy_pct, 1) if directional_accuracy_pct is not None else None,
            "calibration_pct": round(calibration_pct, 1) if calibration_pct is not None else None
        }

        # Get 52w high/low
        # Approx 252 trading days in a year
        df_52w = df.tail(252)
        hi_52w = float(df_52w['adjusted_close'].max())
        lo_52w = float(df_52w['adjusted_close'].min())

        # Real data-status flags
        status_flags = []

        # 1. Low liquidity: avg volume last 20 sessions < 100k
        # We need volume from ohlcv table
        is_index = inst.startswith("NIFTY") or inst == "BANKNIFTY"
        df_vol = pd.read_sql_query("SELECT volume FROM ohlcv WHERE instrument_code = ? ORDER BY date DESC LIMIT 20", conn, params=(inst,))
        if not is_index and not df_vol.empty and df_vol['volume'].mean() < 100000:
            status_flags.append("Low liquidity")

        # 2. Data gap: last date > 3 calendar days ago. Use business days or check against the holidays table instead.
        last_date_str = df.iloc[-1]['date']
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()

        # Calculate business days diff taking holidays into account
        import numpy as np
        c.execute("SELECT date FROM holidays")
        holidays_list = np.array([r[0] for r in c.fetchall()], dtype='datetime64')
        bus_days_diff = np.busday_count(last_date, datetime.now().date(), holidays=holidays_list)

        if bus_days_diff > 3:
            status_flags.append("Stale data")

        # 3. Short history: < 60 data points
        if len(df) < 60:
            status_flags.append("Short history — forecast less reliable")

        if not status_flags:
            status_flags.append("Data OK")

        status_flags.append("Data as of " + last_date_str)

        # Compute trading signals from forecast
        entry_price = last
        exit_price = round(med[-1], 2) if med else last
        stop_loss = round(lo[-1], 2) if lo else last
        risk = round(entry_price - stop_loss, 2)
        reward = round(exit_price - entry_price, 2)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        # Volume confirmation
        import numpy as np
        df_vol_full = pd.read_sql_query(
            "SELECT volume FROM ohlcv WHERE instrument_code = ? ORDER BY date DESC LIMIT 21",
            conn, params=(inst,))
        df_vol_full = df_vol_full.dropna(subset=['volume'])
        if not df_vol_full.empty and len(df_vol_full) >= 2:
            latest_vol = int(df_vol_full.iloc[0]['volume'])
            avg_vol_20 = int(df_vol_full.iloc[1:21]['volume'].mean())
            vol_ratio = round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0
            vol_confirmed = vol_ratio >= 1.0
        else:
            latest_vol = 0
            avg_vol_20 = 0
            vol_ratio = 0
            vol_confirmed = False

        volume = {
            "latest": latest_vol,
            "avg_20d": avg_vol_20,
            "ratio": vol_ratio,
            "confirmed": vol_confirmed,
        }

        # Trend filter — EMA 50 & EMA 200
        prices_arr = df['adjusted_close']
        ema_50 = round(float(prices_arr.ewm(span=50, adjust=False).mean().iloc[-1]), 2) if len(prices_arr) >= 50 else None
        ema_200 = round(float(prices_arr.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if len(prices_arr) >= 200 else None

        if ema_50 and ema_200:
            if last > ema_50 > ema_200:
                trend_label = "BULLISH"
            elif last < ema_50 < ema_200:
                trend_label = "BEARISH"
            elif ema_50 > ema_200:
                trend_label = "WEAKENING"
            else:
                trend_label = "RECOVERING"
        elif ema_50:
            trend_label = "ABOVE EMA50" if last > ema_50 else "BELOW EMA50"
        else:
            trend_label = "N/A"

        trend = {
            "ema_50": ema_50,
            "ema_200": ema_200,
            "label": trend_label,
        }

        # Momentum filter — RSI(14)
        rsi_14, rsi_label = calculate_rsi(prices_arr)
        momentum = {
            "rsi_14": rsi_14,
            "label": rsi_label,
        }

        # Conviction score (0-100)
        score = 50
        if risk_reward >= 2:
            score += 15
        elif risk_reward >= 1:
            score += 5
        else:
            score -= 10

        if acc_rows:
            dir_acc = sum(r[6] for r in acc_rows) / len(acc_rows) * 100
            if dir_acc >= 70:
                score += 15
            elif dir_acc >= 55:
                score += 5
            else:
                score -= 10
            in_cone_pct = sum(r[5] for r in acc_rows) / len(acc_rows) * 100
            if in_cone_pct >= 70:
                score += 10
            elif in_cone_pct < 50:
                score -= 5

        if vol_confirmed:
            score += 5

        if trend_label == "BULLISH" and ret > 0:
            score += 10
        elif trend_label == "BEARISH" and ret < 0:
            score += 10
        elif trend_label in ("BULLISH", "BEARISH"):
            score -= 10

        if abs(ret) > 2:
            score += 5

        conviction = max(0, min(100, score))

        verdict = compute_verdict(conviction, risk_reward, trend_label, ret, vol_confirmed, status_flags)

        trading = {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "exit": exit_price,
            "risk": risk,
            "reward": reward,
            "risk_reward": risk_reward,
            "conviction": conviction,
            "volume": volume,
            "trend": trend,
            "momentum": momentum,
            "verdict": verdict,
        }

        # Forecast contract shape with populated forecast cones
        data = {
            "code": inst,
            "name": inst_name,
            "last": last,
            "hist": hist,
            "med": med,
            "up": up,
            "lo": lo,
            "ret": ret,
            "cone_width_pct": cone_width_pct,
            "asof": last_date_str,
            "trading": trading,
            "fundamentals": {
                "mcap_cr": f_data[0] if f_data and f_data[0] is not None and f_data[0] > 0 else "N/A",
                "pe": f_data[1] if f_data and f_data[1] is not None else "N/A",
                "roe": f_data[2] if f_data and f_data[2] is not None else "N/A",
                "de": f_data[3] if f_data and f_data[3] is not None else "N/A",
                "sales_growth_yoy": f_data[4] if f_data and f_data[4] is not None else "N/A",
                "hi_52w": hi_52w,
                "lo_52w": lo_52w
            },
            "status": status_flags,
            "calibration": calibration,
            "accuracy": accuracy
        }

        inst_data_cache[inst] = data

        with open(os.path.join(output_dir, f"{inst}.json"), "w") as f:
            json.dump(data, f, indent=2)

    # Index-level instruments with no OHLCV/forecast of their own (e.g.
    # NIFTYFINSERVICE, GOLDETF) get a synthetic equal-weighted index built
    # from their constituents, so they render alongside other sectors
    # instead of showing blank/zero cards.
    for child in universe["children"]:
        if child["code"] in inst_data_cache:
            continue
        synthetic = build_synthetic_index(child["code"], child["name"], child["stocks"], inst_data_cache)
        if synthetic:
            with open(os.path.join(output_dir, f"{child['code']}.json"), "w") as f:
                json.dump(synthetic, f, indent=2)

    conn.close()
    print(f"Exported to {output_dir}")

if __name__ == "__main__":
    export_data()
