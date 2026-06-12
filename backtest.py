import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from forecast_batch import load_config, calculate_metrics, ensure_monotonic_widening

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


def run_backtest(db_path=None, start_date=None, end_date=None):
    """
    Walk-forward backtest: at each historical date, use the lookback window
    to forecast 5 sessions ahead, then compare predicted direction vs actual.
    Measures directional hit-rate and compares against a naive baseline.
    """
    import torch
    from chronos import ChronosPipeline

    config = load_config()
    db_path = db_path or config.get("database", {}).get("path", "friday.db")
    forecast_config = config.get("forecast", {})
    backtest_config = config.get("backtest", {})

    lookback = forecast_config.get("lookback", 180)
    horizon = forecast_config.get("horizon", 1)
    sample_count = forecast_config.get("sample_count", 30)
    model_name = forecast_config.get("model", "amazon/chronos-t5-base")
    seed = forecast_config.get("seed", 42)
    step = backtest_config.get("step", 5)

    logging.info("FRIDAY backtest starting")
    logging.info(f"Model: {model_name}, lookback={lookback}, horizon={horizon}, step={step}")

    pipeline = ChronosPipeline.from_pretrained(
        model_name, device_map="auto", dtype=torch.bfloat16,
    )

    conn = sqlite3.connect(db_path)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    c = conn.cursor()

    c.execute("SELECT code FROM instruments WHERE level = 'stock'")
    stocks = [r[0] for r in c.fetchall()]

    if not stocks:
        logging.warning("No stocks in DB. Run the pipeline first.")
        conn.close()
        return None

    all_results = []

    for stock in stocks:
        df = pd.read_sql_query(
            "SELECT date, adjusted_close FROM ohlcv WHERE instrument_code = ? ORDER BY date ASC",
            conn, params=(stock,)
        )
        if len(df) < lookback + horizon + step:
            continue

        prices = df['adjusted_close'].values
        dates = df['date'].values

        model_correct = 0
        naive_correct = 0
        total = 0
        stock_trials = []

        i = lookback
        while i + horizon <= len(prices):
            context_prices = prices[:i]
            actual_start = prices[i - 1]
            actual_end = prices[i + horizon - 1]
            actual_direction = 1 if actual_end > actual_start else (-1 if actual_end < actual_start else 0)

            context = torch.tensor(context_prices[-lookback:].astype(np.float32))
            torch.manual_seed(seed)
            forecast = pipeline.predict(context, prediction_length=horizon, num_samples=sample_count)
            forecast_np = forecast[0].cpu().numpy()
            med = np.percentile(forecast_np, 50, axis=0)

            forecast_direction = 1 if med[-1] > actual_start else (-1 if med[-1] < actual_start else 0)
            naive_direction = 0

            if forecast_direction == actual_direction:
                model_correct += 1
            if naive_direction == actual_direction:
                naive_correct += 1
            total += 1

            stock_trials.append({
                "asof_date": dates[i - 1],
                "actual_start": round(float(actual_start), 2),
                "actual_end": round(float(actual_end), 2),
                "forecast_end": round(float(med[-1]), 2),
                "actual_dir": int(actual_direction),
                "forecast_dir": int(forecast_direction),
                "hit": int(forecast_direction == actual_direction),
            })

            i += step

        if total > 0:
            hit_rate = round(model_correct / total * 100, 1)
            naive_rate = round(naive_correct / total * 100, 1)
            result = {
                "instrument_code": stock,
                "total_trials": total,
                "model_hits": model_correct,
                "model_hit_rate": hit_rate,
                "naive_hits": naive_correct,
                "naive_hit_rate": naive_rate,
                "edge": round(hit_rate - naive_rate, 1),
            }
            all_results.append(result)

            c.execute("""
                INSERT OR REPLACE INTO backtest_results
                (instrument_code, run_date, total_trials, model_hits, model_hit_rate,
                 naive_hits, naive_hit_rate, edge, config_json, trials_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stock, datetime.now().strftime('%Y-%m-%d'), total,
                model_correct, hit_rate, naive_correct, naive_rate,
                round(hit_rate - naive_rate, 1),
                json.dumps({"lookback": lookback, "horizon": horizon,
                            "sample_count": sample_count, "model": model_name, "seed": seed, "step": step}),
                json.dumps(stock_trials),
            ))

            logging.info(f"{stock}: {hit_rate}% hit-rate ({model_correct}/{total}), naive {naive_rate}%, edge {result['edge']}pp")

    conn.commit()

    if all_results:
        summary = summarize_backtest(all_results)
        c.execute("""
            INSERT OR REPLACE INTO backtest_results
            (instrument_code, run_date, total_trials, model_hits, model_hit_rate,
             naive_hits, naive_hit_rate, edge, config_json, trials_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            '_AGGREGATE', datetime.now().strftime('%Y-%m-%d'),
            summary['total_trials'], summary['model_hits'], summary['model_hit_rate'],
            summary['naive_hits'], summary['naive_hit_rate'], summary['edge'],
            json.dumps({"lookback": lookback, "horizon": horizon,
                        "sample_count": sample_count, "model": model_name, "seed": seed, "step": step}),
            json.dumps([]),
        ))
        conn.commit()
        print_report(summary, all_results)
    else:
        logging.warning("No stocks had enough data for backtesting.")

    conn.close()
    return all_results


def summarize_backtest(results):
    total_trials = sum(r['total_trials'] for r in results)
    model_hits = sum(r['model_hits'] for r in results)
    naive_hits = sum(r['naive_hits'] for r in results)
    return {
        "total_stocks": len(results),
        "total_trials": total_trials,
        "model_hits": model_hits,
        "model_hit_rate": round(model_hits / total_trials * 100, 1) if total_trials else 0,
        "naive_hits": naive_hits,
        "naive_hit_rate": round(naive_hits / total_trials * 100, 1) if total_trials else 0,
        "edge": round((model_hits - naive_hits) / total_trials * 100, 1) if total_trials else 0,
    }


def print_report(summary, results):
    print("\n" + "=" * 60)
    print("FRIDAY BACKTEST — Directional Hit-Rate Report")
    print("=" * 60)
    print(f"Stocks tested:        {summary['total_stocks']}")
    print(f"Total forecast trials: {summary['total_trials']}")
    print(f"")
    print(f"Model hit-rate:        {summary['model_hit_rate']}% ({summary['model_hits']}/{summary['total_trials']})")
    print(f"Naive baseline:        {summary['naive_hit_rate']}% ({summary['naive_hits']}/{summary['total_trials']})")
    print(f"Edge over naive:       {summary['edge']}pp")
    print(f"")

    sorted_results = sorted(results, key=lambda r: r['model_hit_rate'], reverse=True)
    print(f"{'Stock':<15} {'Hit-Rate':>10} {'Naive':>8} {'Edge':>8} {'Trials':>8}")
    print("-" * 52)
    for r in sorted_results:
        print(f"{r['instrument_code']:<15} {r['model_hit_rate']:>9}% {r['naive_hit_rate']:>7}% {r['edge']:>7}pp {r['total_trials']:>7}")

    print("-" * 52)
    print(f"\nBaseline: naive = 'no change' (predicts 0% move).")
    print(f"A hit-rate near 50% with positive edge suggests weak signal.")
    print(f"These results are NOT tuned — report honestly per spec.")
    print("=" * 60)


if __name__ == "__main__":
    run_backtest()
