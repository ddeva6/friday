import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from forecast_batch import (load_config, calculate_metrics, ensure_monotonic_widening,
                            prices_to_log_returns, log_returns_to_prices,
                            compute_adaptive_lookback, ensemble_with_baselines)

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


def compute_error_metrics(trials):
    if not trials:
        return {"mae": 0, "rmse": 0, "mape": 0}
    errors = []
    pct_errors = []
    for t in trials:
        err = abs(t["forecast_end"] - t["actual_end"])
        errors.append(err)
        if t["actual_end"] != 0:
            pct_errors.append(err / abs(t["actual_end"]) * 100)
    mae = round(np.mean(errors), 2)
    rmse = round(np.sqrt(np.mean(np.array(errors) ** 2)), 2)
    mape = round(np.mean(pct_errors), 2) if pct_errors else 0
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compute_calibration(trials):
    if not trials:
        return {"cone_coverage": 0, "calibration_score": 0}
    within_cone = sum(1 for t in trials
                      if t.get("lower_bound", float('-inf')) <= t.get("actual_end", 0) <= t.get("upper_bound", float('inf')))
    n = len(trials)
    cone_coverage = round(within_cone / n * 100, 1)
    calibration_score = round(100 - abs(cone_coverage - 80), 1)
    return {"cone_coverage": cone_coverage, "calibration_score": calibration_score}


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

            use_log_returns = forecast_config.get("use_log_returns", True)
            adaptive = forecast_config.get("adaptive_lookback", False)

            if adaptive:
                lb = compute_adaptive_lookback(context_prices.tolist(), min_lb=60, max_lb=lookback)
            else:
                lb = lookback

            window = context_prices[-lb:]

            if use_log_returns and len(window) > 3:
                context_data = prices_to_log_returns(window)
                use_lr = len(context_data) >= 3
            else:
                context_data = window.astype(np.float32)
                use_lr = False

            if not use_lr:
                context_data = window.astype(np.float32)

            context = torch.tensor(context_data)
            torch.manual_seed(seed)
            forecast = pipeline.predict(context, prediction_length=horizon, num_samples=sample_count)
            forecast_np = forecast[0].cpu().numpy()

            if use_lr:
                price_samples = np.zeros_like(forecast_np)
                for s in range(forecast_np.shape[0]):
                    price_samples[s] = log_returns_to_prices(actual_start, forecast_np[s])
                forecast_np = price_samples

            med = np.percentile(forecast_np, 50, axis=0)
            up_bound = np.percentile(forecast_np, 90, axis=0)
            lo_bound = np.percentile(forecast_np, 10, axis=0)

            med_ensembled = ensemble_with_baselines(
                med.tolist(), context_prices.tolist(), horizon, chronos_weight=0.6)

            forecast_direction = 1 if med_ensembled[-1] > actual_start else (-1 if med_ensembled[-1] < actual_start else 0)
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
                "forecast_end": round(float(med_ensembled[-1]), 2),
                "upper_bound": round(float(up_bound[-1]), 2),
                "lower_bound": round(float(lo_bound[-1]), 2),
                "actual_dir": int(actual_direction),
                "forecast_dir": int(forecast_direction),
                "hit": int(forecast_direction == actual_direction),
            })

            i += step

        if total > 0:
            hit_rate = round(model_correct / total * 100, 1)
            naive_rate = round(naive_correct / total * 100, 1)
            error_metrics = compute_error_metrics(stock_trials)
            calibration = compute_calibration(stock_trials)
            result = {
                "instrument_code": stock,
                "total_trials": total,
                "model_hits": model_correct,
                "model_hit_rate": hit_rate,
                "naive_hits": naive_correct,
                "naive_hit_rate": naive_rate,
                "edge": round(hit_rate - naive_rate, 1),
                "mae": error_metrics["mae"],
                "rmse": error_metrics["rmse"],
                "mape": error_metrics["mape"],
                "cone_coverage": calibration["cone_coverage"],
                "calibration_score": calibration["calibration_score"],
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

            logging.info(f"{stock}: {hit_rate}% hit ({model_correct}/{total}), "
                         f"MAPE={error_metrics['mape']}%, cone={calibration['cone_coverage']}%, "
                         f"edge {result['edge']}pp")

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
    avg_mape = round(np.mean([r.get('mape', 0) for r in results]), 2) if results else 0
    avg_cone = round(np.mean([r.get('cone_coverage', 0) for r in results]), 1) if results else 0
    return {
        "total_stocks": len(results),
        "total_trials": total_trials,
        "model_hits": model_hits,
        "model_hit_rate": round(model_hits / total_trials * 100, 1) if total_trials else 0,
        "naive_hits": naive_hits,
        "naive_hit_rate": round(naive_hits / total_trials * 100, 1) if total_trials else 0,
        "edge": round((model_hits - naive_hits) / total_trials * 100, 1) if total_trials else 0,
        "avg_mape": avg_mape,
        "avg_cone_coverage": avg_cone,
    }


def print_report(summary, results):
    print("\n" + "=" * 80)
    print("FRIDAY BACKTEST — Directional Hit-Rate & Accuracy Report")
    print("=" * 80)
    print(f"Stocks tested:          {summary['total_stocks']}")
    print(f"Total forecast trials:  {summary['total_trials']}")
    print(f"")
    print(f"Model hit-rate:         {summary['model_hit_rate']}% ({summary['model_hits']}/{summary['total_trials']})")
    print(f"Naive baseline:         {summary['naive_hit_rate']}% ({summary['naive_hits']}/{summary['total_trials']})")
    print(f"Edge over naive:        {summary['edge']}pp")
    print(f"")
    print(f"Avg MAPE:               {summary.get('avg_mape', 0)}%")
    print(f"Avg cone coverage:      {summary.get('avg_cone_coverage', 0)}% (target: 80%)")
    print(f"")

    sorted_results = sorted(results, key=lambda r: r['model_hit_rate'], reverse=True)
    print(f"{'Stock':<15} {'Hit%':>6} {'Edge':>7} {'MAPE%':>7} {'Cone%':>7} {'Trials':>7}")
    print("-" * 52)
    for r in sorted_results:
        print(f"{r['instrument_code']:<15} {r['model_hit_rate']:>5.1f}% {r['edge']:>6.1f}pp "
              f"{r.get('mape', 0):>6.1f}% {r.get('cone_coverage', 0):>6.1f}% {r['total_trials']:>7}")

    print("-" * 52)
    print(f"\nBaseline: naive = 'no change' (predicts 0% move).")
    print(f"Cone coverage: % of actuals within P10-P90 band (target ~80%).")
    print(f"These results are NOT tuned — report honestly per spec.")
    print("=" * 80)


if __name__ == "__main__":
    run_backtest()
