import subprocess
import sys
import logging
import yaml
from fetch_eod import run_fetcher
from adjust_corporate_actions import run_adjustments
from forecast_batch import run_forecast_batch, load_config
from export_json import export_data

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def main():
    config = load_config()
    logging.info(f"Starting FRIDAY pipeline with config: {config}")

    print("Running Phase 1 Pipeline...")
    print("1. Fetching EOD data...")
    # Fetching data for a small range to keep it quick for testing
    run_fetcher()

    print("2. Adjusting corporate actions...")
    run_adjustments()

    print("Phase 1 data pipeline completed successfully.")

    print("Running Phase 2 Pipeline...")
    print("1. Generating forecasts...")
    run_forecast_batch()

    print("2. Exporting JSON...")
    export_data()

    print("Pipeline completed successfully.")

def run_backtest_cmd():
    from backtest import run_backtest
    config = load_config()
    logging.info(f"Starting FRIDAY backtest with config: {config}")
    run_backtest()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest_cmd()
    else:
        main()
