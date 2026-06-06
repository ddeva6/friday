import subprocess
import sys
from fetch_eod import run_fetcher
from adjust_corporate_actions import run_adjustments

def main():
    print("Running Phase 1 Pipeline...")
    print("1. Fetching EOD data...")
    # Fetching data for a small range to keep it quick for testing
    run_fetcher()

    print("2. Adjusting corporate actions...")
    run_adjustments()

    print("Phase 1 data pipeline completed successfully.")

if __name__ == "__main__":
    main()
