# FRIDAY — Kronos Forecast Terminal

FRIDAY is a personal NSE equity research tool that generates hierarchical 5-day probabilistic forecasts for the Nifty 50, its sector indices, and constituent stocks. Powered by the Chronos time-series foundation model, it provides a drill-through dashboard and a risk-adjusted screener for research and visualization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddeva6/friday/blob/main/friday_colab.ipynb)

## Setup
1. **Install dependencies:**
   ```bash
   pip install yfinance pandas pyyaml requests numpy jsonschema torch chronos-forecasting
   ```
2. **Run the pipeline:**
   ```bash
   python runner.py
   ```
   *This fetches data, generates forecasts (requires GPU), and exports JSON.*

## Architecture
- **Phase 1 (Data):** Fetches EOD OHLCV and fundamentals from `yfinance`. Stores in SQLite with corporate-action adjustments.
- **Phase 2 (Forecast):** Runs the `chronos-t5-base` model to generate 5-day uncertainty cones.
- **Phase 3 (UI):** Exports static JSON files to `data/` and serves the HTML/JS dashboard.

## Performance (T4 GPU)
- **Model:** `chronos-t5-base` (~200M params)
- **VRAM:** ~0.8 GB
- **Time:** ~1-2 min for 50 instruments

## Data Sources
- **OHLCV & Fundamentals:** Yahoo Finance (`yfinance`)
- **Index Constituents:** NSE India

## Disclaimer
*FRIDAY is a research and visualization tool. Forecasts are probabilistic and shown as uncertainty cones. Nothing it outputs is investment advice. Not investment advice.*
