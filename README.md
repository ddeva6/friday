# FRIDAY — Kronos Forecast Terminal

[![CI](https://github.com/ddeva6/friday/actions/workflows/ci.yml/badge.svg)](https://github.com/ddeva6/friday/actions/workflows/ci.yml)
[![Daily Refresh](https://github.com/ddeva6/friday/actions/workflows/refresh.yml/badge.svg)](https://github.com/ddeva6/friday/actions/workflows/refresh.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddeva6/friday/blob/main/friday_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub](https://img.shields.io/github/stars/ddeva6/friday?style=social)](https://github.com/ddeva6/friday)

**FRIDAY** generates 5-day probabilistic forecasts for 50+ NSE stocks using the [Kronos](https://github.com/amazon-science/chronos-forecasting) (Chronos) time-series foundation model. Zero-shot. No training. Just inference.

**Live Dashboard:** [https://ddeva6.github.io/friday/](https://ddeva6.github.io/friday/) (auto-refreshes daily after market close)

<p align="center">
  <em>Hierarchical drill-down: Market → Sector → Stock, with uncertainty cones and a risk-adjusted screener</em>
</p>

---

## Why FRIDAY?

Most retail forecast tools give you a single number. FRIDAY gives you an **uncertainty cone** — the P10/P50/P90 range of possible outcomes. You see not just *where* the model thinks the price is going, but *how confident* it is.

- **Zero-shot forecasting** — no training data, no overfitting. Kronos generalizes from pre-training on 27B time-series observations
- **Hierarchical view** — drill from NIFTY 50 → sector indices → individual stocks
- **Gold ETFs** — track domestic gold prices via NSE-listed Gold ETFs (GOLDBEES, HDFCGOLD, SETFGOLD, AXISGOLD, GOLD1, IVZINGOLD, QGOLDHALF) with the same forecast cones
- **Risk-adjusted screener** — rank stocks by return-per-unit-uncertainty, not just raw return
- **Fully automated** — GitHub Actions fetches data and regenerates forecasts daily at 4:45 PM IST
- **Single HTML file** — the entire dashboard is one self-contained file. No build tools. No framework.

## Quick Start

### Option 1: Just view the dashboard
Visit [https://ddeva6.github.io/friday/](https://ddeva6.github.io/friday/). Data refreshes automatically every weekday.

### Option 2: One-click Colab (GPU)
1. Open the [Colab notebook](https://colab.research.google.com/github/ddeva6/friday/blob/main/friday_colab.ipynb)
2. Set runtime to **T4 GPU**
3. **Run All** — forecasts generate in ~2 minutes

### Option 3: Run locally
```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt  # for forecasts (GPU/CPU)

python runner.py            # fetch → forecast → export
python runner.py backtest   # walk-forward directional backtest
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  NSE India   │     │   yfinance   │     │   Kronos    │     │ GitHub Pages │
│  (universe)  │────▶│  (OHLCV +    │────▶│  (forecast)  │────▶│ (dashboard)  │
│              │     │  fundamentals)│     │              │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
       Phase 1              Phase 1           Phase 2             Phase 3
```

| Phase | What it does | Key file |
|-------|-------------|----------|
| **1 — Data** | Fetches EOD OHLCV + fundamentals from yfinance, index constituents from NSE | `fetch_eod.py` |
| **2 — Forecast** | Runs Kronos-T5 to generate 5-day uncertainty cones (P10/P50/P90) | `forecast_batch.py` |
| **3 — UI** | Exports SQLite → static JSON, serves single-file HTML dashboard | `export_json.py` |

### Models

| Model | Params | Runtime | Use case |
|-------|--------|---------|----------|
| `kronos-t5-base` | ~200M | ~2 min (T4 GPU) | Colab / GPU servers |
| `kronos-t5-tiny` | ~8M | ~1 min (CPU) | Daily GitHub Actions refresh |

## Backtesting

```bash
python runner.py backtest
```

Walk-forward directional backtest: at each step, forecast 5 sessions ahead using only past data, then check if the predicted direction matched reality. Results compared against a naive "no change" baseline.

**This is a sanity check, not a trading signal.** A hit-rate near 50% is expected for a zero-shot model.

## Project Structure

```
friday/
├── fetch_eod.py                  # Phase 1 — data fetching
├── forecast_batch.py             # Phase 2 — Kronos inference
├── export_json.py                # Phase 3 — JSON export
├── runner.py                     # CLI orchestrator
├── backtest.py                   # Walk-forward backtesting
├── schema.sql                    # SQLite schema (8 tables)
├── config.yaml                   # Runtime configuration
├── requirements.txt              # Core dependencies
├── requirements-gpu.txt          # GPU/forecast dependencies
├── friday-forecast-terminal.html # Single-file dashboard
├── friday_colab.ipynb            # Google Colab notebook
├── data/                         # Exported JSON (auto-generated)
├── test_phase1.py                # Phase 1 tests (4)
├── test_phase2.py                # Phase 2 tests (7)
├── test_phase3.py                # Phase 3 tests (6)
├── test_backtest.py              # Backtest tests (6)
└── .github/workflows/
    ├── ci.yml                    # Test suite on push/PR
    ├── refresh.yml               # Daily data + forecast refresh
    └── deploy.yml                # GitHub Pages deploy
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Good first issues:**
- Add chart hover tooltips with price/date
- Mobile-responsive dashboard layout
- Support for additional markets (BSE, crypto, US equities)
- Forecast accuracy tracking over time
- Dark/light theme toggle

Check [open issues](https://github.com/ddeva6/friday/issues) for more ideas.

## Roadmap

- [ ] Multi-market support (BSE, US equities, crypto)
- [ ] Forecast accuracy dashboard (predicted vs actual)
- [ ] Portfolio watchlist with alerts
- [ ] Model comparison (Kronos vs statistical baselines)
- [ ] Real-time intraday mode
- [ ] REST API for programmatic access
- [ ] PWA support (installable on mobile)

## Data Sources

| Source | Data |
|--------|------|
| [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) | OHLCV, fundamentals (P/E, ROE, D/E, market cap) |
| [NSE India](https://www.nseindia.com/) | Index constituents |
| [Kronos (Chronos)](https://github.com/amazon-science/chronos-forecasting) | Time-series foundation model |

## License

[MIT](LICENSE) — use it, fork it, build on it.

## Disclaimer

*FRIDAY is a research and visualization tool. Forecasts are probabilistic uncertainty cones, not predictions. Nothing it outputs is investment advice. Use at your own risk.*

---

**Star this repo** if you find it useful — it helps others discover it.
