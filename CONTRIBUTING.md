# Contributing to FRIDAY

Thanks for your interest in contributing! FRIDAY is an open-source NSE equity research tool, and we welcome contributions of all kinds.

## Ways to Contribute

- **Bug reports** — found something broken? Open an issue
- **Feature requests** — have an idea? Open an issue with the `enhancement` label
- **Code contributions** — fix a bug, add a feature, improve tests
- **Documentation** — improve README, add examples, fix typos
- **Data sources** — add support for new markets or data providers

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/friday.git
cd friday
```

### 2. Install dependencies

```bash
pip install yfinance pandas pyyaml requests numpy jsonschema pytest
```

For forecast generation (requires GPU or CPU with patience):
```bash
pip install torch chronos-forecasting
```

### 3. Run tests

```bash
python -m pytest test_phase1.py test_phase2.py test_phase3.py test_backtest.py -v -m "not integration"
```

All 23 offline tests should pass before submitting a PR.

### 4. Run the pipeline locally

```bash
python runner.py          # full pipeline (needs GPU for forecasts)
python runner.py backtest # walk-forward backtest
```

## Development Guidelines

### Code Style
- Python 3.11+
- No unnecessary abstractions — keep it simple
- No comments unless the "why" is non-obvious
- Functions should do one thing

### Pull Requests
- One PR per feature/fix
- Include a clear description of what changed and why
- Add tests for new functionality
- Ensure all 23 offline tests pass
- Don't break the dashboard — test the HTML if you change it

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- First line under 72 characters
- Explain the "why" in the body if needed

## Architecture Overview

```
fetch_eod.py          Phase 1 — EOD data from yfinance + NSE
forecast_batch.py     Phase 2 — Kronos time-series forecasts
export_json.py        Phase 3 — Export DB to static JSON
runner.py             CLI orchestrator
backtest.py           Walk-forward directional backtest
schema.sql            SQLite schema (8 tables)
friday-forecast-terminal.html   Single-file dashboard
friday_colab.ipynb    Google Colab notebook
config.yaml           Runtime configuration
```

### Data Flow
```
NSE/yfinance → SQLite → Kronos model → SQLite → JSON files → GitHub Pages
```

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/ddeva6/friday/labels/good%20first%20issue) — these are scoped, well-defined tasks suitable for newcomers.

## Questions?

Open an issue or start a discussion. We're happy to help you get started.
